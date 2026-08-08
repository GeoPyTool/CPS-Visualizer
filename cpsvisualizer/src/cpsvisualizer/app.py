"""
CPS-Visualizer GUI - PySide6-based graphical interface for LA-ICP-MS
surface scan data visualization and similarity analysis.

Design follows the Web interface: a Map Viewer shows full-scan element maps
at the true data aspect ratio (every cell a strict square), datasets can be
stepped through one by one, rendered as Raw / Enhanced / Filtered, and any two
datasets can be compared with a draggable wipe divider.  The native system
theme is used (no forced dark/light), so the window follows the OS appearance.
"""
import sys
import math
import os
import warnings
import importlib.metadata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PySide6.QtGui import QAction, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFileDialog,
    QGridLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QMenu, QMessageBox, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QSplitter,
    QStyledItemDelegate, QTableView, QTabWidget,
    QToolBar, QVBoxLayout, QWidget, QApplication,
)
from PySide6.QtCore import QAbstractTableModel, Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle

from cpsvisualizer.core import (
    TRANSFORM_FUNCTIONS, DISTANCE_FUNCTIONS, DISTANCE_NAMES,
    load_data_files, apply_transforms, compute_pairwise_matrix,
    log_centering_transform, display_scale, ink_colormap, FIG_BG,
)
from cpsvisualizer.statistics import (
    compute_pca, compute_pearson_correlation_matrix,
    compute_spearman_correlation_matrix, compute_anova,
    compute_uncertainty_all, compute_descriptive_statistics_all,
)
from cpsvisualizer.metrics import (
    compute_all_image_metrics, batch_evaluate_transforms,
)
from cpsvisualizer.comparison import (
    compute_pca_embedding, compute_tsne_embedding, compute_umap_embedding,
    compute_hierarchical_clustering, compute_kmeans_clustering,
    compute_all_comparisons,
)
from cpsvisualizer.adaptive import find_optimal_power

warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 'truetype'

TRANSFORM_NAMES = list(TRANSFORM_FUNCTIONS.keys())


class CPSApplication(QApplication):
    """QApplication subclass that catches exceptions raised inside Qt
    signal/slot callbacks and shows them in a message box instead of
    letting the app crash."""

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception as e:
            import traceback
            msg = traceback.format_exc()
            print(msg, file=sys.stderr)
            try:
                box = QMessageBox()
                box.setIcon(QMessageBox.Critical)
                box.setWindowTitle('CPS-Visualizer - Error')
                box.setText(f'{type(e).__name__}: {e}')
                box.setDetailedText(msg)
                box.exec()
            except Exception:
                pass
            return False


def _subplot_grid(n):
    """Return (rows, cols) for an optimal nearly-square grid of n subplots."""
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return rows, cols


def _square_aspect(n_rows, n_cols):
    """Matplotlib aspect that renders the whole plot as a square.

    For an R x C data matrix we want the overall figure square:
    plot_width  = C * cell_w
    plot_height = R * cell_h
    Setting cell_w : cell_h = R : C  (i.e. aspect = C/R in matplotlib,
    where aspect = unit-height / unit-width) makes plot_width = plot_height.
    """
    if n_rows == 0 or n_cols == 0:
        return 1.0
    return float(n_cols) / float(n_rows)


def _equal_lims(ax, xs, ys, pad=0.10):
    """Force equal, data-spanned x/y limits so the scatter plot is square."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size == 0 or ys.size == 0:
        return
    x0, x1 = float(np.nanmin(xs)), float(np.nanmax(xs))
    y0, y1 = float(np.nanmin(ys)), float(np.nanmax(ys))
    rx = (x1 - x0) or 1.0
    ry = (y1 - y0) or 1.0
    r = max(rx, ry) * (1.0 + pad)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    ax.set_xlim(cx - r / 2.0, cx + r / 2.0)
    ax.set_ylim(cy - r / 2.0, cy + r / 2.0)


def _downsample(arr, max_rows=160, max_cols=260):
    """Vectorized block-average downsample for fast rendering."""
    r, c = arr.shape
    if r <= max_rows and c <= max_cols:
        return arr
    step_r = max(1, int(math.ceil(r / max_rows)))
    step_c = max(1, int(math.ceil(c / max_cols)))
    nr = int(math.ceil(r / step_r))
    nc = int(math.ceil(c / step_c))
    pr = nr * step_r - r
    pc = nc * step_c - c
    if pr > 0 or pc > 0:
        arr = np.pad(arr, ((0, pr), (0, pc)), mode='edge')
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    view = arr[:nr * step_r, :nc * step_c].reshape(nr, step_r, nc, step_c)
    return view.mean(axis=(1, 3))


class PandasModel(QAbstractTableModel):
    """Qt table model adapter for pandas DataFrames."""
    def __init__(self, df=None, parent=None):
        super().__init__(parent)
        self._df = df if df is not None else pd.DataFrame()

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            try:
                return str(self._df.columns[section])
            except IndexError:
                return None
        try:
            return str(self._df.index[section])
        except IndexError:
            return None

    def data(self, index, role=Qt.DisplayRole):
        if role in (Qt.DisplayRole, Qt.EditRole):
            try:
                return str(self._df.iloc[index.row(), index.column()])
            except Exception:
                pass
        return None

    def flags(self, index):
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable

    def setData(self, index, value, role=Qt.EditRole):
        row_label = self._df.index[index.row()]
        col_label = self._df.columns[index.column()]
        dtype = self._df[col_label].dtype
        if dtype != object:
            value = None if value == '' else dtype.type(value)
        self._df.at[row_label, col_label] = value
        return True

    def rowCount(self, parent=None):
        return len(self._df.index)

    def columnCount(self, parent=None):
        return len(self._df.columns)

    def sort(self, column, order):
        colname = self._df.columns[column]
        self.layoutAboutToBeChanged.emit()
        try:
            self._df.sort_values(colname, ascending=(order == Qt.AscendingOrder), inplace=True)
            self._df.reset_index(inplace=True, drop=True)
        except Exception:
            pass
        self.layoutChanged.emit()


class CustomQTableView(QTableView):
    """Table view with copy-to-clipboard context menu."""
    def __init__(self, *args):
        super().__init__(*args)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers |
                             QAbstractItemView.DoubleClicked)
        self.setSortingEnabled(True)

    def keyPressEvent(self, event):
        return

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        act = QAction("Copy", self)
        act.triggered.connect(self._copy_sel)
        menu.addAction(act)
        menu.exec_(event.globalPos())

    def _copy_sel(self):
        sel = self.selectionModel().selection().indexes()
        if not sel:
            return
        rows = sorted(i.row() for i in sel)
        cols = sorted(i.column() for i in sel)
        nr = rows[-1] - rows[0] + 1
        nc = cols[-1] - cols[0] + 1
        table = [[''] * nc for _ in range(nr)]
        for idx in sel:
            table[idx.row() - rows[0]][idx.column() - cols[0]] = idx.data()
        QGuiApplication.clipboard().setText('\n'.join('\t'.join(r) for r in table))


class AppForm(QMainWindow):
    """Reusable data viewer sub-window."""
    def __init__(self, parent=None, df=None, title='AppForm'):
        super().__init__(parent)
        self.df = df if df is not None else pd.DataFrame()
        self.file_hint = title
        self.setWindowTitle(title)
        self._build()

    def _build(self):
        self.resize(400, 600)
        frame = QWidget()
        self.table = CustomQTableView()
        self.table.setModel(PandasModel(self.df))
        btn = QPushButton('&Save')
        btn.clicked.connect(self._save)
        layout = QVBoxLayout()
        layout.addWidget(self.table)
        layout.addWidget(btn)
        frame.setLayout(layout)
        self.setCentralWidget(frame)

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Data File', self.file_hint,
            'CSV Files (*.csv);;Excel Files (*.xlsx)')
        if not path:
            return
        df = self.df.set_index('Label') if 'Label' in self.df.columns else self.df
        if 'csv' in path:
            df.to_csv(path, encoding='utf-8')
        else:
            df.to_excel(path)


class CheckBoxDelegate(QStyledItemDelegate):
    """Delegate for rendering checkboxes in list views."""
    def createEditor(self, parent, option, index):
        cb = QCheckBox(parent)
        cb.stateChanged.connect(self.commitData)
        return cb

    def setEditorData(self, editor, index):
        editor.setChecked(index.data(Qt.CheckStateRole) == Qt.Checked)

    def setModelData(self, editor, model, index):
        model.setData(index,
                      Qt.Checked if editor.isChecked() else Qt.Unchecked,
                      Qt.CheckStateRole)


class MultiSelectComboBox(QComboBox):
    """Combo box with checkable list items for multi-selection."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setPlaceholderText("Select items")
        self._list = QListWidget(self)
        self._list.setItemDelegate(CheckBoxDelegate())
        self._list.itemChanged.connect(self._update_text)
        self.setModel(self._list.model())
        self.setView(self._list)
        self._list.setMaximumHeight(200)

    def addItem(self, text):
        item = QListWidgetItem(text)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Unchecked)
        self._list.addItem(item)

    def addItems(self, texts):
        for t in texts:
            self.addItem(t)

    def _update_text(self):
        selected = [self._list.item(i).text()
                    for i in range(self._list.count())
                    if self._list.item(i).checkState() == Qt.Checked]
        self.lineEdit().setText(", ".join(selected))

    def selectedItems(self):
        return [self._list.item(i).text()
                for i in range(self._list.count())
                if self._list.item(i).checkState() == Qt.Checked]


class CPSVisualizer(QMainWindow):
    """Main application window for CPS-Visualizer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_data()
        self._init_ui()

    # ------------------------------------------------------------------
    # Data initialisation
    # ------------------------------------------------------------------
    def _init_data(self):
        self.dpi = 100
        self.df_list = []
        self.df_name_list = []
        self.result_df_dict = {}
        self.plot_flag = 'select'

        self._dist_funcs = {f.__name__: f for f in DISTANCE_FUNCTIONS}
        self._dist_names = list(self._dist_funcs.keys())
        self._trans_funcs = dict(TRANSFORM_FUNCTIONS)

        # Map Viewer state
        self._view_idx = 0
        self._render_mode = 'raw'      # raw | enhanced | filtered
        self._view_mode = 'single'     # single | wipe
        self._wipe_pos = 0.5
        self._pick_a = None
        self._pick_b = None
        self._im_top = None            # top image for wipe clipping

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _init_ui(self):
        self.setWindowTitle('CPS-Visualizer')
        self.resize(1280, 780)

        toolbar = QToolBar()
        self.addToolBar(toolbar)
        actions = [
            ('Open Data',      'Ctrl+O', self._on_open),
            ('Clear Data',     None,     self._on_clear),
            ('Calculate All',  'Ctrl+W', self._on_calc_all),
            ('Save Data',      'Ctrl+S', self._on_save),
            ('Plot All',       'Ctrl+A', self._on_plot_all),
            ('Wipe Compare',   'Ctrl+E', self._on_wipe_toggle),
            ('Save Plot',      'Ctrl+P', self._on_save_plot),
        ]
        for label, shortcut, slot in actions:
            act = QAction(label, self)
            if shortcut:
                act.setShortcut(shortcut)
            act.triggered.connect(slot)
            toolbar.addAction(act)

        # ---------------- Left sidebar ----------------
        left = QVBoxLayout()
        left.setSpacing(12)

        self._data_label = QLabel('Select Data')
        self._data_sel = QListWidget(self)
        self._data_sel.setSelectionMode(QListWidget.MultiSelection)
        self._data_sel.itemSelectionChanged.connect(self._on_data_sel_change)

        self._func_label = QLabel('Processing Pipeline')
        self._func_sel = QListWidget(self)
        self._func_sel.addItems(TRANSFORM_NAMES)
        self._func_sel.setSelectionMode(QListWidget.MultiSelection)
        self._func_sel.itemSelectionChanged.connect(self._on_transform_change)

        self._render_label = QLabel('Render Mode')
        self._render_sel = QComboBox(self)
        self._render_sel.addItems(['raw', 'enhanced', 'filtered'])
        self._render_sel.currentTextChanged.connect(self._on_render_mode_change)

        self._cmp_label = QLabel('Distance Metric')
        self._cmp_sel = QComboBox(self)
        self._cmp_sel.addItems(self._dist_names)
        self._cmp_sel.currentTextChanged.connect(self._on_cmp_changed)

        left.addWidget(self._data_label)
        left.addWidget(self._data_sel, 3)
        left.addWidget(self._func_label)
        left.addWidget(self._func_sel, 3)
        left.addWidget(self._render_label)
        left.addWidget(self._render_sel)
        left.addWidget(self._cmp_label)
        left.addWidget(self._cmp_sel)

        left_panel = QWidget()
        left_panel.setLayout(left)
        left_panel.setMinimumWidth(260)
        left_panel.setMaximumWidth(320)

        # ---------------- Right tabs ----------------
        self._tabs = QTabWidget(self)

        self._build_map_tab()
        self._build_distance_tab()
        self._build_analysis_tab()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 900])
        self.setCentralWidget(splitter)
        self.show()

    # ---------------- Map Viewer tab ----------------
    def _build_map_tab(self):
        tab = QWidget()
        v = QVBoxLayout()
        v.setSpacing(8)

        # toolbar
        tb = QHBoxLayout()
        self._map_prev = QPushButton('<')
        self._map_prev.setFixedWidth(38)
        self._map_prev.clicked.connect(lambda: self._step_map(-1))
        self._map_counter = QLabel('- / -')
        self._map_next = QPushButton('>')
        self._map_next.setFixedWidth(38)
        self._map_next.clicked.connect(lambda: self._step_map(1))
        self._map_counter.setAlignment(Qt.AlignCenter)
        self._map_counter.setMinimumWidth(90)

        self._map_mode_sel = QComboBox()
        self._map_mode_sel.addItems(['single', 'wipe'])
        self._map_mode_sel.currentTextChanged.connect(self._on_view_mode_change)

        self._pick_a = QComboBox()
        self._pick_b = QComboBox()
        self._pick_a.currentTextChanged.connect(lambda _: self._on_transform_change())
        self._pick_b.currentTextChanged.connect(lambda _: self._on_transform_change())
        self._swap_btn = QPushButton('Swap')
        self._swap_btn.clicked.connect(self._on_swap)
        self._pick_a.setMinimumWidth(110)
        self._pick_b.setMinimumWidth(110)

        tb.addWidget(self._map_prev)
        tb.addWidget(self._map_counter)
        tb.addWidget(self._map_next)
        tb.addSpacing(16)
        tb.addWidget(QLabel('Mode:'))
        tb.addWidget(self._map_mode_sel)
        tb.addSpacing(16)
        tb.addWidget(QLabel('A:'))
        tb.addWidget(self._pick_a)
        tb.addWidget(QLabel('B:'))
        tb.addWidget(self._pick_b)
        tb.addWidget(self._swap_btn)
        tb.addStretch(1)
        v.addLayout(tb)

        # map figure in a scroll area (full-scan map may be large)
        self._map_fig = Figure(figsize=(7, 7), dpi=self.dpi)
        self._map_canvas = FigureCanvas(self._map_fig)
        self._map_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._map_canvas)
        v.addWidget(scroll, 1)

        # wipe slider
        self._wipe_slider = QSlider(Qt.Horizontal)
        self._wipe_slider.setRange(0, 100)
        self._wipe_slider.setValue(50)
        self._wipe_slider.valueChanged.connect(self._on_wipe_slider)
        self._wipe_slider.setEnabled(False)
        v.addWidget(self._wipe_slider)

        tab.setLayout(v)
        self._tabs.addTab(tab, 'Map Viewer')

    # ---------------- Distance tab ----------------
    def _build_distance_tab(self):
        tab = QWidget()
        v = QVBoxLayout()
        self._dist_canvas = FigureCanvas(Figure(figsize=(12, 6), dpi=self.dpi))
        self._table = CustomQTableView()
        self._table.setMaximumHeight(160)
        table_scroll = QScrollArea()
        table_scroll.setWidgetResizable(True)
        table_scroll.setWidget(self._table)
        table_scroll.setMaximumHeight(180)
        v.addWidget(self._dist_canvas, 1)
        v.addWidget(table_scroll, 0)
        tab.setLayout(v)
        self._tabs.addTab(tab, 'Distance')

    # ---------------- Analysis tab (stats / comparison / quality / AODA) ----------------
    def _build_analysis_tab(self):
        tab = QWidget()
        self._analysis_tabs = QTabWidget()
        self._build_stats_pane(self._analysis_tabs)
        self._build_comparison_pane(self._analysis_tabs)
        self._build_quality_pane(self._analysis_tabs)
        self._build_aoda_pane(self._analysis_tabs)
        self._analysis_tabs.currentChanged.connect(self._on_analysis_tab_changed)
        v = QVBoxLayout()
        v.addWidget(self._analysis_tabs)
        tab.setLayout(v)
        self._tabs.addTab(tab, 'Analysis')

    def _scroll_pane(self):
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        inner = QWidget()
        sc.setWidget(inner)
        return sc, inner

    def _build_stats_pane(self, parent):
        sc, inner = self._scroll_pane()
        v = QVBoxLayout(inner)
        self._pca_canvas = FigureCanvas(Figure(figsize=(6, 6), dpi=self.dpi))
        self._corr_canvas = FigureCanvas(Figure(figsize=(6, 6), dpi=self.dpi))
        for c in (self._pca_canvas, self._corr_canvas):
            c.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            c.setMinimumHeight(350)
        v.addWidget(self._pca_canvas)
        v.addWidget(self._corr_canvas)
        v.addStretch(1)
        parent.addTab(sc, 'Statistics')

    def _build_comparison_pane(self, parent):
        sc, inner = self._scroll_pane()
        v = QVBoxLayout(inner)
        self._cmp_canvas = FigureCanvas(Figure(figsize=(12, 8), dpi=self.dpi))
        self._cmp_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._cmp_canvas.setMinimumHeight(600)
        v.addWidget(self._cmp_canvas)
        v.addStretch(1)
        parent.addTab(sc, 'Comparison')

    def _build_quality_pane(self, parent):
        sc, inner = self._scroll_pane()
        v = QVBoxLayout(inner)
        self._quality_canvas = FigureCanvas(Figure(figsize=(8, 5), dpi=self.dpi))
        v.addWidget(self._quality_canvas)
        v.addStretch(1)
        parent.addTab(sc, 'Image Quality')

    def _build_aoda_pane(self, parent):
        sc, inner = self._scroll_pane()
        v = QVBoxLayout(inner)
        self._aoda_table = CustomQTableView()
        self._aoda_table.setModel(PandasModel())
        v.addWidget(self._aoda_table, 1)
        v.addStretch(1)
        parent.addTab(sc, 'AODA')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _selected_data_indices(self):
        return [self.df_name_list.index(it.text())
                for it in self._data_sel.selectedItems()]

    def _selected_transform_names(self):
        return [it.text() for it in self._func_sel.selectedItems()]

    def _clean_name(self, path):
        return os.path.splitext(os.path.basename(path))[0]

    def _view_transforms(self):
        """Transforms for the map viewer: user pipeline + render-mode extras."""
        base = list(self._selected_transform_names())
        if self._render_mode == 'raw':
            return base
        if self._render_mode == 'enhanced':
            return base + ['equalize_hist', 'normalize_01']
        return base + ['sobel_gradient', 'normalize_01']

    def _apply(self, data, names):
        for name in names:
            if name in self._trans_funcs:
                try:
                    data = self._trans_funcs[name](data)
                except Exception as e:
                    QMessageBox.critical(self, 'Transform Error', str(e))
        return data

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------
    def _on_open(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, 'Open Files', '',
            'CSV Files (*.csv);;Excel Files (*.xls *.xlsx)')
        if not paths:
            return
        self._on_clear()
        for p in paths:
            df = pd.read_csv(p) if p.endswith('.csv') else pd.read_excel(p)
            self.df_list.append(df)
            self.df_name_list.append(self._clean_name(p))
        self._data_sel.addItems(self.df_name_list)
        self._fill_picks()
        self._on_cmp_changed()
        self._render_map()

    def _on_clear(self):
        self.df_list.clear()
        self.df_name_list.clear()
        self.result_df_dict.clear()
        self._table.setModel(PandasModel())
        self._data_sel.clear()
        self._pick_a.clear()
        self._pick_b.clear()
        self._map_counter.setText('- / -')
        for fig in self._all_figs():
            fig.clear()
        self._redraw_all()
        self.plot_flag = 'select'

    def _all_figs(self):
        return [self._map_fig, self._dist_canvas.figure,
                self._pca_canvas.figure, self._corr_canvas.figure,
                self._cmp_canvas.figure, self._quality_canvas.figure]

    def _redraw_all(self):
        for c in (self._map_canvas, self._dist_canvas, self._pca_canvas,
                  self._corr_canvas, self._cmp_canvas,
                  self._quality_canvas):
            c.draw()

    def _on_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Data File', 'DataFileOutput',
            'CSV Files (*.csv);;Excel Files (*.xlsx)')
        if path:
            df = self.result_df_dict.get(self._cmp_sel.currentText(), pd.DataFrame())
            if 'csv' in path:
                df.to_csv(path, encoding='utf-8')
            else:
                df.to_excel(path)

    def _on_calc_all(self):
        directory = QFileDialog.getExistingDirectory(None, 'Select Directory')
        if not directory:
            return
        for func in DISTANCE_FUNCTIONS:
            name = func.__name__
            if name not in self.result_df_dict:
                self.result_df_dict[name] = compute_pairwise_matrix(
                    self.df_list, self.df_name_list, func)
            path = os.path.join(directory, f'result_{name}.csv')
            try:
                self.result_df_dict[name].to_csv(path, encoding='utf-8')
            except Exception:
                pass

    def _on_cmp_changed(self):
        name = self._cmp_sel.currentText()
        if not name or name not in self._dist_funcs:
            return
        if not self.df_list:
            return
        if name not in self.result_df_dict:
            self.result_df_dict[name] = compute_pairwise_matrix(
                self.df_list, self.df_name_list, self._dist_funcs[name])
        self._table.setModel(PandasModel(self.result_df_dict[name].round(4)))
        self._plot_distance_grid()

    DIST_GRID = ['Euclidean', 'Manhattan', 'Chebyshev', 'Minkowski',
                 'Cosine', 'Correlation', 'Hsim_Distance', 'Close_Distance',
                 'Bray_Curtis', 'Canberra']

    def _plot_distance_grid(self):
        """Render 8 representative distance metrics as a 2x4 grid of
        0-1 normalized heatmaps, with a narrow vertical colorbar in its
        own column (col 5) so heatmaps stay evenly distributed."""
        fig = self._dist_canvas.figure
        fig.clear()
        fig.patch.set_facecolor(FIG_BG)
        if not self.df_list:
            ax = fig.add_subplot(111)
            ax.set_facecolor(FIG_BG)
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            fig.tight_layout()
            self._dist_canvas.draw()
            return
        names = self.df_name_list
        n = len(names)
        all_metrics = [m for m in self.DIST_GRID if m in self._dist_funcs]
        nrows, ncols = 2, 4
        metrics = all_metrics[:nrows * ncols]
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(nrows, ncols + 1, figure=fig,
                      width_ratios=[1, 1, 1, 1, 0.06],
                      wspace=0.25, hspace=0.30,
                      left=0.06, right=0.97, top=0.95, bottom=0.12)
        images = []
        for pos, mname in enumerate(metrics):
            if mname not in self.result_df_dict:
                self.result_df_dict[mname] = compute_pairwise_matrix(
                    self.df_list, names, self._dist_funcs[mname])
            df = self.result_df_dict[mname]
            vals = df.values.astype(float)
            if n > 1:
                off = vals[~np.eye(n, dtype=bool)]
                lo, hi = float(off.min()), float(off.max())
                if hi - lo > 1e-12:
                    vals = (vals - lo) / (hi - lo)
                    np.fill_diagonal(vals, 0.0)
                # log-stretch so small differences spread across more
                # of the grayscale range for finer discrimination
                vals = np.log1p(vals * 255.0) / np.log1p(255.0)
            row, col = pos // ncols, pos % ncols
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor(FIG_BG)
            im = ax.imshow(vals, cmap=ink_colormap(), vmin=0.0, vmax=1.0,
                           aspect='equal')
            images.append(im)
            ax.set_title(mname, fontsize=9)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            if row == nrows - 1:
                ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_yticklabels(names, fontsize=7)
            else:
                ax.set_yticklabels([])
        # vertical colorbar in the narrow 5th column, spanning both rows
        cax = fig.add_subplot(gs[:, ncols])
        if images:
            import matplotlib as _mpl
            norm = _mpl.colors.Normalize(vmin=0.0, vmax=1.0)
            sm = _mpl.cm.ScalarMappable(cmap=ink_colormap(), norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.ax.tick_params(labelsize=7)
            # put label on the left side of the bar so it is not clipped
            cbar.ax.yaxis.set_label_position('left')
            cbar.ax.yaxis.set_ticks_position('right')
            cbar.set_label('normalized distance (0-1)', fontsize=8)
        else:
            cax.axis('off')
        self._dist_canvas.draw()

    def _plot_distance_heatmap(self, name):
        if name not in self.result_df_dict:
            return
        df = self.result_df_dict[name]
        fig = self._dist_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor(FIG_BG)
        ax.set_facecolor(FIG_BG)
        if df.values.size == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            fig.tight_layout()
            self._dist_canvas.draw()
            return
        vals = df.values.astype(float)
        n = len(vals)
        if n > 1:
            off = vals[~np.eye(n, dtype=bool)]
            lo, hi = float(off.min()), float(off.max())
            if hi - lo > 1e-12:
                vals = (vals - lo) / (hi - lo)
                np.fill_diagonal(vals, 0.0)
        im = ax.imshow(vals, cmap=ink_colormap(), vmin=0.0, vmax=1.0, aspect='equal')
        ax.set_xticks(range(len(df.columns)))
        ax.set_yticks(range(len(df.index)))
        ax.set_xticklabels(df.columns, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(df.index, fontsize=8)
        ax.set_title(f'{name}  (normalized 0-1)')
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label('normalized distance')
        fig.tight_layout()
        self._dist_canvas.draw()

    # ------------------------------------------------------------------
    # Map Viewer
    # ------------------------------------------------------------------
    def _fill_picks(self):
        self._pick_a.clear()
        self._pick_b.clear()
        self._pick_a.addItems(self.df_name_list)
        self._pick_b.addItems(self.df_name_list)
        if len(self.df_name_list) >= 2:
            self._pick_b.setCurrentIndex(1)

    def _on_data_sel_change(self):
        # selecting datasets updates pickers
        self._on_cmp_changed()
        self._render_map()

    def _step_map(self, delta):
        n = len(self.df_name_list)
        if n == 0:
            return
        self._view_idx = (self._view_idx + delta) % n
        self._data_sel.setCurrentRow(self._view_idx)
        self._render_map()

    def _render_map(self):
        if not self.df_name_list:
            self._map_counter.setText('- / -')
            return
        self._view_idx = min(max(self._view_idx, 0), len(self.df_name_list) - 1)
        self._map_counter.setText(f'{self._view_idx + 1} / {len(self.df_name_list)}')

        transforms = self._view_transforms()
        fig = self._map_fig
        fig.clear()

        if self._view_mode == 'single':
            self._render_single_map(fig, transforms)
        else:
            self._render_wipe_map(fig, transforms)

        fig.tight_layout()
        self._map_canvas.draw()

    def _render_single_map(self, fig, transforms):
        idx = self._view_idx
        name = self.df_name_list[idx]
        data = self._apply(self.df_list[idx].to_numpy(), transforms)
        data = _downsample(data)
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor(FIG_BG)
        ax.set_facecolor(FIG_BG)
        if self._render_mode == 'raw':
            s, lo, hi = display_scale(data)
        else:
            s, lo, hi = display_scale(data, 0.0, 100.0)
        im = ax.imshow(s, cmap=ink_colormap(), vmin=lo, vmax=hi,
                       aspect='auto', interpolation='nearest')
        ax.set_title(f'{name}  |  {self._render_mode}'
                     + (f'  |  {"+".join(transforms)}' if transforms else ''))
        ax.set_xlabel('column')
        ax.set_ylabel('row')
        # force the whole plot to be a clean square, centered in the figure
        ax.set_aspect(_square_aspect(*data.shape), adjustable='box', anchor='C')
        fig.colorbar(im, ax=ax, shrink=0.8)
        self._wipe_slider.setEnabled(False)

    def _render_wipe_map(self, fig, transforms):
        a_name = self._pick_a.currentText() or (self.df_name_list[0] if self.df_name_list else None)
        b_name = self._pick_b.currentText() or a_name
        if a_name not in self.df_name_list or b_name not in self.df_name_list:
            return
        a_idx = self.df_name_list.index(a_name)
        b_idx = self.df_name_list.index(b_name)
        data_a = self._apply(self.df_list[a_idx].to_numpy(), transforms)
        data_b = self._apply(self.df_list[b_idx].to_numpy(), transforms)
        min_r = min(data_a.shape[0], data_b.shape[0])
        min_c = min(data_a.shape[1], data_b.shape[1])
        data_a = _downsample(data_a[:min_r, :min_c])
        data_b = _downsample(data_b[:min_r, :min_c])

        ax = fig.add_subplot(111)
        fig.patch.set_facecolor(FIG_BG)
        ax.set_facecolor(FIG_BG)
        # Web-style wipe: two stacked images.
        # B (transparent ink) fills the axes as the base layer.
        sb, lb, hb = display_scale(data_b)
        ax.imshow(sb, cmap=ink_colormap(), vmin=lb, vmax=hb,
                  aspect='auto', interpolation='nearest')
        # A (opaque ink) is drawn on top and CLIPPED to the LEFT of the
        # divider, so the left side shows A and the right side shows B.
        sa, la, ha = display_scale(data_a)
        self._im_a = ax.imshow(sa, cmap=ink_colormap(opaque_min=True),
                               vmin=la, vmax=ha, aspect='auto',
                               interpolation='nearest')
        self._im_b = None
        self._wipe_ax = ax
        self._wipe_a_name = a_name
        self._wipe_b_name = b_name
        ax.set_title(f'Wipe compare: {a_name} vs {b_name}')
        ax.set_xlabel('column')
        ax.set_ylabel('row')
        ax.set_aspect(_square_aspect(*data_a.shape), adjustable='box', anchor='C')
        self._wipe_slider.setEnabled(True)
        self._on_wipe_slider(self._wipe_slider.value())

    def _on_wipe_slider(self, val):
        """Clip image A (top) to the LEFT of the divider; B (bottom) shows right."""
        self._wipe_pos = val / 100.0
        if not hasattr(self, '_im_a') or self._im_a is None:
            return
        ax = self._im_a.axes
        xlim = ax.get_xlim()
        # A is visible only from xmin up to the divider position
        clip_x = xlim[0] + (xlim[1] - xlim[0]) * self._wipe_pos
        half_span = (xlim[1] - xlim[0]) * 2.0
        rect = Rectangle((xlim[0] - 1.0, -half_span), clip_x - (xlim[0] - 1.0), 2 * half_span,
                         transform=ax.transData, facecolor='none')
        self._im_a.set_clip_path(rect)
        self._map_canvas.draw()

    def _on_view_mode_change(self, mode):
        self._view_mode = mode
        self._render_map()

    def _on_wipe_toggle(self):
        # toolbar shortcut toggles wipe mode
        if self._view_mode == 'wipe':
            self._map_mode_sel.setCurrentIndex(0)
        else:
            self._map_mode_sel.setCurrentIndex(1)

    def _on_swap(self):
        a = self._pick_a.currentText()
        b = self._pick_b.currentText()
        if not a or not b:
            return
        ia = self._pick_a.findText(b)
        ib = self._pick_b.findText(a)
        if ia >= 0:
            self._pick_a.setCurrentIndex(ia)
        if ib >= 0:
            self._pick_b.setCurrentIndex(ib)

    # ------------------------------------------------------------------
    # Transforms + plotting
    # ------------------------------------------------------------------
    def _on_render_mode_change(self, mode):
        self._render_mode = mode
        self._on_transform_change()

    def _on_transform_change(self):
        if not self.df_name_list:
            return
        self._render_map()

    def _on_plot_all(self):
        self.plot_flag = 'all'
        if not self.df_name_list:
            return
        n = len(self.df_name_list)
        fig = self._map_fig
        fig.clear()
        rows, cols = _subplot_grid(n)
        transforms = self._view_transforms()
        fig.patch.set_facecolor(FIG_BG)
        for pos, (name, df) in enumerate(zip(self.df_name_list, self.df_list), 1):
            ax = fig.add_subplot(rows, cols, pos)
            ax.set_facecolor(FIG_BG)
            data = _downsample(self._apply(df.to_numpy(), transforms))
            if self._render_mode == 'raw':
                s, lo, hi = display_scale(data)
            else:
                s, lo, hi = display_scale(data, 0.0, 100.0)
            ax.imshow(s, cmap=ink_colormap(), vmin=lo, vmax=hi,
                      aspect='auto', interpolation='nearest')
            ax.set_aspect(_square_aspect(*data.shape), adjustable='box', anchor='C')
            ax.set_title(name, fontsize=9)
        fig.tight_layout()
        self._map_canvas.draw()

    def _on_save_plot(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Plot', 'CPS_Image',
            'PNG (*.png);;JPEG (*.jpg *.jpeg);;SVG (*.svg);;PDF (*.pdf)')
        if path:
            dpi = self.dpi * 6 if path.lower().endswith(('.png', '.jpg', '.jpeg')) else None
            self._map_canvas.figure.savefig(path, dpi=dpi)

    # ------------------------------------------------------------------
    # Analysis panes
    # ------------------------------------------------------------------
    def _analysis_selected(self, tab_idx):
        """Render the chosen analysis pane on demand."""
        names = [self.df_name_list[i] for i in self._selected_data_indices()]
        if not names:
            names = self.df_name_list
        if not names:
            return
        dfs = [pd.DataFrame(_downsample(self.df_list[self.df_name_list.index(n)].to_numpy(),
                                        80, 120)) for n in names]
        transforms = self._selected_transform_names()
        if transforms:
            dfs = [pd.DataFrame(self._apply(d.to_numpy(), transforms)) for d in dfs]

        if tab_idx == 0:      # Statistics
            self._render_stats(dfs, names)
        elif tab_idx == 1:    # Comparison
            self._render_comparison(dfs, names)
        elif tab_idx == 2:    # Image Quality
            self._render_quality(dfs, names)
        elif tab_idx == 3:    # AODA
            self._render_aoda(dfs, names)

    def _on_analysis_tab_changed(self, idx):
        self._analysis_selected(idx)

    def _render_stats(self, dfs, names):
        fig = self._pca_canvas.figure
        fig.clear()
        fig.patch.set_facecolor(FIG_BG)
        try:
            pca = compute_pca(dfs, names)
            ax = fig.add_subplot(111)
            coords = pca['coords']
            ev = pca['explained_variance']
            ax.scatter(coords[:, 0], coords[:, 1], s=80, c='#4060d8',
                       edgecolors='k', linewidths=0.5)
            for i, n in enumerate(names):
                ax.annotate(n, (coords[i, 0], coords[i, 1]),
                            textcoords="offset points", xytext=(4, 4), fontsize=9)
            ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)' if ev else 'PC1')
            ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)' if len(ev) > 1 else 'PC2')
            ax.set_title('PCA: Element Distribution Patterns')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_aspect('equal', adjustable='box')
            _equal_lims(ax, coords[:, 0], coords[:, 1])
        except Exception as e:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f'PCA error: {e}', ha='center', va='center')
        fig.tight_layout()
        self._pca_canvas.draw()

        fig = self._corr_canvas.figure
        fig.clear()
        fig.patch.set_facecolor(FIG_BG)
        try:
            corr, _ = compute_pearson_correlation_matrix(dfs, names)
            ax = fig.add_subplot(111)
            im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
            ax.set_xticks(range(len(names)))
            ax.set_yticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(names, fontsize=8)
            ax.set_title('Pearson Correlation Matrix')
            fig.colorbar(im, ax=ax, shrink=0.85)
        except Exception as e:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f'Correlation error: {e}', ha='center', va='center')
        fig.tight_layout()
        self._corr_canvas.draw()

    def _render_comparison(self, dfs, names):
        from matplotlib.gridspec import GridSpec
        fig = self._cmp_canvas.figure
        fig.clear()
        fig.patch.set_facecolor(FIG_BG)
        # 5 square subplots: top row 3, bottom row 2 (centred)
        # Use a 2x6 grid; top row uses cols 0-1, 2-3, 4-5; bottom row
        # uses cols 1-2, 3-4 (offset by 1 to centre).
        gs = GridSpec(2, 6, figure=fig,
                      width_ratios=[1, 1, 1, 1, 1, 1],
                      wspace=0.35, hspace=0.35,
                      left=0.05, right=0.98, top=0.95, bottom=0.07)

        def _draw_scatter(ax, fn, title):
            ax.set_facecolor(FIG_BG)
            try:
                r = fn()
                if isinstance(r, dict) and r.get('error'):
                    ax.text(0.5, 0.5, str(r['error']), ha='center', va='center')
                elif 'embedding' in r:
                    emb = np.asarray(r['embedding'])
                    if emb.ndim == 2 and emb.shape[1] >= 2:
                        ax.scatter(emb[:, 0], emb[:, 1], s=80, c='#4060d8',
                                   edgecolors='k', linewidths=0.5)
                        for i, n in enumerate(names):
                            ax.annotate(n, (emb[i, 0], emb[i, 1]),
                                        textcoords="offset points",
                                        xytext=(4, 4), fontsize=8)
                        ax.set_aspect('equal', adjustable='box')
                        _equal_lims(ax, emb[:, 0], emb[:, 1])
                ax.set_title(title, fontsize=10)
                ax.grid(True, alpha=0.3, linestyle='--')
            except Exception as e:
                ax.text(0.5, 0.5, str(e), ha='center', va='center')

        # top row: PCA (cols 0-1), t-SNE (cols 2-3), UMAP (cols 4-5)
        ax_pca = fig.add_subplot(gs[0, 0:2])
        ax_tsne = fig.add_subplot(gs[0, 2:4])
        ax_umap = fig.add_subplot(gs[0, 4:6])
        # bottom row: Dendrogram (cols 1-2), K-Means (cols 3-4) — centred
        ax_dendro = fig.add_subplot(gs[1, 1:3])
        ax_kmeans = fig.add_subplot(gs[1, 3:5])

        for ax in (ax_pca, ax_tsne, ax_umap, ax_dendro, ax_kmeans):
            ax.set_facecolor(FIG_BG)

        _draw_scatter(ax_pca,
                      lambda: compute_pca_embedding(dfs, names), 'PCA')
        _draw_scatter(ax_tsne,
                      lambda: compute_tsne_embedding(
                          dfs, names, perplexity=min(5, len(dfs) - 1)),
                      't-SNE')
        _draw_scatter(ax_umap,
                      lambda: compute_umap_embedding(dfs, names), 'UMAP')

        # dendrogram
        try:
            from scipy.cluster.hierarchy import dendrogram
            hier = compute_hierarchical_clustering(dfs, names)
            if hier.get('linkage') is not None:
                dendrogram(hier['linkage'], labels=names, ax=ax_dendro,
                           leaf_rotation=45, leaf_font_size=8)
                ax_dendro.set_title(
                    f"Dendrogram (r={hier['cophenetic_correlation']:.3f})",
                    fontsize=10)
            else:
                ax_dendro.text(0.5, 0.5, 'Need >=2 samples',
                               ha='center', va='center')
        except Exception as e:
            ax_dendro.text(0.5, 0.5, str(e), ha='center', va='center')

        # k-means
        try:
            km = compute_kmeans_clustering(dfs, names,
                                            n_clusters=min(3, len(dfs)))
            pca_emb = compute_pca_embedding(dfs, names)['embedding']
            colors = ['#4060d8', '#ed7d31', '#1a9e5c', '#c78a0a', '#d64545']
            for k in set(km['labels']):
                mask = np.array(km['labels']) == k
                ax_kmeans.scatter(pca_emb[mask, 0], pca_emb[mask, 1], s=80,
                                  c=colors[k % len(colors)],
                                  label=f'Cluster {k + 1}',
                                  edgecolors='k', linewidths=0.5)
                for i, n in enumerate(names):
                    if mask[i]:
                        ax_kmeans.annotate(n, (pca_emb[i, 0], pca_emb[i, 1]),
                                           textcoords="offset points",
                                           xytext=(4, 4), fontsize=8)
            ax_kmeans.legend(fontsize=7)
            ax_kmeans.set_title(f"K-Means (k={km['n_clusters']})", fontsize=10)
            ax_kmeans.set_aspect('equal', adjustable='box')
            _equal_lims(ax_kmeans, pca_emb[:, 0], pca_emb[:, 1])
            ax_kmeans.grid(True, alpha=0.3, linestyle='--')
        except Exception as e:
            ax_kmeans.text(0.5, 0.5, str(e), ha='center', va='center')

        self._cmp_canvas.draw()

    def _render_quality(self, dfs, names):
        fig = self._quality_canvas.figure
        fig.clear()
        fig.patch.set_facecolor(FIG_BG)
        ax = fig.add_subplot(111)
        try:
            results = batch_evaluate_transforms(dfs, names, self._trans_funcs)
            tnames = list(results.keys())
            if not tnames:
                ax.text(0.5, 0.5, 'No metrics', ha='center', va='center')
            else:
                psnr = [np.mean([m.get('psnr', 0) for m in results[t].values()]) for t in tnames]
                entropy = [np.mean([m['entropy_transformed']['normalized_entropy']
                                    for m in results[t].values() if 'error' not in m])
                           for t in tnames]
                cei = [np.mean([m.get('cei', 0) for m in results[t].values()]) for t in tnames]
                x = np.arange(len(tnames))
                w = 0.25
                ax.bar(x - w, psnr, w, label='PSNR')
                ax.bar(x, entropy, w, label='Entropy')
                ax.bar(x + w, cei, w, label='CEI')
                ax.set_xticks(x)
                ax.set_xticklabels(tnames, rotation=45, ha='right', fontsize=8)
                ax.legend(fontsize=8)
                ax.set_title('Image Quality Metrics by Transform')
        except Exception as e:
            ax.text(0.5, 0.5, str(e), ha='center', va='center')
        fig.tight_layout()
        self._quality_canvas.draw()

    def _render_aoda(self, dfs, names):
        if len(dfs) < 3:
            self._aoda_table.setModel(PandasModel())
            return
        try:
            # heavy pipeline search: use a compact downsample + metric pre-screen
            compact = [pd.DataFrame(_downsample(d.to_numpy(), 20, 30)) for d in dfs]
            result = find_optimal_power(compact, names, n_jobs=1, verbose=False, top_n=10)
            cols = ['method', 'power', 'dps', 'winner', 'n_evaluations', 'rank']
            self._aoda_table.setModel(PandasModel(result[cols].round(4)))
        except Exception:
            self._aoda_table.setModel(PandasModel())


def _install_excepthook(app_ref=None):
    """Install a global exception hook that shows a message box instead of
    crashing.  Catches unhandled exceptions in Qt slots (which Qt otherwise
    swallows or aborts on) and in the main thread."""
    def _hook(exc_type, exc_value, exc_tb):
        import traceback
        msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(msg, file=sys.stderr)
        try:
            from PySide6.QtWidgets import QMessageBox
            box = QMessageBox()
            box.setIcon(QMessageBox.Critical)
            box.setWindowTitle('CPS-Visualizer - Error')
            box.setText(f'{exc_type.__name__}: {exc_value}')
            box.setDetailedText(msg)
            box.exec()
        except Exception:
            pass  # GUI not available (headless / shutting down) - already logged
    sys.excepthook = _hook


def main():
    try:
        metadata = importlib.metadata.metadata(
            sys.modules["__main__"].__package__)
        QApplication.setApplicationName(metadata.get("Formal-Name",
                    'CPS-Visualizer'))
    except Exception:
        QApplication.setApplicationName('CPS-Visualizer')
    try:
        app = CPSApplication(sys.argv)
    except RuntimeError:
        QApplication.instance().quit()
        app = CPSApplication(sys.argv)
    _install_excepthook(app)
    try:
        window = CPSVisualizer()
    except Exception as e:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(None, 'CPS-Visualizer - Startup Error',
                             f'Failed to start:\n{e}')
        sys.exit(1)
    sys.exit(app.exec())


if __name__ == '__main__':
    try:
        app = CPSApplication(sys.argv)
    except RuntimeError:
        QApplication.instance().quit()
        app = CPSApplication(sys.argv)
    _install_excepthook(app)
    try:
        window = CPSVisualizer()
    except Exception as e:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(None, 'CPS-Visualizer - Startup Error',
                             f'Failed to start:\n{e}')
        sys.exit(1)
    window.show()
    sys.exit(app.exec())
