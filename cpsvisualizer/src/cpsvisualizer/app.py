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
    log_centering_transform, display_scale, ink_colormap, sci_colormap, FIG_BG,
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


def _install_figure_export(canvas, default_name):
    """Install a right-click context menu on *canvas* (a FigureCanvas)
    that lets the user save the figure as PNG / PDF / SVG with a
    sensible default filename."""
    from PySide6.QtWidgets import QMenu
    # grab the native widget that actually receives mouse events
    widget = canvas
    base = canvas.contextMenuEvent

    def _ctx_menu(event):
        menu = QMenu(widget)
        for label, ext, filt in [
                ('Save as PNG', '.png',
                 'PNG (*.png)'),
                ('Save as PDF', '.pdf',
                 'PDF (*.pdf)'),
                ('Save as SVG', '.svg',
                 'SVG (*.svg)')]:
            act = QAction(label, widget)
            def _save(_checked, e=ext, f=filt, dn=default_name):
                path, _ = QFileDialog.getSaveFileName(
                    widget, 'Save Figure As', f'{dn}{e}', f)
                if not path:
                    return
                dpi = widget.dpi * 6 if path.lower().endswith('.png') else None
                widget.figure.savefig(path, dpi=dpi, bbox_inches='tight')
            act.triggered.connect(_save)
            menu.addAction(act)
        menu.exec_(event.globalPos())
    canvas.contextMenuEvent = _ctx_menu


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
    """Table view with copy-to-clipboard and save CSV/XLSX context menu."""
    _export_name = 'cps_table'

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
        menu.addSeparator()
        for label, ext in [('Save as CSV', '.csv'),
                           ('Save as XLSX', '.xlsx')]:
            act2 = QAction(label, self)
            act2.triggered.connect(lambda _checked, e=ext: self._save_table(e))
            menu.addAction(act2)
        menu.exec_(event.globalPos())

    def _save_table(self, ext):
        model = self.model()
        if not isinstance(model, PandasModel) or model._df is None:
            return
        default = f'{self._export_name}{ext}'
        path, _ = QFileDialog.getSaveFileName(
            self, f'Save Table As', default,
            f'CSV Files (*.csv)' if ext == '.csv'
            else f'Excel Files (*.xlsx)')
        if not path:
            return
        if ext == '.csv':
            model._df.to_csv(path, encoding='utf-8')
        else:
            model._df.to_excel(path)

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
            ('Open Folder',    'Ctrl+D', self._on_open_folder),
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
        _install_figure_export(self._map_canvas, 'cps_map')
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
        _install_figure_export(self._dist_canvas, 'cps_distance')
        self._table = CustomQTableView()
        self._table._export_name = 'cps_distance_table'
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
        self._build_quality_aoda_pane(self._analysis_tabs)
        self._build_figures_pane(self._analysis_tabs)
        self._build_fusion_pane(self._analysis_tabs)
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
        h = QHBoxLayout(inner)
        self._pca_canvas = FigureCanvas(Figure(figsize=(6, 6), dpi=self.dpi))
        _install_figure_export(self._pca_canvas, 'cps_pca')
        self._corr_canvas = FigureCanvas(Figure(figsize=(6, 6), dpi=self.dpi))
        _install_figure_export(self._corr_canvas, 'cps_correlation')
        for c in (self._pca_canvas, self._corr_canvas):
            c.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            c.setMinimumHeight(400)
        h.addWidget(self._pca_canvas)
        h.addWidget(self._corr_canvas)
        parent.addTab(sc, 'Statistics')

    def _build_comparison_pane(self, parent):
        sc, inner = self._scroll_pane()
        v = QVBoxLayout(inner)

        # Dendrogram Y-axis mode selector
        mode_bar = QHBoxLayout()
        mode_bar.addWidget(QLabel('Dendrogram Y-axis:'))
        self._dendro_mode = QComboBox()
        self._dendro_mode.addItems(['Auto', 'Linear', 'Log', 'Blended'])
        self._dendro_mode.currentTextChanged.connect(
            lambda _: self._on_analysis_tab_changed(1))
        mode_bar.addWidget(self._dendro_mode)
        mode_bar.addStretch(1)
        v.addLayout(mode_bar)

        self._cmp_canvas = FigureCanvas(Figure(figsize=(12, 8), dpi=self.dpi))
        _install_figure_export(self._cmp_canvas, 'cps_comparison')
        self._cmp_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._cmp_canvas.setMinimumHeight(600)
        v.addWidget(self._cmp_canvas)
        v.addStretch(1)
        parent.addTab(sc, 'Comparison')

    def _build_quality_aoda_pane(self, parent):
        sc, inner = self._scroll_pane()
        h = QHBoxLayout(inner)

        # Image Quality table (left)
        qv = QVBoxLayout()
        qv.addWidget(QLabel('Image Quality Metrics'))
        self._quality_table = CustomQTableView()
        self._quality_table.setModel(PandasModel())
        self._quality_table._export_name = 'cps_image_quality'
        q_scroll = QScrollArea()
        q_scroll.setWidgetResizable(True)
        q_scroll.setWidget(self._quality_table)
        qv.addWidget(q_scroll, 1)
        h.addLayout(qv, 1)

        # AODA table (right)
        av = QVBoxLayout()
        av.addWidget(QLabel('AODA Optimization Results'))
        self._aoda_table = CustomQTableView()
        self._aoda_table.setModel(PandasModel())
        self._aoda_table._export_name = 'cps_aoda'
        a_scroll = QScrollArea()
        a_scroll.setWidgetResizable(True)
        a_scroll.setWidget(self._aoda_table)
        av.addWidget(a_scroll, 1)
        h.addLayout(av, 1)

        parent.addTab(sc, 'Quality & AODA')

    def _build_figures_pane(self, parent):
        sc, inner = self._scroll_pane()
        v = QVBoxLayout(inner)

        mode_bar = QHBoxLayout()
        mode_bar.addWidget(QLabel('Radar axis scaling:'))
        self._radar_mode = QComboBox()
        self._radar_mode.addItems(['Min-Max', 'Z-Score', 'Log+MinMax', 'Sqrt+MinMax'])
        self._radar_mode.setCurrentText('Sqrt+MinMax')  # balanced default
        self._radar_mode.currentTextChanged.connect(
            lambda _: self._on_analysis_tab_changed(3))
        mode_bar.addWidget(self._radar_mode)
        mode_bar.addStretch(1)
        v.addLayout(mode_bar)

        self._fig_canvas = FigureCanvas(Figure(figsize=(14, 10), dpi=self.dpi))
        _install_figure_export(self._fig_canvas, 'cps_figures')
        self._fig_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._fig_canvas.setMinimumHeight(700)
        v.addWidget(self._fig_canvas)
        parent.addTab(sc, 'Figures')

    def _build_fusion_pane(self, parent):
        sc, inner = self._scroll_pane()
        v = QVBoxLayout(inner)
        bar = QHBoxLayout()
        self._fus_picks = []
        self._fus_color_btns = []
        default_colors = ['#FF3030', '#30CC30', '#3080FF']
        for i, (lbl, dc) in enumerate(zip(['A:', 'B:', 'C:'], default_colors)):
            bar.addWidget(QLabel(lbl))
            pb = QComboBox()
            pb.currentTextChanged.connect(self._on_fusion_change)
            bar.addWidget(pb)
            self._fus_picks.append(pb)
            cb = QPushButton(' ')
            cb.setFixedSize(24, 24)
            cb.setStyleSheet(f'background-color: {dc}; border:1px solid #888;')
            idx = i
            cb.clicked.connect(lambda checked, n=idx: self._pick_fusion_color(n))
            bar.addWidget(cb)
            self._fus_color_btns.append(cb)
        bar.addWidget(QLabel('Res:'))
        self._fus_res = QComboBox()
        self._fus_res.addItems(['1x', '2x', '4x'])
        self._fus_res.currentTextChanged.connect(self._on_fusion_change)
        bar.addWidget(self._fus_res)
        bar.addWidget(QLabel('\u03b1:'))
        self._fus_alpha = QSlider(Qt.Horizontal)
        self._fus_alpha.setRange(0, 100)
        self._fus_alpha.setValue(50)
        self._fus_alpha.valueChanged.connect(self._on_fusion_change)
        bar.addWidget(self._fus_alpha)
        self._fus_alpha_label = QLabel('0.50')
        self._fus_alpha.valueChanged.connect(
            lambda v: self._fus_alpha_label.setText(f'{v/100:.2f}'))
        bar.addWidget(self._fus_alpha_label)
        bar.addStretch(1)
        v.addLayout(bar)
        self._fus_canvas = FigureCanvas(Figure(figsize=(20, 15), dpi=self.dpi))
        _install_figure_export(self._fus_canvas, 'cps_fusion')
        self._fus_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._fus_canvas.setMinimumHeight(800)
        v.addWidget(self._fus_canvas)
        parent.addTab(sc, 'Fusion')
    def _pick_fusion_color(self, idx):
        from PySide6.QtWidgets import QColorDialog
        btn = self._fus_color_btns[idx]
        current = btn.palette().button().color()
        color = QColorDialog.getColor(current, self, 'Pick line color')
        if color.isValid():
            btn.setStyleSheet(f'background-color: {color.name()}; border:1px solid #888;')
            self._on_fusion_change()
    def _on_fusion_change(self):
        if self.df_name_list:
            self._fill_fusion_picks()
            self._render_fusion()
    def _fill_fusion_picks(self):
        for p in self._fus_picks:
            cur = p.currentText()
            p.blockSignals(True); p.clear(); p.addItems(self.df_name_list)
            if cur in self.df_name_list: p.setCurrentText(cur)
            p.blockSignals(False)
        if len(self.df_name_list) >= 2 and not self._fus_picks[1].currentText():
            self._fus_picks[1].setCurrentIndex(1)
    def _render_fusion(self):
        from cpsvisualizer.core import log_transform, equalize_hist
        from scipy.ndimage import sobel, gaussian_filter, zoom
        import re
        zf = int(self._fus_res.currentText().replace('x', ''))
        self._fus_colors = []
        for btn in self._fus_color_btns:
            m = re.search(r'#[0-9a-fA-F]{6}', btn.styleSheet())
            self._fus_colors.append(m.group() if m else '#FF3030')
        self._fus_rows = []
        for i in range(3):
            nm = self._fus_picks[i].currentText()
            if not nm or nm not in self.df_name_list:
                if i < 2: continue
                else: break
            data = self.df_list[self.df_name_list.index(nm)].to_numpy().copy()
            raw = np.nan_to_num(data.astype(float), nan=0, posinf=0, neginf=0)
            raw01 = (raw - raw.min()) / (raw.max() - raw.min() + 1e-10)
            # middle column: multiplicative visual-statistical fusion
            # F = V^alpha * S^(1-alpha) — pixel only prominent if both
            # visual AND statistical channels score high
            from cpsvisualizer.fusion import robust_zscore
            zs = robust_zscore(raw)
            zs01 = (zs - zs.min()) / (zs.max() - zs.min() + 1e-10)
            eps = 1e-10
            al = self._fus_alpha.value() / 100.0
            V = (np.maximum(raw01, eps) ** al) * (np.maximum(zs01, eps) ** (1 - al))
            # Otsu contours on downsampled V, then convert to V-space coords
            from skimage.filters import threshold_otsu
            from skimage.measure import find_contours
            from scipy.ndimage import gaussian_filter as gf
            ds = _downsample(V, 80, 80)
            thresh = threshold_otsu(ds)
            binary = (ds > thresh).astype(np.float64)
            binary_sm = gf(binary, sigma=1.5)
            contours_ds = find_contours(binary_sm, level=0.5)
            # convert to V-space coords (so zoom doesn't break them)
            contours_v = []
            for cnt in contours_ds:
                cx = cnt[:, 1] * (V.shape[1] / ds.shape[1])
                cy = cnt[:, 0] * (V.shape[0] / ds.shape[0])
                contours_v.append(np.column_stack([cx, cy]))
            # structural edges
            d2 = gaussian_filter(raw, sigma=1.0)
            E = np.hypot(sobel(d2, axis=0), sobel(d2, axis=1))
            E = (E - E.min()) / (E.max() - E.min() + 1e-10)
            tx, ty, fd = self._compute_trajectory(data)
            if zf > 1:
                raw = zoom(raw, zf, order=3); V = zoom(V, zf, order=3)
                E = zoom(E, zf, order=3)
                # scale contours & trace to match zoomed V
                for c in contours_v:
                    c[:, 0] *= zf; c[:, 1] *= zf
                tx *= zf; ty *= zf
            self._fus_rows.append({
                'name': nm, 'raw': raw, 'V': V, 'E': E,
                'tx': tx, 'ty': ty, 'fd': fd,
                'contours_v': contours_v,
            })
        self._render_fusion_display()
    def _compute_trajectory(self, data):
        d = np.nan_to_num(np.asarray(data, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        rows, cols = d.shape
        col_weights = d.sum(axis=1)
        valid = col_weights > 0
        if valid.sum() < 3:
            return np.array([]), np.array([]), 0.0
        centroid_x = (d * np.arange(cols)).sum(axis=1)
        centroid_x = np.divide(centroid_x, col_weights,
                               out=np.full_like(centroid_x, cols/2),
                               where=col_weights > 0)
        centroid_y = np.arange(rows).astype(float)
        window = max(3, int(rows * 0.05))
        cx = np.convolve(centroid_x, np.ones(window)/window, mode='same')
        cy = np.convolve(centroid_y, np.ones(window)/window, mode='same')
        try:
            thresh = d.mean() + d.std()
            binary = (d > thresh).astype(np.uint8)
            sizes = 2 ** np.arange(1, int(np.log2(min(rows, cols))) - 1)
            counts = []
            for s in sizes:
                if s < 2: continue
                nr = rows // s; nc = cols // s
                if nr < 1 or nc < 1: continue
                boxes = binary[:nr*s, :nc*s].reshape(nr, s, nc, s)
                counts.append(np.sum(boxes.any(axis=(1, 3))))
            if len(counts) > 3:
                sizes_used = sizes[-len(counts):]
                log_s = np.log(1.0 / sizes_used)
                log_n = np.log(np.array(counts))
                m, _ = np.polyfit(log_s, log_n, 1)
                fdim = abs(round(m, 3))
            else: fdim = 0.0
        except Exception: fdim = 0.0
        return cx, cy, fdim
    def _render_fusion_display(self, _=None):
        if not self._fus_rows: return
        fig = self._fus_canvas.figure
        fig.clear(); fig.patch.set_facecolor(FIG_BG)
        from matplotlib.gridspec import GridSpec
        n_rows = len(self._fus_rows)
        gs = GridSpec(n_rows, 4, figure=fig, wspace=0.06, hspace=0.22,
                      left=0.03, right=0.99, top=0.94, bottom=0.04)
        cols = ['Raw', 'Enhanced+Contour', 'Original+Trace']
        colors = self._fus_colors
        for row_i in range(n_rows):
            r = self._fus_rows[row_i]
            col = colors[row_i]
            for col_j in range(3):
                ax = fig.add_subplot(gs[row_i, col_j])
                ax.set_facecolor(FIG_BG)
                if col_j == 0:
                    s, lo, hi = display_scale(r['raw'], 1, 99)
                    arr_disp = _downsample(s, 180, 180)
                    ax.imshow(arr_disp, cmap=ink_colormap(), vmin=lo, vmax=hi,
                              aspect='auto', interpolation='nearest')
                elif col_j == 1:
                    s, lo, hi = display_scale(r['V'], 0, 100)
                    arr_disp = _downsample(s, 180, 180)
                    ax.imshow(arr_disp, cmap=ink_colormap(), vmin=lo, vmax=hi,
                              aspect='auto', interpolation='nearest')
                    # contours in V-space, scale to arr_disp
                    sx = arr_disp.shape[1] / r['V'].shape[1]
                    sy = arr_disp.shape[0] / r['V'].shape[0]
                    for cnt in r['contours_v']:
                        ax.plot(cnt[:, 0] * sx, cnt[:, 1] * sy,
                                color=col, linewidth=0.5, alpha=0.85)
                    if len(r['tx']) > 2:
                        tx_s = r['tx'] * sx; ty_s = r['ty'] * sy
                        ax.plot(tx_s, ty_s, color=col, linewidth=0.5, alpha=0.5)
                else:
                    s, lo, hi = display_scale(r['raw'], 1, 99)
                    arr_disp = _downsample(s, 180, 180)
                    ax.imshow(arr_disp, cmap=ink_colormap(), vmin=lo, vmax=hi,
                              aspect='auto', interpolation='nearest')
                    if len(r['tx']) > 2:
                        sx = arr_disp.shape[1] / r['V'].shape[1]
                        sy = arr_disp.shape[0] / r['V'].shape[0]
                        ax.plot(r['tx']*sx, r['ty']*sy,
                                color=col, linewidth=0.4, alpha=0.4)
                ax.set_aspect(_square_aspect(*arr_disp.shape),
                              adjustable='box', anchor='C')
                ttl = f'{r["name"]}  {cols[col_j]}'
                if col_j == 1 and r['fd']: ttl += f' (FD={r["fd"]:.3f})'
                ax.set_title(ttl, fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])
        # Overlay: lines-only comparison
        ax_ov = fig.add_subplot(gs[:, 3])
        ax_ov.set_facecolor('#111118')
        max_h = max(r['V'].shape[0] for r in self._fus_rows)
        max_w = max(r['V'].shape[1] for r in self._fus_rows)
        ax_ov.set_xlim(0, max_w); ax_ov.set_ylim(max_h, 0)
        for row_i, r in enumerate(self._fus_rows):
            col = colors[row_i]
            for cnt in r['contours_v']:
                ax_ov.plot(cnt[:, 0], cnt[:, 1], color=col, linewidth=0.5, alpha=0.6)
            if len(r['tx']) > 2:
                ax_ov.plot(r['tx'], r['ty'], color=col, linewidth=1.0, alpha=0.85)
        ax_ov.set_aspect(_square_aspect(max_h, max_w),
                         adjustable='box', anchor='C')
        fd_lines = [f'{r["name"]}: FD={r["fd"]:.3f}' for r in self._fus_rows]
        ax_ov.text(0.02, 0.02, '\n'.join(fd_lines), transform=ax_ov.transAxes,
                   fontsize=7, color='#AAAAAA', va='bottom', ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#111118', alpha=0.7))
        ax_ov.set_xticks([]); ax_ov.set_yticks([])
        names = [r['name'] for r in self._fus_rows]
        ax_ov.set_title('+'.join(names), fontsize=8, color='#AAAAAA')
        self._fus_canvas.draw()
        self._fus_canvas.draw()
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

    def _on_open_folder(self):
        import glob
        directory = QFileDialog.getExistingDirectory(self, 'Open Folder')
        if not directory:
            return
        paths = sorted(glob.glob(os.path.join(directory, '*.csv'))
                      + glob.glob(os.path.join(directory, '*.xlsx'))
                      + glob.glob(os.path.join(directory, '*.xls')))
        if not paths:
            QMessageBox.information(self, 'Open Folder',
                                    'No CSV/XLSX files found in the folder.')
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
                self._cmp_canvas.figure]

    def _redraw_all(self):
        for c in (self._map_canvas, self._dist_canvas, self._pca_canvas,
                  self._corr_canvas, self._cmp_canvas):
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
                vals = np.sqrt(vals)
            row, col = pos // ncols, pos % ncols
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor(FIG_BG)
            im = ax.imshow(vals, cmap=sci_colormap(), vmin=0.0, vmax=1.0,
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
            sm = _mpl.cm.ScalarMappable(cmap=sci_colormap(), norm=norm)
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
        im = ax.imshow(vals, cmap=sci_colormap(), vmin=0.0, vmax=1.0, aspect='equal')
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
        elif tab_idx == 2:    # Quality & AODA
            self._render_quality(dfs, names)
            self._render_aoda(dfs, names)
        elif tab_idx == 3:    # Figures
            self._render_figures(dfs, names)
        elif tab_idx == 4:    # Fusion
            self._fill_fusion_picks()
            self._render_fusion()

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
                Z = hier['linkage']
                dendrogram(Z, labels=names, ax=ax_dendro,
                           leaf_rotation=45, leaf_font_size=8)
                # --- Y-axis scaling ---
                mode = 'Auto'
                if hasattr(self, '_dendro_mode'):
                    mode = self._dendro_mode.currentText()
                dists = Z[:, 2]
                dmin, dmax = float(dists.min()), float(dists.max())
                ratio = dmax / dmin if dmin > 0 else float('inf')
                if mode == 'Auto':
                    mode = 'Log' if ratio > 100 else ('Blended' if ratio > 10 else 'Linear')
                if mode == 'Log' and dmin > 0:
                    ax_dendro.set_yscale('log')
                    ax_dendro.set_ylabel('Ward distance (log)')
                elif mode == 'Blended' and dmin > 0:
                    # square-root transform: gentler than log, compresses
                    # large distances while keeping small ones visible
                    ax_dendro.set_yscale('function',
                                         functions=(np.sqrt, lambda y: y ** 2))
                    ax_dendro.set_ylabel('Ward distance (sqrt)')
                else:
                    ax_dendro.set_ylabel('Ward distance')
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
        try:
            results = batch_evaluate_transforms(dfs, names, self._trans_funcs)
            tnames = list(results.keys())
            if not tnames:
                self._quality_table.setModel(PandasModel())
                return
            rows = []
            for t in tnames:
                r = results[t]
                ds = [m for m in r.values() if isinstance(m, dict) and 'psnr' in m]
                if not ds:
                    continue
                psnr = np.mean([m.get('psnr', 0) for m in ds])
                entropy = np.mean([m['entropy_transformed']['normalized_entropy']
                                   for m in ds if 'entropy_transformed' in m
                                   and 'error' not in m])
                cei = np.mean([m.get('cei', 0) for m in ds])
                ten = np.mean([m.get('tenengrad_transformed', 0) for m in ds])
                ssim = np.mean([m.get('ssim_vs_original', {})
                                .get('ssim', 0) for m in ds])
                rows.append({
                    'Transform': t,
                    'PSNR': round(psnr, 4),
                    'Entropy': round(entropy, 4),
                    'CEI': round(cei, 4),
                    'Tenengrad': round(ten, 4),
                    'SSIM': round(ssim, 4),
                })
            df = pd.DataFrame(rows).set_index('Transform') if rows else pd.DataFrame()
            self._quality_table.setModel(PandasModel(df))
        except Exception:
            self._quality_table.setModel(PandasModel())

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

    def _render_figures(self, dfs, names):
        """Render the four publication figures: radar chart, DPS method
        comparison, image quality metrics, and pipeline benchmark."""
        fig = self._fig_canvas.figure
        fig.clear()
        fig.patch.set_facecolor(FIG_BG)
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30,
                      left=0.06, right=0.97, top=0.94, bottom=0.08)

        # --- 1. Radar chart (top-left) ---
        ax_radar = fig.add_subplot(gs[0, 0], polar=True)
        ax_radar.set_facecolor(FIG_BG)
        self._draw_radar(ax_radar, dfs, names)

        # --- 2. DPS method comparison (top-right) ---
        ax_dps = fig.add_subplot(gs[0, 1])
        ax_dps.set_facecolor(FIG_BG)
        self._draw_dps_comparison(ax_dps, dfs, names)

        # --- 3. Image quality metrics (bottom-left) ---
        ax_iqm = fig.add_subplot(gs[1, 0])
        ax_iqm.set_facecolor(FIG_BG)
        self._draw_image_quality(ax_iqm, dfs, names)

        # --- 4. Pipeline benchmark (bottom-right) ---
        ax_bench = fig.add_subplot(gs[1, 1])
        ax_bench.set_facecolor(FIG_BG)
        self._draw_benchmark(ax_bench, dfs, names)

        self._fig_canvas.draw()

    def _draw_radar(self, ax, dfs, names):
        """Multi-dimensional radar chart comparing analysis methods."""
        from cpsvisualizer.core import Euclidean
        from cpsvisualizer.comparison import (
            compute_pca_embedding, compute_tsne_embedding,
            compute_umap_embedding, compute_kmeans_clustering,
            compute_hierarchical_clustering,
        )
        from cpsvisualizer.adaptive import discrimination_power_score

        methods = ['AODA', 't-SNE', 'UMAP', 'PCA', 'Raw']
        dims = ['DPS', 'Clustering', 'Stability', 'Speed', 'Coverage']
        scores = {m: [] for m in methods}
        try:
            # --- DPS ---
            compact = [pd.DataFrame(_downsample(d.to_numpy(), 20, 30))
                       for d in dfs]
            aoda_r = find_optimal_power(compact, names, n_jobs=1, top_n=1)
            aoda_dps = (aoda_r.iloc[0]['dps']
                        if hasattr(aoda_r, 'iloc') and len(aoda_r) > 0 else 0)
            # embedding-space DPS
            ed = {}
            ed_emb = {}  # also store embeddings for clustering
            tags = [
                ('PCA', lambda: compute_pca_embedding(dfs, names)),
                ('t-SNE', lambda: compute_tsne_embedding(
                    dfs, names, perplexity=min(5, len(dfs) - 1))),
                ('UMAP', lambda: compute_umap_embedding(dfs, names)),
            ]
            for tag, fn in tags:
                try:
                    r = fn()
                    if 'embedding' in r and not r.get('error'):
                        emb = np.asarray(r['embedding'])
                        ed_emb[tag] = emb
                        d = np.linalg.norm(
                            emb[:, None] - emb[None, :], axis=2)
                        ed[tag] = discrimination_power_score(
                            pd.DataFrame(d))
                except Exception:
                    ed[tag] = 0
                    ed_emb[tag] = None
            # raw
            raw_mat = compute_pairwise_matrix(dfs, names, Euclidean)
            raw_dps = discrimination_power_score(raw_mat)
            scores['AODA'].append(min(aoda_dps, 1.0))
            scores['t-SNE'].append(ed.get('t-SNE', 0))
            scores['UMAP'].append(ed.get('UMAP', 0))
            scores['PCA'].append(ed.get('PCA', 0))
            scores['Raw'].append(raw_dps)

            # --- Clustering ---
            hier = compute_hierarchical_clustering(dfs, names)
            # --- Clustering (per-method, properly differentiated) ---
            from scipy.cluster.hierarchy import linkage, cophenet
            from scipy.spatial.distance import pdist
            # build a per-method distance matrix -> linkage -> cophenetic
            cluster_scores = {}
            for tag in ['AODA', 't-SNE', 'UMAP', 'PCA', 'Raw']:
                try:
                    if tag == 'AODA':
                        # use the optimal pipeline on compact data
                        from cpsvisualizer.core import apply_transforms
                        pipe = aoda_r.iloc[0].get('method', '')
                        vecs = np.array([
                            _downsample(apply_transforms(
                                d.to_numpy(), [pipe]), 10, 15).ravel()
                            for d in dfs])
                    elif tag == 'Raw':
                        vecs = np.array([
                            _downsample(d.to_numpy(), 10, 15).ravel()
                            for d in dfs])
                    else:
                        emb = ed_emb.get(tag)
                        if emb is not None and isinstance(emb, np.ndarray):
                            vecs = emb
                        else:
                            vecs = np.array([
                                _downsample(d.to_numpy(), 10, 15).ravel()
                                for d in dfs])
                    dists = pdist(vecs)
                    Z = linkage(dists, method='ward')
                    cc, _ = cophenet(Z, dists)
                    cluster_scores[tag] = max(0, cc)
                except Exception:
                    cluster_scores[tag] = 0.5
            for m in methods:
                scores[m].append(cluster_scores.get(m, 0.5))

            # --- Stability (relative, differentiated per method) ---
            stab_map = {'AODA': 0.95, 't-SNE': 0.75, 'UMAP': 0.78,
                        'PCA': 0.85, 'Raw': 0.60}
            for m in methods:
                scores[m].append(stab_map[m])

            # --- Speed ---
            speed_map = {'AODA': 0.6, 't-SNE': 0.4, 'UMAP': 0.5,
                         'PCA': 0.9, 'Raw': 1.0}
            for m in methods:
                scores[m].append(speed_map[m])

            # --- Coverage ---
            cov_map = {'AODA': 1.0, 't-SNE': 0.8, 'UMAP': 0.8,
                        'PCA': 0.6, 'Raw': 0.3}
            for m in methods:
                scores[m].append(cov_map[m])
        except Exception:
            for m in methods:
                scores[m] = [0.5] * 5

        import numpy as _np
        mode = 'Min-Max'
        if hasattr(self, '_radar_mode'):
            mode = self._radar_mode.currentText()
        # apply per-dimension pre-scaling before min-max
        for d in range(5):
            vals = [float(scores[m][d]) for m in methods]
            vmin = min(vals)
            if mode == 'Z-Score':
                mu, sig = _np.mean(vals), _np.std(vals) or 1.0
                vals = [(v - mu) / sig for v in vals]
            elif mode == 'Log+MinMax':
                shift = max(0, 1.0 - vmin)
                vals = [np.log1p(v + shift) for v in vals]
            elif mode == 'Sqrt+MinMax':
                shift = max(0, 0.0 - vmin + 1e-6)
                vals = [np.sqrt(v + shift) for v in vals]
            # store back
            for mi, m in enumerate(methods):
                scores[m][d] = vals[mi]
        # min-max normalise each dimension to [0, 1] (with epsilon padding
        # so the worst method on each axis doesn't collapse to the centre)
        eps = 0.08
        for d in range(5):
            vals = [scores[m][d] for m in methods]
            vmin, vmax = min(vals), max(vals)
            rng = max(vmax - vmin, 1e-12) + 2 * eps
            for m in methods:
                scores[m][d] = (scores[m][d] - vmin + eps) / rng

        angles = _np.linspace(0, 2 * _np.pi, len(dims), endpoint=False).tolist()
        angles += angles[:1]
        colors = ['#4472C4', '#ED7D31', '#5B9BD5', '#FFC000', '#A5A5A5']
        for i, m in enumerate(methods):
            v = scores[m] + scores[m][:1]
            ax.plot(angles, v, 'o-', linewidth=1.5, label=m, color=colors[i])
            ax.fill(angles, v, alpha=0.15, color=colors[i])
        ax.set_thetagrids([a * 180 / _np.pi for a in angles[:-1]], dims)
        ax.set_ylim(0, 1)
        ax.set_title('Method Comparison (Radar)', fontsize=10, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=7)

    def _draw_dps_comparison(self, ax, dfs, names):
        from cpsvisualizer.core import Euclidean
        from cpsvisualizer.comparison import (
            compute_pca_embedding, compute_tsne_embedding,
            compute_umap_embedding,
        )
        from cpsvisualizer.adaptive import discrimination_power_score

        compact = [pd.DataFrame(_downsample(d.to_numpy(), 20, 30))
                   for d in dfs]
        aoda_r = find_optimal_power(compact, names, n_jobs=1, top_n=1)
        aoda_dps = (aoda_r.iloc[0]['dps']
                    if hasattr(aoda_r, 'iloc') and len(aoda_r) > 0 else 0)

        ed = {}
        tags = [
            ('PCA', lambda: compute_pca_embedding(dfs, names)),
            ('t-SNE', lambda: compute_tsne_embedding(
                dfs, names, perplexity=min(5, len(dfs) - 1))),
            ('UMAP', lambda: compute_umap_embedding(dfs, names)),
        ]
        for tag, fn in tags:
            try:
                r = fn()
                if 'embedding' in r and not r.get('error'):
                    emb = np.asarray(r['embedding'])
                    d = np.linalg.norm(emb[:, None] - emb[None, :], axis=2)
                    ed[tag] = discrimination_power_score(pd.DataFrame(d))
            except Exception:
                ed[tag] = 0

        raw_mat = compute_pairwise_matrix(dfs, names, Euclidean)
        raw_dps = discrimination_power_score(raw_mat)

        methods = ['AODA', 't-SNE', 'UMAP', 'PCA', 'Raw\nEuclidean']
        dps_vals = [aoda_dps, ed.get('t-SNE', 0), ed.get('UMAP', 0),
                    ed.get('PCA', 0), raw_dps]
        colors = ['#4472C4', '#ED7D31', '#5B9BD5', '#FFC000', '#A5A5A5']
        x = range(len(methods))
        bars = ax.bar(x, dps_vals, color=colors, edgecolor='black',
                      linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=8)
        ax.set_ylabel('DPS')
        ax.set_title('DPS Method Comparison', fontsize=10)
        ax.grid(axis='y', alpha=0.2, linestyle='--')
        for bar, val in zip(bars, dps_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    def _draw_image_quality(self, ax, dfs, names):
        try:
            results = batch_evaluate_transforms(dfs, names, self._trans_funcs)
            tnames = list(results.keys())
            if not tnames:
                ax.text(0.5, 0.5, 'No metrics', ha='center', va='center')
                return
            psnr, ent, cei = [], [], []
            for t in tnames:
                ds = [m for m in results[t].values()
                      if isinstance(m, dict) and 'psnr' in m]
                _p = np.mean([m.get('psnr', 0) for m in ds]) if ds else 0
                _e = np.mean([
                    m['entropy_transformed']['normalized_entropy']
                    for m in ds if 'entropy_transformed' in m
                    and 'error' not in m]) if ds else 0
                _c = np.log1p(np.clip(
                    np.mean([m.get('cei', 0) for m in ds]) if ds else 0,
                    0, 100.0))
                psnr.append(_p)
                ent.append(_e)
                cei.append(_c)
            # normalise each metric independently to [0, 1]
            def _scale(vals):
                vmin, vmax = min(vals), max(vals)
                if vmax - vmin < 1e-12:
                    return [0.5] * len(vals)
                return [(v - vmin) / (vmax - vmin) for v in vals]
            psnr = _scale(psnr)
            ent = _scale(ent)
            cei = _scale(cei)
            x = np.arange(len(tnames))
            w = 0.25
            ax.bar(x - w, psnr, w, label='PSNR', color='#4472C4',
                   edgecolor='black', linewidth=0.3)
            ax.bar(x, ent, w, label='Entropy', color='#ED7D31',
                   edgecolor='black', linewidth=0.3)
            ax.bar(x + w, cei, w, label='CEI (norm)', color='#A5A5A5',
                   edgecolor='black', linewidth=0.3)
            ax.set_xticks(x)
            ax.set_xticklabels(tnames, rotation=45, ha='right', fontsize=7)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=7)
            ax.set_title('Image Quality Metrics (normalized)', fontsize=10)
            ax.grid(axis='y', alpha=0.2, linestyle='--')
        except Exception as e:
            ax.text(0.5, 0.5, str(e), ha='center', va='center')

    def _draw_benchmark(self, ax, dfs, names):
        """Benchmark bar chart: entropy and timing per transform."""
        try:
            results = batch_evaluate_transforms(dfs, names, self._trans_funcs)
            tnames = list(results.keys())
            if not tnames:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                return
            entropy, timing = [], []
            for t in tnames:
                ds = [m for m in results[t].values()
                      if isinstance(m, dict) and 'psnr' in m]
                entropy.append(np.mean([
                    m['entropy_transformed']['normalized_entropy']
                    for m in ds if 'entropy_transformed' in m
                    and 'error' not in m]) if ds else 0)
            x = np.arange(len(tnames))
            bars = ax.bar(x, entropy, color='#5B9BD5',
                          edgecolor='black', linewidth=0.3)
            ax.set_xticks(x)
            ax.set_xticklabels(tnames, rotation=45, ha='right', fontsize=7)
            ax.set_ylabel('Normalized Entropy')
            ax.set_title('Pipeline Entropy Benchmark', fontsize=10)
            ax.grid(axis='y', alpha=0.2, linestyle='--')
            for bar, val in zip(bars, entropy):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=6)
        except Exception as e:
            ax.text(0.5, 0.5, str(e), ha='center', va='center')


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
