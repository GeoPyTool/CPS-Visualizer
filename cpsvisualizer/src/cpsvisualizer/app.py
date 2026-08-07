"""
CPS-Visualizer GUI — PySide6-based graphical interface for LA-ICP-MS
surface scan data visualization and similarity analysis.
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
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QMenu, QMessageBox, QPushButton,
    QSizePolicy, QSlider, QStyledItemDelegate, QTableView,
    QToolBar, QVBoxLayout, QWidget, QApplication,
)
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from cpsvisualizer.core import (
    TRANSFORM_FUNCTIONS, DISTANCE_FUNCTIONS, DISTANCE_NAMES,
    load_data_files, apply_transforms, compute_pairwise_matrix,
    log_centering_transform,
)
from cpsvisualizer.statistics import (
    compute_pca, compute_pearson_correlation_matrix,
    compute_spearman_correlation_matrix, compute_anova,
    compute_uncertainty_all, compute_descriptive_statistics_all,
)
from cpsvisualizer.metrics import (
    compute_psnr, compute_entropy, compute_contrast_enhancement_index,
    compute_tenengrad, compute_all_image_metrics, batch_evaluate_transforms,
)
from cpsvisualizer.comparison import (
    compute_pca_embedding, compute_tsne_embedding, compute_umap_embedding,
    compute_hierarchical_clustering, compute_kmeans_clustering,
    compute_all_comparisons, compute_method_comparison_metrics,
)

warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 'truetype'

TRANSFORM_NAMES = list(TRANSFORM_FUNCTIONS.keys())


def _subplot_grid(n):
    """Return (rows, cols) for an optimal nearly-square grid of n subplots."""
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return rows, cols


def visual_diff(df_A=None, df_B=None):
    """Generate a 3x3 comparison figure showing log, centered, and
    equalized representations of two datasets side by side."""
    if df_A is None or df_B is None:
        return
    data_A = df_A.values if isinstance(df_A, pd.DataFrame) else df_A
    data_B = df_B.values if isinstance(df_B, pd.DataFrame) else df_B
    data_A_log = np.log1p(data_A)
    data_B_log = np.log1p(data_B)
    diff_norm = (data_A_log - data_B_log) / (np.abs(data_A_log) + np.abs(data_B_log) + 1e-10)
    data_A_centered = log_centering_transform(data_A)
    data_B_centered = log_centering_transform(data_B)
    diff_centered = (data_A_centered - data_B_centered) / (np.abs(data_A_centered) + np.abs(data_B_centered) + 1e-10)
    from skimage.exposure import equalize_hist
    data_A_eq = equalize_hist(data_A_log)
    data_B_eq = equalize_hist(data_B_log)
    diff_eq = equalize_hist(diff_norm)
    panels = [
        ('Data A (Log)', data_A_log), ('Data B (Log)', data_B_log),
        ('Diff (Log)', diff_norm),
        ('Data A (Centered)', data_A_centered), ('Data B (Centered)', data_B_centered),
        ('Diff (Centered)', diff_centered),
        ('Data A (Equalized)', data_A_eq), ('Data B (Equalized)', data_B_eq),
        ('Diff (Equalized)', diff_eq),
    ]
    fig = plt.figure(figsize=(10, 10))
    for idx, (title, arr) in enumerate(panels, 1):
        ax = fig.add_subplot(3, 3, idx)
        ax.imshow(arr, aspect='auto', cmap='gray')
        ax.set_title(title)
        plt.colorbar(ax.imshow(arr, aspect='auto', cmap='gray'), ax=ax)
    fig.tight_layout()
    return fig


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


class QSwitch(QSlider):
    """Binary toggle slider widget."""
    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self.setRange(0, 1)
        self.setFixedSize(60, 20)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.setValue(1 if self.value() > 0.5 else 0)


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
        self._list.setMaximumHeight(100)

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
        self.dpi = 50
        self.df_list = []
        self.df_name_list = []
        self.result_df_dict = {}
        self.plot_flag = 'select'

        # Build a name→function dispatch dict from core.py
        self._dist_funcs = {f.__name__: f for f in DISTANCE_FUNCTIONS}
        self._dist_names = [""] + list(self._dist_funcs.keys())
        self._trans_funcs = dict(TRANSFORM_FUNCTIONS)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _init_ui(self):
        self.setWindowTitle('CPS-Visualizer')
        self.resize(1024, 600)

        toolbar = QToolBar()
        self.addToolBar(toolbar)

        actions = [
            ('Open Data',  'Ctrl+O', self._on_open),
            ('Clear Data', None,     self._on_clear),
            ('Calculate All', 'Ctrl+W', self._on_calc_all),
            ('Save Data',  'Ctrl+S', self._on_save),
            ('Plot Data',  None,     self._on_plot),
            ('Plot All',   'Ctrl+A', self._on_plot_all),
            ('Save Plot',  'Ctrl+P', self._on_save_plot),
            ('Clear Plot', None,     self._on_clear_plot),
        ]
        for label, shortcut, slot in actions:
            act = QAction(label, self)
            if shortcut:
                act.setShortcut(shortcut)
            act.triggered.connect(slot)
            toolbar.addAction(act)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar.addWidget(spacer)

        # Left panel – data selector + function selector
        self._data_label = QLabel('Select Data')
        self._data_sel = QListWidget(self)
        self._data_sel.setSelectionMode(QListWidget.MultiSelection)
        self._data_sel.itemSelectionChanged.connect(self._on_plot)

        self._func_label = QLabel('Select Transform')
        self._func_sel = QListWidget(self)
        self._func_sel.addItems(TRANSFORM_NAMES)
        self._func_sel.setSelectionMode(QListWidget.MultiSelection)
        self._func_sel.itemSelectionChanged.connect(self._on_transform_change)

        self._cmp_label = QLabel('Distance Metric')
        self._cmp_sel = QComboBox(self)
        self._cmp_sel.addItems(self._dist_names)
        self._cmp_sel.currentTextChanged.connect(self._on_cmp_changed)

        self._table = CustomQTableView()

        left_left = QVBoxLayout()
        left_left.addWidget(self._cmp_label)
        left_left.addWidget(self._cmp_sel)
        left_left.addWidget(self._table, 10)

        left_right = QVBoxLayout()
        left_right.addWidget(self._data_label)
        left_right.addWidget(self._data_sel)
        left_right.addWidget(self._func_label)
        left_right.addWidget(self._func_sel)

        left = QHBoxLayout()
        left.addLayout(left_left, 10)
        left.addLayout(left_right, 1)

        # Right panel – matplotlib canvas
        self._fig = Figure((4, 3), dpi=self.dpi)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        right = QVBoxLayout()
        right.addWidget(self._canvas)

        base = QHBoxLayout()
        base.addLayout(left, 4)
        base.addLayout(right, 6)

        main_frame = QWidget()
        main_frame.setLayout(base)
        self.setCentralWidget(main_frame)
        self.show()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _selected_data_indices(self):
        return [self.df_name_list.index(it.text())
                for it in self._data_sel.selectedItems()]

    def _selected_transform_names(self):
        return [it.text() for it in self._func_sel.selectedItems()]

    def _clean_name(self, path):
        return os.path.splitext(os.path.basename(path))[0].split('_')[0]

    # ---- File I/O ----
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
        self._on_cmp_changed()

    def _on_clear(self):
        self.df_list.clear()
        self.df_name_list.clear()
        self.result_df_dict.clear()
        self._table.setModel(PandasModel())
        self._data_sel.clear()
        self._fig.clear()
        self._canvas.draw()
        self.plot_flag = 'select'

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
        arrays = [df.values.ravel() for df in self.df_list]
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
        if name not in self.result_df_dict:
            self.result_df_dict[name] = compute_pairwise_matrix(
                self.df_list, self.df_name_list, self._dist_funcs[name])
        self._table.setModel(PandasModel(self.result_df_dict[name].round(4)))

    # ---- Plotting ----
    def _on_transform_change(self):
        if self.plot_flag == 'select':
            self._on_plot()
        else:
            self._on_plot_all()

    def _apply_transforms(self, data):
        for name in self._selected_transform_names():
            try:
                data = self._trans_funcs[name](data)
            except Exception as e:
                QMessageBox.critical(self, 'Transform Error', str(e))
        return data

    def _on_plot(self):
        self.plot_flag = 'select'
        indices = self._selected_data_indices()
        if not indices:
            return
        self._fig.clear()
        rows, cols = _subplot_grid(len(indices))
        for pos, idx in enumerate(indices, 1):
            ax = self._fig.add_subplot(rows, cols, pos)
            data = self.df_list[idx].to_numpy()
            data = self._apply_transforms(data)
            ax.imshow(data, cmap='gray', aspect='auto')
            ax.set_title(self.df_name_list[idx])
        self._canvas.draw()

    def _on_plot_all(self):
        self.plot_flag = 'all'
        if not self.df_name_list:
            return
        self._fig.clear()
        rows, cols = _subplot_grid(len(self.df_name_list))
        for pos, (name, df) in enumerate(zip(self.df_name_list, self.df_list), 1):
            ax = self._fig.add_subplot(rows, cols, pos)
            data = df.to_numpy()
            data = self._apply_transforms(data)
            ax.imshow(data, cmap='gray', aspect='auto')
            ax.set_title(name)
        self._canvas.draw()

    def _on_save_plot(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Plot', 'CPS_Image',
            'PNG (*.png);;JPEG (*.jpg *.jpeg);;SVG (*.svg);;PDF (*.pdf)')
        if path:
            dpi = self.dpi * 10 if path.lower().endswith(('.png', '.jpg', '.jpeg')) else None
            self._canvas.figure.savefig(path, dpi=dpi)

    def _on_clear_plot(self):
        self._fig.clear()
        self._canvas.draw()


def main():
    try:
        metadata = importlib.metadata.metadata(
            sys.modules["__main__"].__package__)
        QApplication.setApplicationName(metadata.get("Formal-Name",
                    'CPS-Visualizer'))
    except Exception:
        QApplication.setApplicationName('CPS-Visualizer')
    try:
        app = QApplication(sys.argv)
    except RuntimeError:
        QApplication.instance().quit()
        app = QApplication(sys.argv)
    window = CPSVisualizer()
    sys.exit(app.exec())


if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
    except RuntimeError:
        QApplication.instance().quit()
        app = QApplication(sys.argv)
    window = CPSVisualizer()
    window.show()
    sys.exit(app.exec())
