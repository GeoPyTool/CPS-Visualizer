"""
Calculation and visualization of CPS (counts per second) for ICPMS scan data.
"""
import importlib.metadata
import sys

from PySide6 import QtWidgets
import json
import pickle
import sqlite3
import sys
import re
import os
import numpy as np
import itertools
import math

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.patches import ConnectionStyle, Polygon
from matplotlib.collections import PatchCollection
from matplotlib import collections
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
try:
    from importlib import metadata as importlib_metadata
except ImportError:
    # Backwards compatibility - importlib.metadata was added in Python 3.8
    import importlib_metadata

from PySide6.QtGui import QAction, QGuiApplication
from PySide6.QtWidgets import QComboBox,QAbstractItemView, QHBoxLayout, QLabel, QMainWindow, QApplication, QMenu, QSizePolicy, QWidget, QToolBar, QFileDialog, QTableView, QVBoxLayout, QHBoxLayout, QWidget, QSlider,  QGroupBox , QLabel , QWidgetAction, QPushButton, QSizePolicy, QMessageBox
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QLabel
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QVariantAnimation, Qt

from PySide6.QtWidgets import QComboBox, QListWidget, QListWidgetItem, QCheckBox, QStyledItemDelegate, QVBoxLayout, QWidget, QApplication
from PySide6.QtCore import Qt, Signal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
from PySide6.QtGui import QGuiApplication

from PySide6.QtCore import QAbstractTableModel, Qt, QModelIndex

from scipy.stats import gmean

from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import os
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import sobel
from skimage import feature
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from skimage.filters import threshold_otsu, threshold_local
from skimage import exposure
from itertools import combinations

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] =  'truetype'

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取当前文件的目录
current_directory = os.path.dirname(current_file_path)
working_directory = os.path.dirname(current_file_path)
# 改变当前工作目录
os.chdir(current_directory)

class PandasModel(QAbstractTableModel):
    def __init__(self, df=pd.DataFrame(), parent=None):
        QAbstractTableModel.__init__(self, parent=parent)
        self._df = df
        self._changed = False
        self._filters = {}
        self._sortBy = []
        self._sortDirection = []

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal:
            try:
                return self._df.columns.tolist()[section]
            except (IndexError,):
                return None
        elif orientation == Qt.Vertical:
            try:
                return self._df.index.tolist()[section]
            except (IndexError,):
                return None

    def data(self, index, role):
        if role == Qt.DisplayRole or role == Qt.EditRole:
            try:
                return str(self._df.iloc[index.row(), index.column()])
            except:
                pass
        elif role == Qt.CheckStateRole:
            return None

        return None

    def flags(self, index):
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable

    def setData(self, index, value, role=Qt.EditRole):
        row = self._df.index[index.row()]
        col = self._df.columns[index.column()]
        dtype = self._df[col].dtype
        if dtype != object:
            value = None if value == '' else dtype.type(value)
        self._df.at[row, col] = value
        self._changed = True
        return True

    def rowCount(self, parent=QModelIndex()):
        return len(self._df.index)

    def columnCount(self, parent=QModelIndex()):
        return len(self._df.columns)

    def sort(self, column, order):
        colname = self._df.columns.tolist()[column]
        self.layoutAboutToBeChanged.emit()
        try:
            self._df.sort_values(colname, ascending=order == Qt.AscendingOrder, inplace=True)
        except:
            pass
        try:
            self._df.reset_index(inplace=True, drop=True)
        except:
            pass
        self.layoutChanged.emit()

class CustomQTableView(QTableView):
    df = pd.DataFrame()
    def __init__(self, *args):
        super().__init__(*args)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers |
                             QAbstractItemView.DoubleClicked)
        self.setSortingEnabled(True)

    def keyPressEvent(self, event):  # Reimplement the event here
        return

    def contextMenuEvent(self, event):
        contextMenu = QMenu(self)
        copyAction = QAction("Copy", self)
        contextMenu.addAction(copyAction)
        copyAction.triggered.connect(self.copySelection)
        contextMenu.exec_(event.globalPos())

    def copySelection(self):
        selection = self.selectionModel().selection().indexes()
        if selection:
            rows = sorted(index.row() for index in selection)
            columns = sorted(index.column() for index in selection)
            rowcount = rows[-1] - rows[0] + 1
            colcount = columns[-1] - columns[0] + 1
            table = [[''] * colcount for _ in range(rowcount)]
            for index in selection:
                row = index.row() - rows[0]
                column = index.column() - columns[0]
                table[row][column] = index.data()
            stream = '\n'.join('\t'.join(row) for row in table)
            QGuiApplication.clipboard().setText(stream)

class AppForm(QMainWindow):
    def __init__(self, parent=None, df=pd.DataFrame(),title = 'AppForm'):
        self.df = df
        self.title = title 
        self.FileName_Hint = title
        QMainWindow.__init__(self, parent)
        self.setWindowTitle(self.title)
        self.create_main_frame()

    def create_main_frame(self):        
        self.resize(400, 600) 
        self.main_frame = QWidget()
        self.table = CustomQTableView()
        model = PandasModel(self.df)
        self.table.setModel(model)  # 设置表格视图的模型
        self.save_button = QPushButton('&Save')
        self.save_button.clicked.connect(self.saveDataFile)
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.table)
        self.vbox.addWidget(self.save_button)
        self.main_frame.setLayout(self.vbox)
        self.setCentralWidget(self.main_frame)

    def saveDataFile(self):

        DataFileOutput, ok2 = QFileDialog.getSaveFileName(self,
                                                          'Save Data File',
                                                          working_directory + self.FileName_Hint,
                                                          'CSV Files (*.csv);;Excel Files (*.xlsx)')  # 数据文件保存输出

        if "Label" in self.df.columns.values.tolist():
            self.df = self.df.set_index('Label')

        if (DataFileOutput != ''):

            if ('csv' in DataFileOutput):
                self.df.to_csv(DataFileOutput, sep=',', encoding='utf-8')

            elif ('xls' in DataFileOutput):
                self.df.to_excel(DataFileOutput)

class QSwitch(QSlider):
    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self.setRange(0, 1)
        self.setFixedSize(60, 20)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self.value() > 0.5:
            self.setValue(1)
        else:
            self.setValue(0)

def mutual_info(df_A=pd.DataFrame,df_B=pd.DataFrame):
    data_A = df_A.values
    data_B = df_B.values
    pass
    # 计算互信息
    mutual_info_score_list = []
    mutual_info_regression_list = []

    # 使用mutual_info_score计算互信息（截断数据集）
    for i in range(data_A.shape[1]):
        # 获取当前列的样本数
        len_A = len(data_A[:, i])
        len_B = len(data_B[:, i])
        
        # 取较小的样本数
        min_len = min(len_A, len_B)
        
        # 截断数据
        truncated_A = data_A[:min_len, i]
        truncated_B = data_B[:min_len, i]
        
        # 计算互信息分数
        mi_s = mutual_info_score(truncated_A, truncated_B)
        
        # 将结果添加到列表中
        mutual_info_score_list.append(mi_s)

    # 判断哪个数据集的样本数更多
    if data_A.shape[0] > data_B.shape[0]:
        # 如果data_A的样本数更多，重复data_B
        data_B_repeated = np.tile(data_B, (int(np.ceil(data_A.shape[0] / data_B.shape[0])), 1))[:data_A.shape[0], :]
        data_A_repeated = data_A
    else:
        # 如果data_B的样本数更多，重复data_A
        data_A_repeated = np.tile(data_A, (int(np.ceil(data_B.shape[0] / data_A.shape[0])), 1))[:data_B.shape[0], :]
        data_B_repeated = data_B

    for i in range(data_A.shape[1]):
        mi_r = mutual_info_regression(data_A_repeated[:, i].reshape(-1, 1), data_B_repeated[:, i])
        mutual_info_regression_list.append(mi_r[0])

    # 计算平均互信息
    average_mutual_info_s = np.mean(mutual_info_score_list)
    average_mutual_info_r = np.mean(mutual_info_regression_list)

    # 线性加权平均
    weight_s_linear = data_A.shape[0] / (data_A.shape[0] + data_B.shape[0])
    weight_r_linear = data_B.shape[0] / (data_A.shape[0] + data_B.shape[0])
    average_mutual_info_linear = weight_s_linear * average_mutual_info_s + weight_r_linear * average_mutual_info_r

    # 对数加权平均
    log_weight_s = np.log(data_A.shape[0])
    log_weight_r = np.log(data_B.shape[0])
    total_log_weight = log_weight_s + log_weight_r
    normalized_log_weight_s = log_weight_s / total_log_weight
    normalized_log_weight_r = log_weight_r / total_log_weight
    average_mutual_info_log = normalized_log_weight_s * average_mutual_info_s + normalized_log_weight_r * average_mutual_info_r

    # 指数加权平均
    exp_weight_s = np.exp(data_A.shape[0])
    exp_weight_r = np.exp(data_B.shape[0])
    total_exp_weight = exp_weight_s + exp_weight_r
    normalized_exp_weight_s = exp_weight_s / total_exp_weight
    normalized_exp_weight_r = exp_weight_r / total_exp_weight
    average_mutual_info_exp = normalized_exp_weight_s * average_mutual_info_s + normalized_exp_weight_r * average_mutual_info_r

    # 归一化权重
    total_samples = data_A.shape[0] + data_B.shape[0]
    normalized_weight_s = data_A.shape[0] / total_samples
    normalized_weight_r = data_B.shape[0] / total_samples
    average_mutual_info_normalized = normalized_weight_s * average_mutual_info_s + normalized_weight_r * average_mutual_info_r

    # print(f"Average Mutual Information (mutual_info_score): {average_mutual_info_s}")
    # print(f"Average Mutual Information (mutual_info_regression): {average_mutual_info_r}")
    # print(f"Linear Weighted Average Mutual Information: {average_mutual_info_linear}")
    # print(f"Log Weighted Average Mutual Information: {average_mutual_info_log}")
    # print(f"Exp Weighted Average Mutual Information: {average_mutual_info_exp}")
    # print(f"Normalized Weighted Average Mutual Information: {average_mutual_info_normalized}")

def log_transform(data):
    return np.log1p(data)

def log_centering_transform(data):
    # 对数据进行对数变换
    log_data = np.log1p(data)  # 使用log1p避免log(0)的问题

    # 对变换后的数据进行中心化处理
    centered_log_data = log_data - np.mean(log_data, axis=0)

    return centered_log_data

def z_score_normalization(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def standardize(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def equalize_hist(data):
    return exposure.equalize_hist(data)

def visual_diff(df_A=pd.DataFrame,df_B=pd.DataFrame):
    # 可视化，这部分做一下改进，这里的A和B都可以作为二维图像来呈现，就做一个单独A、单独B、AB对比这三个情况的吧？
    data_A = df_A.values
    data_B = df_B.values
    
    # 对 data_A 和 data_B 进行对数变换
    data_A_log = np.log1p(data_A)
    data_B_log = np.log1p(data_B)    
    # 计算标准化差值
    diff_norm_log = (data_A_log- data_B_log) / (np.abs(data_A_log) + np.abs(data_B_log) + 1e-10)

    # 对数中心化变换
    data_A_log_centered = log_centering_transform(data_A)
    data_B_log_centered = log_centering_transform(data_B)
    diff_norm_log_centered = (data_A_log_centered- data_B_log_centered) / (np.abs(data_A_log_centered) + np.abs(data_B_log_centered) + 1e-10)

    # 对数变换后的图像进行直方图均衡化
    data_A_log_eq = exposure.equalize_hist(data_A_log)
    data_B_log_eq = exposure.equalize_hist(data_B_log)
    diff_norm_eq = exposure.equalize_hist(diff_norm_log)

    # 可视化
    plt.figure(figsize=(10, 10))

    # 原始图像
    plt.subplot(3, 3, 1)
    plt.title('Data A (Log Transformed)')
    plt.imshow(data_A_log, aspect='auto', cmap='gray')
    plt.colorbar()

    plt.subplot(3, 3, 2)
    plt.title('Data B (Log Transformed)')
    plt.imshow(data_B_log, aspect='auto', cmap='gray')
    plt.colorbar()

    plt.subplot(3, 3, 3)
    plt.title('Normalized Difference (Log Transformed)')
    plt.imshow(diff_norm_log, aspect='auto', cmap='gray')
    plt.colorbar()

    # 标准化变换
    plt.subplot(3, 3, 4)
    plt.title('Data A (Log Centered)')
    plt.imshow(data_A_log_centered, aspect='auto', cmap='gray')
    plt.colorbar()

    plt.subplot(3, 3, 5)
    plt.title('Data B (Log Centered)')
    plt.imshow(data_B_log_centered, aspect='auto', cmap='gray')
    plt.colorbar()

    plt.subplot(3, 3, 6)
    plt.title('Normalized Difference (Log Centered)')
    plt.imshow(diff_norm_log_centered, aspect='auto', cmap='gray')
    plt.colorbar()

    # 直方图均衡化
    plt.subplot(3, 3, 7)
    plt.title('Data A (Equalized)')
    plt.imshow(data_A_log_eq, aspect='auto', cmap='gray')
    plt.colorbar()

    plt.subplot(3, 3, 8)
    plt.title('Data B (Equalized)')
    plt.imshow(data_B_log_eq, aspect='auto', cmap='gray')
    plt.colorbar()

    plt.subplot(3, 3, 9)
    plt.title('Normalized Difference (Equalized)')
    plt.imshow(diff_norm_eq, aspect='auto', cmap='gray')
    plt.colorbar()

    plt.tight_layout()
    # plt.axis('off') 
    plt.show()


class CheckBoxDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        checkbox = QCheckBox(parent)
        checkbox.stateChanged.connect(self.commitData)
        return checkbox

    def setEditorData(self, editor, index):
        value = index.data(Qt.CheckStateRole)
        editor.setChecked(value == Qt.Checked)

    def setModelData(self, editor, model, index):
        model.setData(index, Qt.Checked if editor.isChecked() else Qt.Unchecked, Qt.CheckStateRole)

class MultiSelectComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setPlaceholderText("Select items")
        self.list_widget = QListWidget(self)
        self.list_widget.setItemDelegate(CheckBoxDelegate())
        self.list_widget.itemChanged.connect(self.updateLineEdit)
        self.setModel(self.list_widget.model())
        self.setView(self.list_widget)
        self.list_widget.setMaximumHeight(100)  # 限制高度为100像素

    def addItem(self, text):
        item = QListWidgetItem(text)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Unchecked)
        self.list_widget.addItem(item)

    def addItems(self, texts):
        for text in texts:
            self.addItem(text)

    def updateLineEdit(self):
        selected_items = [self.list_widget.item(i).text() for i in range(self.list_widget.count()) if self.list_widget.item(i).checkState() == Qt.Checked]
        self.lineEdit().setText(", ".join(selected_items))

    def selectedItems(self):
        return [self.list_widget.item(i).text() for i in range(self.list_widget.count()) if self.list_widget.item(i).checkState() == Qt.Checked]


class CPSVisualizer(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.init_data()
        self.init_ui()
    def init_data(self):      
        self.dpi = 50
        self.df = pd.DataFrame()    
        self.df_list=[]
        self.df_name_list = []
        self.function_list = ["log_transform","centering_transform","z_score_normalization","standardize","equalize_hist"]
        self.distance_list = ["Euclidean", "Manhattan", "Chebyshev", 'Minkowski', 
                 'Cosine', 'Correlation', 'Jaccard', 'Dice', 
                 'Kulsinski', 'Rogers-Tanimoto', 'Russell-Rao', 
                 'Sokal-Michener', 'Sokal-Sneath', 'Yule',
                 'Hsim_Distance','Close_Distance',
                 'mutual_info_score_unflattern',
                 'mutual_info_score_flattern',
                 'mutual_info_regression_unflattern',
                 'mutual_info_regression_flattern',
                 'calculate_ssim', 'luminance', 'contrast', 'structure',]

    def init_ui(self):
        self.setWindowTitle('CPS-Visualizer: Calculation and visualization of CPS (counts per second) for ICPMS scan data.')
        self.resize(1024, 600)  # 设置窗口尺寸为1024*600
        

        ['Average Mutual Information Score',
        'Average Mutual Information Regression',
        'Exp Weighted Average Mutual Information',
        'Normalized Weighted Average Mutual Information']

        # 创建工具栏
        self.toolbar = QToolBar()
        self.addToolBar(self.toolbar)

        self.main_frame = QWidget()
        # 创建一个空的QWidget作为间隔
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 在工具栏中添加一个Open action
        open_action = QAction('Open Data', self)
        open_action.setShortcut('Ctrl+O')  # 设置快捷键为Ctrl+O
        open_action.triggered.connect(self.open_files)  # 连接到open_files方法
        self.toolbar.addAction(open_action)

        # 在工具栏中添加一个Clear action
        clear_action = QAction('Clear Data', self)
        clear_action.setShortcut('Ctrl+C') # 设置快捷键为Ctrl+C
        clear_action.triggered.connect(self.clear_data)  # 连接到clear_data方法
        self.toolbar.addAction(clear_action)       

        # 在工具栏中添加一个Plot action
        plot_action = QAction('Plot Data', self)
        plot_action.setShortcut('Ctrl+P')  # 设置快捷键为Ctrl+P
        plot_action.triggered.connect(self.plot_data)
        self.toolbar.addAction(plot_action)

        # 在工具栏中添加一个 Save action
        save_action = QAction('Save Data', self)
        save_action.setShortcut('Ctrl+S')  # 设置快捷键为Ctrl+S
        save_action.triggered.connect(self.save_data)
        self.toolbar.addAction(save_action)

        # 选择数据列表
        self.data_label = QLabel('Select Data')
        self.data_selector = QListWidget(self)
        self.data_selector.addItems(self.df_name_list)
        self.data_selector.setSelectionMode(QListWidget.MultiSelection)  # 设置为多选模式
        self.data_selector.itemSelectionChanged.connect(self.plot_data)  # 连接到 plot_data 方法


        # 选择数据处理方法
        self.function_label = QLabel('Select Function')
        self.function_selector = QListWidget(self)
        self.function_selector.addItems(self.function_list)
        self.function_selector.setSelectionMode(QListWidget.MultiSelection)  # 设置为多选模式
        self.function_selector.itemSelectionChanged.connect(self.plot_data)  # 连接到 plot_data 方法

        # 选择距离衡量方法
        self.Compare_label = QLabel('Select Compare Metric')
        self.Compare_selector = QComboBox(self)
        self.Compare_selector.addItems(self.distance_list)
        # self.Compare_selector.setSelectionMode(QListWidget.MultiSelection)
        self.Compare_selector.currentTextChanged.connect(self.apply_function_to_df_pairs)


        self.toolbar.addWidget(spacer) # Add a separator before the first switch
        # 创建一个表格视图
        self.table = CustomQTableView()

        # 创建一个Matplotlib画布
        self.fig = Figure((4,3), dpi=self.dpi)

        self.canvas = FigureCanvas(self.fig)

        # 设置canvas的QSizePolicy
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.canvas.setSizePolicy(sizePolicy)
        # 创建一个水平布局并添加表格视图和画布
        base_layout = QHBoxLayout()
        self.left_layout = QHBoxLayout()
        self.left_layout_left = QVBoxLayout()
        self.left_layout_right = QVBoxLayout()
        self.right_layout = QVBoxLayout()
        self.left_layout_right.addWidget(self.data_label)
        self.left_layout_right.addWidget(self.data_selector)
        self.left_layout_right.addWidget(self.function_label)
        self.left_layout_right.addWidget(self.function_selector)
        self.left_layout_left.addWidget(self.Compare_label) 
        self.left_layout_left.addWidget(self.Compare_selector) 
        self.left_layout_left.addWidget(self.table,10) 
        self.left_layout.addLayout(self.left_layout_left,10)     
        self.left_layout.addLayout(self.left_layout_right,1)
        self.right_layout.addWidget(self.canvas)
        base_layout.addLayout(self.left_layout,4)
        base_layout.addLayout(self.right_layout,6)

        # 创建一个QWidget，设置其布局为我们刚刚创建的布局，然后设置其为中心部件
        self.main_frame.setLayout(base_layout)
        self.setCentralWidget(self.main_frame)
        self.show()

    def open_files(self):
        self.df_list = []  # 初始化一个空列表来存储每个文件的 DataFrame

        self.df_name_list = []

        global working_directory 
        file_names, _ = QFileDialog.getOpenFileNames(self, 'Open Files', '', 'CSV Files (*.csv);;Excel Files (*.xls *.xlsx)')
        if file_names:
            for file_name in file_names:
                if file_name.endswith('.csv'):
                    df = pd.read_csv(file_name)
                elif file_name.endswith('.xls') or file_name.endswith('.xlsx'):
                    df = pd.read_excel(file_name)
                self.df_list.append(df)  # 将每个 DataFrame 添加到列表中
                tmp_name = os.path.basename(file_name)
                cleaned_name = self.clean_tmp_name(tmp_name)
                self.df_name_list.append(cleaned_name)

        # print(self.df_list)
        # print(self.df_name_list)
        self.data_selector.addItems(self.df_name_list)
        # model = PandasModel(self.df)
        # self.table.setModel(model) 

        self.apply_function_to_df_pairs()

    def clean_tmp_name(self,tmp_name):
        # Remove the extension
        name_without_ext = os.path.splitext(tmp_name)[0]
        # Remove everything after the first underscore
        cleaned_name = name_without_ext.split('_')[0]
        return cleaned_name



    def mutual_info_score_unflattern(self,df_A=pd.DataFrame,df_B=pd.DataFrame):
        data_A = df_A.values
        data_B = df_B.values
        # 计算互信息
        mutual_info_score_list = []

        # 获取两个数据集的列数
        num_columns_A = data_A.shape[1]
        num_columns_B = data_B.shape[1]

        # 取列数的最小值作为循环的范围
        min_columns = min(num_columns_A, num_columns_B)

        # 使用最小列数作为循环范围
        for i in range(min_columns):
            # 获取当前列的样本数
            len_A = len(data_A[:, i])
            len_B = len(data_B[:, i])
            
            # 取较小的样本数
            min_len = min(len_A, len_B)
            
            # 截断数据
            truncated_A = data_A[:min_len, i]
            truncated_B = data_B[:min_len, i]
            
            # 计算互信息分数
            mi_s = mutual_info_score(truncated_A, truncated_B)
            
            # 将结果添加到列表中
            mutual_info_score_list.append(mi_s)
        average_mutual_info_s = np.mean(mutual_info_score_list)
        # print(f"Mutual Information Score Average: {average_mutual_info_s}")
        return(average_mutual_info_s)

    def mutual_info_score_flattern(self,df_A=pd.DataFrame,df_B=pd.DataFrame):    
        data_A = df_A.values
        data_B = df_B.values
        # 获取两个数据集的样本数    
        len_A = data_A.shape[0]
        len_B = data_B.shape[0]
        # 取较小的样本数
        min_len = min(len_A, len_B)
        # 截断数据
        truncated_A = data_A[:min_len, :].flatten()
        truncated_B = data_B[:min_len, :].flatten()
        # 计算互信息分数
        mi_s = mutual_info_score(truncated_A, truncated_B)
        # print(f"Mutual Information Score (Flattened): {mi_s}")
        return(mi_s)

    def mutual_info_regression_unflattern(self,df_A=pd.DataFrame,df_B=pd.DataFrame):
        data_A = df_A.values
        data_B = df_B.values
        # 计算互信息
        mutual_info_regression_list = []

        # 获取两个数据集的列数
        num_columns_A = data_A.shape[1]
        num_columns_B = data_B.shape[1]

        # 取列数的最小值作为循环的范围
        min_columns = min(num_columns_A, num_columns_B)

        # 判断哪个数据集的样本数更多
        if data_A.shape[0] > data_B.shape[0]:
            # 如果data_A的样本数更多，重复data_B
            data_B_repeated = np.tile(data_B, (int(np.ceil(data_A.shape[0] / data_B.shape[0])), 1))[:data_A.shape[0], :]
            data_A_repeated = data_A
        else:
            # 如果data_B的样本数更多，重复data_A
            data_A_repeated = np.tile(data_A, (int(np.ceil(data_B.shape[0] / data_A.shape[0])), 1))[:data_B.shape[0], :]
            data_B_repeated = data_B

        
        # 使用最小列数作为循环范围
        for i in range(min_columns):
            mi_r = mutual_info_regression(data_A_repeated[:, i].reshape(-1, 1), data_B_repeated[:, i])
            mutual_info_regression_list.append(mi_r[0])    

        # 计算平均互信息
        average_mutual_info_r = np.mean(mutual_info_regression_list)
        # print(f"Mutual Information Regression Average: {average_mutual_info_r}")
        return(average_mutual_info_r)

    def mutual_info_regression_flattern(self,df_A=pd.DataFrame,df_B=pd.DataFrame):    
        data_A = df_A.values
        data_B = df_B.values
        # 获取两个数据集的样本数    
        len_A = data_A.shape[0]
        len_B = data_B.shape[0]
        # 取较小的样本数
        min_len = min(len_A, len_B)

        # 判断哪个数据集的样本数更多
        if data_A.shape[0] > data_B.shape[0]:
            # 如果data_A的样本数更多，重复data_B
            data_B_repeated = np.tile(data_B, (int(np.ceil(data_A.shape[0] / data_B.shape[0])), 1))[:data_A.shape[0], :]
            data_A_repeated = data_A
        else:
            # 如果data_B的样本数更多，重复data_A
            data_A_repeated = np.tile(data_A, (int(np.ceil(data_B.shape[0] / data_A.shape[0])), 1))[:data_B.shape[0], :]
            data_B_repeated = data_B


        # 将数据展平为一维数组
        flattened_A = data_A_repeated.flatten()
        flattened_B = data_B_repeated.flatten()

        # 计算互信息分数
        mi_r = mutual_info_regression(flattened_A.reshape(-1, 1), flattened_B)
        # print(f"Mutual Information Regression (Flattened): {mi_r[0]}")
        return(mi_r[0])

    def calculate_ssim(self,df_A: pd.DataFrame, df_B: pd.DataFrame):
        # 确保两个数据集的形状匹配
        if df_A.shape != df_B.shape:
            raise ValueError("The shape of both dataframes must be the same")
        # 处理缺失值（例如，使用0值填充）
        df_A = df_A.fillna(0)
        df_B = df_B.fillna(0)
        # 将数据转换为numpy数组
        data_A = df_A.values
        data_B = df_B.values
        # 计算SSIM
        data_range = data_B.max() - data_B.min()
        ssim_value, ssim_img = ssim(data_A, data_B, full=True, data_range=data_range)
        # print(f"SSIM: {ssim_value}")

        # 可视化SSIM图像
        # 可视化
        plt.figure(figsize=(10, 3))

        # 原始图像
        plt.subplot(1, 3, 1)
        plt.title('Data A (RAW)')
        plt.imshow(data_A, aspect='auto', cmap='gray')
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.title('Data B (RAW)')
        plt.imshow(data_B, aspect='auto', cmap='gray')
        plt.colorbar()
        
        plt.subplot(1, 3, 3)
        plt.imshow(ssim_img, aspect='auto', cmap='gray')
        plt.title(f'SSIM Image: {ssim_value}')
        plt.colorbar()
        plt.show()
        return ssim_value, ssim_img

    def calculate_ssim_components(self,df_A: pd.DataFrame, df_B: pd.DataFrame, method='max_range'):
        # 确保两个数据集的形状匹配
        img1 = df_A.values
        img2 = df_B.values

        # 计算动态范围
        if method == 'max_range':
            # 计算动态范围 方法1 先计算动态范围，然后选择最大的
            data_range_1 = img1.max() - img1.min()
            data_range_2 = img2.max() - img2.min()
            data_range = max(data_range_1, data_range_2)
        else:        
            # 计算动态范围 方法2 使用两张图像的最大值和最小值的差
            global_max = max(img1.max(), img2.max())
            global_min = min(img1.min(), img2.min())
            data_range = global_max - global_min

        # 计算亮度、对比度和结构分量
        # 常数
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        C3 = C2 / 2

        # 确保两个图像的形状匹配
        if img1.shape != img2.shape:
            raise ValueError("The shape of both images must be the same")

        # 计算亮度分量
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        luminance = (2 * mu1 * mu2 + C1) / (mu1**2 + mu2**2 + C1)

        # 计算对比度分量
        sigma1 = np.std(img1)
        sigma2 = np.std(img2)
        contrast = (2 * sigma1 * sigma2 + C2) / (sigma1**2 + sigma2**2 + C2)

        # 计算结构分量
        covariance = np.mean((img1 - mu1) * (img2 - mu2))
        structure = (covariance + C3) / (sigma1 * sigma2 + C3)

        # print(f"Luminance: {luminance}, Contrast: {contrast}, Structure: {structure}")

        return luminance, contrast, structure

    def luminance(self,df_A: pd.DataFrame, df_B: pd.DataFrame, method='max_range'):
        # 确保两个数据集的形状匹配
        img1 = df_A.values
        img2 = df_B.values

        # 计算动态范围
        if method == 'max_range':
            # 计算动态范围 方法1 先计算动态范围，然后选择最大的
            data_range_1 = img1.max() - img1.min()
            data_range_2 = img2.max() - img2.min()
            data_range = max(data_range_1, data_range_2)
        else:        
            # 计算动态范围 方法2 使用两张图像的最大值和最小值的差
            global_max = max(img1.max(), img2.max())
            global_min = min(img1.min(), img2.min())
            data_range = global_max - global_min

        # 计算亮度、对比度和结构分量
        # 常数
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        C3 = C2 / 2

        # 确保两个图像的形状匹配
        if img1.shape != img2.shape:
            raise ValueError("The shape of both images must be the same")

        # 计算亮度分量
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        luminance = (2 * mu1 * mu2 + C1) / (mu1**2 + mu2**2 + C1)

        # 计算对比度分量
        sigma1 = np.std(img1)
        sigma2 = np.std(img2)
        contrast = (2 * sigma1 * sigma2 + C2) / (sigma1**2 + sigma2**2 + C2)

        # 计算结构分量
        covariance = np.mean((img1 - mu1) * (img2 - mu2))
        structure = (covariance + C3) / (sigma1 * sigma2 + C3)

        # print(f"Luminance: {luminance}, Contrast: {contrast}, Structure: {structure}")

        return luminance

    def contrast(self,df_A: pd.DataFrame, df_B: pd.DataFrame, method='max_range'):
        # 确保两个数据集的形状匹配
        img1 = df_A.values
        img2 = df_B.values

        # 计算动态范围
        if method == 'max_range':
            # 计算动态范围 方法1 先计算动态范围，然后选择最大的
            data_range_1 = img1.max() - img1.min()
            data_range_2 = img2.max() - img2.min()
            data_range = max(data_range_1, data_range_2)
        else:        
            # 计算动态范围 方法2 使用两张图像的最大值和最小值的差
            global_max = max(img1.max(), img2.max())
            global_min = min(img1.min(), img2.min())
            data_range = global_max - global_min

        # 计算亮度、对比度和结构分量
        # 常数
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        C3 = C2 / 2

        # 确保两个图像的形状匹配
        if img1.shape != img2.shape:
            raise ValueError("The shape of both images must be the same")

        # 计算亮度分量
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        luminance = (2 * mu1 * mu2 + C1) / (mu1**2 + mu2**2 + C1)

        # 计算对比度分量
        sigma1 = np.std(img1)
        sigma2 = np.std(img2)
        contrast = (2 * sigma1 * sigma2 + C2) / (sigma1**2 + sigma2**2 + C2)

        # 计算结构分量
        covariance = np.mean((img1 - mu1) * (img2 - mu2))
        structure = (covariance + C3) / (sigma1 * sigma2 + C3)

        # print(f"Luminance: {luminance}, Contrast: {contrast}, Structure: {structure}")

        return contrast

    def structure(self,df_A: pd.DataFrame, df_B: pd.DataFrame, method='max_range'):
        # 确保两个数据集的形状匹配
        img1 = df_A.values
        img2 = df_B.values

        # 计算动态范围
        if method == 'max_range':
            # 计算动态范围 方法1 先计算动态范围，然后选择最大的
            data_range_1 = img1.max() - img1.min()
            data_range_2 = img2.max() - img2.min()
            data_range = max(data_range_1, data_range_2)
        else:        
            # 计算动态范围 方法2 使用两张图像的最大值和最小值的差
            global_max = max(img1.max(), img2.max())
            global_min = min(img1.min(), img2.min())
            data_range = global_max - global_min

        # 计算亮度、对比度和结构分量
        # 常数
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        C3 = C2 / 2

        # 确保两个图像的形状匹配
        if img1.shape != img2.shape:
            raise ValueError("The shape of both images must be the same")

        # 计算亮度分量
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        luminance = (2 * mu1 * mu2 + C1) / (mu1**2 + mu2**2 + C1)

        # 计算对比度分量
        sigma1 = np.std(img1)
        sigma2 = np.std(img2)
        contrast = (2 * sigma1 * sigma2 + C2) / (sigma1**2 + sigma2**2 + C2)

        # 计算结构分量
        covariance = np.mean((img1 - mu1) * (img2 - mu2))
        structure = (covariance + C3) / (sigma1 * sigma2 + C3)

        # print(f"Luminance: {luminance}, Contrast: {contrast}, Structure: {structure}")

        return structure

    def Euclidean(self,df_A: pd.DataFrame, df_B: pd.DataFrame) -> float:
        return np.linalg.norm(df_A.values.ravel() - df_B.values.ravel())

    def Manhattan(self,df_A: pd.DataFrame, df_B: pd.DataFrame) -> float:
        return np.sum(np.abs(df_A.values.ravel() - df_B.values.ravel()))

    def Chebyshev(self,df_A: pd.DataFrame, df_B: pd.DataFrame) -> float:
        return np.max(np.abs(df_A.values.ravel() - df_B.values.ravel()))

    def Minkowski(self,df_A: pd.DataFrame, df_B: pd.DataFrame, p: float = 3) -> float:
        return np.sum(np.abs(df_A.values.ravel() - df_B.values.ravel()) ** p) ** (1 / p)

    def Cosine(self,df_A: pd.DataFrame, df_B: pd.DataFrame) -> float:
        A_flat = df_A.values.ravel()
        B_flat = df_B.values.ravel()
        return 1 - np.dot(A_flat, B_flat) / (np.linalg.norm(A_flat) * np.linalg.norm(B_flat))

    def Correlation(self,df_A: pd.DataFrame, df_B: pd.DataFrame) -> float:
        A_flat = df_A.values.ravel()
        B_flat = df_B.values.ravel()
        A_mean = A_flat - np.mean(A_flat)
        B_mean = B_flat - np.mean(B_flat)
        return 1 - np.dot(A_mean, B_mean) / (np.linalg.norm(A_mean) * np.linalg.norm(B_mean))

    def Jaccard(self,df_A: pd.DataFrame, df_B: pd.DataFrame) -> float:
        A_flat = df_A.values.ravel().astype(bool)
        B_flat = df_B.values.ravel().astype(bool)
        intersection = np.sum(A_flat & B_flat)
        union = np.sum(A_flat | B_flat)
        return 1 - intersection / union

    def Dice(self,df_A: pd.DataFrame, df_B: pd.DataFrame) -> float:
        A_flat = df_A.values.ravel().astype(bool)
        B_flat = df_B.values.ravel().astype(bool)
        intersection = np.sum(A_flat & B_flat)
        return 1 - (2 * intersection) / (np.sum(A_flat) + np.sum(B_flat))

    def Kulsinski(self,df_A: pd.DataFrame, df_B: pd.DataFrame) -> float:
        A_flat = df_A.values.ravel().astype(bool)
        B_flat = df_B.values.ravel().astype(bool)
        intersection = np.sum(A_flat & B_flat)
        n = len(A_flat)
        return (n - intersection + np.sum(A_flat != B_flat)) / (n + np.sum(A_flat != B_flat))

    def Rogers_Tanimoto(self,df_A: pd.DataFrame, df_B: pd.DataFrame) -> float:
        A_flat = df_A.values.ravel().astype(bool)
        B_flat = df_B.values.ravel().astype(bool)
        n = len(A_flat)
        return (np.sum(A_flat != B_flat) + np.sum(~A_flat & ~B_flat)) / n

    def Russell_Rao(self,df_A: pd.DataFrame, df_B: pd.DataFrame) -> float:
        A_flat = df_A.values.ravel().astype(bool)
        B_flat = df_B.values.ravel().astype(bool)
        return np.sum(A_flat & B_flat) / len(A_flat)

    def Sokal_Michener(self,df_A: pd.DataFrame, df_B: pd.DataFrame) -> float:
        A_flat = df_A.values.ravel().astype(bool)
        B_flat = df_B.values.ravel().astype(bool)
        n = len(A_flat)
        return (np.sum(A_flat == B_flat) + np.sum(~A_flat & ~B_flat)) / n

    def Sokal_Sneath(self,df_A: pd.DataFrame, df_B: pd.DataFrame) -> float:
        A_flat = df_A.values.ravel().astype(bool)
        B_flat = df_B.values.ravel().astype(bool)
        intersection = np.sum(A_flat & B_flat)
        return (2 * intersection) / (np.sum(A_flat) + np.sum(B_flat))

    def Yule(self,df_A: pd.DataFrame, df_B: pd.DataFrame) -> float:
        A_flat = df_A.values.ravel().astype(bool)
        B_flat = df_B.values.ravel().astype(bool)
        n = len(A_flat)
        return (np.sum(A_flat & ~B_flat) + np.sum(~A_flat & B_flat)) / n

    def Hsim_Distance(self, df_A: pd.DataFrame, df_B: pd.DataFrame):
        a = df_A.values.ravel()
        b = df_B.values.ravel()
        tmp = []
        result = 0
        for i in range(min(len(a), len(b))):
            tmp.append(1.0 / (1 + np.abs(a[i] - b[i])))

        # print(tmp)
        result = np.sum(tmp) / (min([len(a), len(b)]))
        return (result)

    def Close_Distance(self, df_A: pd.DataFrame, df_B: pd.DataFrame):
        a = df_A.values.ravel()
        b = df_B.values.ravel()
        tmp = []
        result = 0
        for i in range(min([len(a), len(b)])):
            tmp.append(np.power(np.e, -np.abs(a[i] - b[i])))

        # print(tmp)
        result = np.sum(tmp) / (min([len(a), len(b)]))
        return (result)


    def apply_function_to_df_pairs(self):
        n = len(self.df_list)
        result = [[None for _ in range(n)] for _ in range(n)]
        
        # for i, df1 in enumerate(self.df_list):
        #     for j, df2 in enumerate(self.df_list):
        #         compare_selected = self.Compare_selector.currentText()
        #         if compare_selected == 'mutual_info_score_unflattern':
        #             result[i][j] = self.mutual_info_score_unflattern(df1, df2)
        #         elif compare_selected == 'mutual_info_score_flattern':
        #             result[i][j] = self.mutual_info_score_flattern(df1, df2)
        #         elif compare_selected == 'mutual_info_regression_unflattern':
        #             result[i][j] = self.mutual_info_regression_unflattern(df1, df2)
        #         elif compare_selected == 'mutual_info_regression_flattern':
        #             result[i][j] = self.mutual_info_regression_flattern(df1, df2)
        #         elif compare_selected == 'calculate_ssim':
        #             result[i][j] = self.calculate_ssim(df1, df2)
        #         elif compare_selected == 'calculate_ssim_components':
        #             result[i][j] = self.calculate_ssim_components(df1, df2)
        #         elif compare_selected == 'luminance':
        #             result[i][j] = self.luminance(df1, df2)
        #         elif compare_selected == 'contrast':
        #             result[i][j] = self.contrast(df1, df2)
        #         elif compare_selected == 'structure':
        #             result[i][j] = self.structure(df1, df2)
        #         elif compare_selected == 'Euclidean':
        #             result[i][j] = self.Euclidean(df1, df2)
        #         elif compare_selected == 'Manhattan':
        #             result[i][j] = self.Manhattan(df1, df2)
        #         elif compare_selected == 'Chebyshev':
        #             result[i][j] = self.Chebyshev(df1, df2)
        #         elif compare_selected == 'Minkowski':
        #             result[i][j] = self.Minkowski(df1, df2)
        #         elif compare_selected == 'Cosine':
        #             result[i][j] = self.Cosine(df1, df2)
        #         elif compare_selected == 'Correlation':
        #             result[i][j] = self.Correlation(df1, df2)
        #         elif compare_selected == 'Jaccard':
        #             result[i][j] = self.Jaccard(df1, df2)
        #         elif compare_selected == 'Dice':
        #             result[i][j] = self.Dice(df1, df2)
        #         elif compare_selected == 'Kulsinski':
        #             result[i][j] = self.Kulsinski(df1, df2)
        #         elif compare_selected == 'Rogers_Tanimoto':
        #             result[i][j] = self.Rogers_Tanimoto(df1, df2)
        #         elif compare_selected == 'Russell_Rao':
        #             result[i][j] = self.Russell_Rao(df1, df2)
        #         elif compare_selected == 'Sokal_Michener':
        #             result[i][j] = self.Sokal_Michener(df1, df2)
        #         elif compare_selected == 'Sokal_Sneath':
        #             result[i][j] = self.Sokal_Sneath(df1, df2)
        #         elif compare_selected == 'Yule':
        #             result[i][j] = self.Yule(df1, df2)
        #         elif compare_selected == 'Hsim_Distance':
        #             result[i][j] = self.Hsim_Distance(df1, df2)
        #         elif compare_selected == 'Close_Distance':
        #             result[i][j] = self.Close_Distance(df1, df2)
        #         else:
        #             result[i][j] = self.fun(df1, df2)

        # 将 DataFrame 转换为 numpy 数组
        arrays = np.array([df.values.ravel() for df in self.df_list])
        
        # 创建一个 n x n 的结果矩阵
        results = np.zeros((n, n))
        distance_list = ['mutual_info_score_unflattern',
                 'mutual_info_score_flattern',
                 'mutual_info_regression_unflattern',
                 'mutual_info_regression_flattern',
                 'calculate_ssim',
                 'luminance', 'contrast', 'structure',
                 "Euclidean", "Manhattan", "Chebyshev", 'Minkowski', 'Cosine', 'Correlation', 'Jaccard', 'Dice', 'Kulsinski', 'Rogers-Tanimoto', 'Russell-Rao', 'Sokal-Michener', 'Sokal-Sneath', 'Yule']

        # 使用广播机制计算距离
        for i in range(n):
            A = arrays[i]
            B = arrays[i:]  # 只计算上三角部分
            compare_selected = self.Compare_selector.currentText()
            if compare_selected == 'mutual_info_score_unflattern':
                func = self.mutual_info_score_unflattern
            elif compare_selected == 'mutual_info_score_flattern':
                func = self.mutual_info_score_flattern
            elif compare_selected == 'mutual_info_regression_unflattern':
                func = self.mutual_info_regression_unflattern
            elif compare_selected == 'mutual_info_regression_flattern':
                func = self.mutual_info_regression_flattern
            elif compare_selected == 'calculate_ssim':
                func = self.calculate_ssim
            elif compare_selected == 'luminance':
                func = self.luminance
            elif compare_selected == 'contrast':
                func = self.contrast
            elif compare_selected == 'structure':
                func = self.structure
            elif compare_selected == 'Euclidean':
                func = self.Euclidean
            elif compare_selected == 'Manhattan':
                func = self.Manhattan
            elif compare_selected == 'Chebyshev':
                func = self.Chebyshev
            elif compare_selected == 'Minkowski':
                func = self.Minkowski
            elif compare_selected == 'Cosine':
                func = self.Cosine
            elif compare_selected == 'Correlation':
                func = self.Correlation
            elif compare_selected == 'Jaccard':
                func = self.Jaccard
            elif compare_selected == 'Dice':
                func = self.Dice
            elif compare_selected == 'Kulsinski':
                func = self.Kulsinski
            elif compare_selected == 'Rogers_Tanimoto':
                func = self.Rogers_Tanimoto
            elif compare_selected == 'Russell_Rao':
                func = self.Russell_Rao
            elif compare_selected == 'Sokal_Michener':
                func = self.Sokal_Michener
            elif compare_selected == 'Sokal_Sneath':
                func = self.Sokal_Sneath
            elif compare_selected == 'Yule':
                func = self.Yule
            elif compare_selected == 'Hsim_Distance':
                func = self.Hsim_Distance
            elif compare_selected == 'Close_Distance':
                func = self.Close_Distance
            else:
                func = self.fun

            # distances = np.array([func(A, b) for b in B])
            # results[i, i:] = distances
            # results[i:, i] = distances  # 对称矩阵

            # 使用广播机制计算距离
            for i in range(n):
                A = arrays[i]
                B = arrays[i:]  # 只计算上三角部分
                # 使用 numpy 的广播机制计算距离，并将 A 和 b 转换为 DataFrame
                distances = np.array([func(pd.DataFrame(A.reshape(self.df_list[i].shape)), pd.DataFrame(b.reshape(self.df_list[i].shape))) for b in B])
                results[i, i:] = distances
                results[i:, i] = distances  # 对称矩阵
    
        
        
        labels = self.df_name_list
        result_df = pd.DataFrame(results, index=labels, columns=labels)  
        self.df = result_df
        model = PandasModel(self.df)
        self.table.setModel(model)

    # Example usage:
    # 2024年7月10日进度
    # Assuming `fun` is a function that takes two DataFrames and returns a value
    def fun(self, df1, df2):
        # Example function: return the sum of the shapes of the two DataFrames
        return df1.shape[0] + df2.shape[0]

    def clear_data(self):
        # 清空数据
        self.df = pd.DataFrame()
        self.table.setModel(PandasModel(self.df))
        self.df_list=[]
        self.df_name_list = []
        # 清空图表
        self.canvas.figure.clear()
        self.canvas.draw()
        self.data_selector.clear()

    def plot_data(self):
        # 清除之前的图像
        self.canvas.figure.clear()
        self.fig.clear()
        
        selected_items = self.data_selector.selectedItems()
        selected_texts = [item.text() for item in selected_items]
        
        # 获取选择的数量
        num_selected = len(selected_texts)
        
        if num_selected == 0:
            return
        
        # 计算行数和列数
        cols = math.ceil(math.sqrt(num_selected))
        rows = math.ceil(num_selected / cols)
        
        # 动态创建子图
        for i, text in enumerate(selected_texts):
            ax = self.fig.add_subplot(rows, cols, i + 1)
            index = self.df_name_list.index(text)
            # ax.imshow(self.df_list[index], cmap='gray', aspect='auto')

            tmp_data = self.df_list[index].to_numpy()


            selected_functions = self.function_selector.selectedItems()
            selected_func = [item.text() for item in selected_functions]

            try:
                if 'log_transform' in selected_func:
                    tmp_data = self.log_transform(tmp_data)
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Log Transform failed: {e}")

            try:
                if 'centering_transform' in selected_func:
                    tmp_data = self.centering_transform(tmp_data)
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Centering Transform failed: {e}")

            try:
                if 'z_score_normalization' in selected_func:
                    tmp_data = self.z_score_normalization(tmp_data)
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Z-Score Normalization failed: {e}")

            try:
                if 'standardize' in selected_func:
                    tmp_data = self.standardize(tmp_data)
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Standardization failed: {e}")

            try:
                if 'equalize_hist' in selected_func:
                    tmp_data = self.equalize_hist(tmp_data)
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Histogram Equalization failed: {e}")
            ax.imshow(tmp_data, cmap='gray', aspect='auto')
            ax.set_title(text)  # 设置小标题
        
        self.canvas.draw()

    def save_data(self):
        # 保存数据
        DataFileOutput, ok2 = QFileDialog.getSaveFileName(self,
                                                          'Save Data File','DataFileOutput',
                                                          'CSV Files (*.csv);;Excel Files (*.xlsx)')  # 数据文件保存输出
          
        if (DataFileOutput != ''):
            if ('csv' in DataFileOutput):
                self.df.to_csv(DataFileOutput, sep=',', encoding='utf-8')
            elif ('xls' in DataFileOutput):
                self.df.to_excel(DataFileOutput)



    def log_transform(self, data):
        return np.log1p(data)

    # def log_centering_transform(self, data):
    #     # 对数据进行对数变换
    #     log_data = np.log1p(data)  # 使用log1p避免log(0)的问题

    #     # 对变换后的数据进行中心化处理
    #     centered_log_data = log_data - np.mean(log_data, axis=0)

    #     return centered_log_data

    def centering_transform(self, data):
        # 中心化处理
        centered_log_data = data - np.mean(data, axis=0)
        return centered_log_data

    def z_score_normalization(self, data):
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    def standardize(self, data):
        scaler = StandardScaler()
        return scaler.fit_transform(data)

    def equalize_hist(self, data):
        return exposure.equalize_hist(data)


def main():
    # Linux desktop environments use an app's .desktop file to integrate the app
    # in to their application menus. The .desktop file of this app will include
    # the StartupWMClass key, set to app's formal name. This helps associate the
    # app's windows to its menu item.
    #
    # For association to work, any windows of the app must have WMCLASS property
    # set to match the value set in app's desktop file. For PySide6, this is set
    # with setApplicationName().

    # Find the name of the module that was used to start the app
    app_module = sys.modules["__main__"].__package__
    # Retrieve the app's metadata
    metadata = importlib.metadata.metadata(app_module)

    QtWidgets.QApplication.setApplicationName(metadata["Formal-Name"])

    app = QtWidgets.QApplication(sys.argv)
    main_window = CPSVisualizer()
    sys.exit(app.exec())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = CPSVisualizer()
    main_window.show()  # 显示主窗口
    sys.exit(app.exec())