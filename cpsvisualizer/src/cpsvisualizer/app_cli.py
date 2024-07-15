"""
Calculation and visualization of CPS (counts per second) for ICPMS scan data.
"""
import argparse
import importlib.metadata
import sys
import warnings
warnings.filterwarnings("ignore")
import glob
import importlib.metadata
import itertools
import json
import math
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import re
import sqlite3
import sys
import warnings


from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.patches import ConnectionStyle, Polygon
from matplotlib.collections import PatchCollection
from matplotlib import collections
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator


from joblib import Parallel, delayed
try:
    from importlib import metadata as importlib_metadata
except ImportError:
    # Backwards compatibility - importlib.metadata was added in Python 3.8
    import importlib_metadata

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from scipy.ndimage import sobel
from scipy.stats import gmean
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from skimage import feature
from skimage.metrics import structural_similarity as ssim
from sklearn.preprocessing import StandardScaler
from skimage.filters import threshold_otsu, threshold_local
from skimage import exposure
from itertools import combinations

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] =  'truetype'

# # 获取当前解释器运行时的启动路径
# startup_path = os.getcwd()
# # 获取当前文件的绝对路径
# current_file_path = os.path.abspath(__file__)

# # 获取当前文件的目录
# current_directory = os.path.dirname(current_file_path)
# # 改变当前工作目录
# os.chdir(current_directory)


class CPS_CLI:
    def __init__(self, parent=None):
        self.init_data()
    def init_data(self):      
        self.df = pd.DataFrame()    
        self.df_list=[]
        self.trans_df_list=[]
        self.calc_df_list=[]
        self.result_df_dict={}
        self.df_name_list = []
        self.trans_function_list = ["log_transform","centering_transform","z_score_normalization","standardize","equalize_hist"]
        self.trans_function_used_dict ={}
        self.distance_function_list = ['Euclidean','Manhattan','Chebyshev','Minkowski','Cosine','Correlation','Jaccard','Dice','Kulsinski','Rogers_Tanimoto','Russell_Rao','Sokal_Michener','Sokal_Sneath','Yule','mutual_info_regression_flattern','mutual_info_regression_unflattern','mutual_info_score_flattern','mutual_info_score_unflattern','calculate_ssim','luminance','contrast','structure','Hsim_Distance','Close_Distance']


    def open_files(self, file_paths=[]):
        self.clear_data()
        # global current_directory 
        file_names = file_paths
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
 
    def clean_tmp_name(self,tmp_name):
        # Remove the extension
        name_without_ext = os.path.splitext(tmp_name)[0]
        # Remove everything after the first underscore
        cleaned_name = name_without_ext.split('_')[0]
        return cleaned_name

    def mutual_info_score_unflattern_old(self,df_A=pd.DataFrame,df_B=pd.DataFrame):
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

    def mutual_info_score_unflattern_bad(self, labels_true, labels_pred):
        n_samples, n_features = labels_true.shape
        
        # 使用并行计算来计算互信息
        mi_scores = Parallel(n_jobs=-1)(delayed(mutual_info_score)(labels_true[:, i], labels_pred[:, i]) for i in range(n_features))
        
        # 计算平均互信息
        mi = np.mean(mi_scores)
        return mi
    
    def mutual_info_score_unflattern(self, labels_true, labels_pred):
        labels_true = labels_true.values if isinstance(labels_true, pd.DataFrame) else labels_true
        labels_pred = labels_pred.values if isinstance(labels_pred, pd.DataFrame) else labels_pred
        
        n_samples, n_features = labels_true.shape
        
        # 使用并行计算来计算互信息
        mi_scores = Parallel(n_jobs=-1)(delayed(mutual_info_score)(labels_true[:, i], labels_pred[:, i]) for i in range(n_features))
        
        # 计算平均互信息
        mi = np.mean(mi_scores)
        return mi

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

    def mutual_info_regression_unflattern_old(self,df_A=pd.DataFrame,df_B=pd.DataFrame):
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

    def mutual_info_regression_unflattern(self, df_A=pd.DataFrame, df_B=pd.DataFrame):
        data_A = df_A.values
        data_B = df_B.values

        # 获取两个数据集的列数
        num_columns_A = data_A.shape[1]
        num_columns_B = data_B.shape[1]

        # 取列数的最小值作为循环的范围
        min_columns = min(num_columns_A, num_columns_B)

        # 判断哪个数据集的样本数更多，并重复较少的样本数
        if data_A.shape[0] > data_B.shape[0]:
            data_B_repeated = np.tile(data_B, (int(np.ceil(data_A.shape[0] / data_B.shape[0])), 1))[:data_A.shape[0], :]
            data_A_repeated = data_A
        else:
            data_A_repeated = np.tile(data_A, (int(np.ceil(data_B.shape[0] / data_A.shape[0])), 1))[:data_B.shape[0], :]
            data_B_repeated = data_B

        # 使用并行计算来计算互信息
        mutual_info_regression_list = Parallel(n_jobs=-1)(
            delayed(mutual_info_regression)(data_A_repeated[:, i].reshape(-1, 1), data_B_repeated[:, i])
            for i in range(min_columns)
        )

        # 计算平均互信息
        average_mutual_info_r = np.mean([mi_r[0] for mi_r in mutual_info_regression_list])
        return average_mutual_info_r

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

        # # 可视化SSIM图像
        # # 可视化
        # plt.figure(figsize=(10, 3))

        # # 原始图像
        # plt.subplot(1, 3, 1)
        # plt.title('Data A (RAW)')
        # plt.imshow(data_A, aspect='auto', cmap='gray')
        # plt.colorbar()

        # plt.subplot(1, 3, 2)
        # plt.title('Data B (RAW)')
        # plt.imshow(data_B, aspect='auto', cmap='gray')
        # plt.colorbar()
        
        # plt.subplot(1, 3, 3)
        # plt.imshow(ssim_img, aspect='auto', cmap='gray')
        # plt.title(f'SSIM Image: {ssim_value}')
        # plt.colorbar()
        # plt.show()
        # return ssim_value, ssim_img
        return ssim_value

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

    def Hsim_Distance(self, df_A: pd.DataFrame, df_B: pd.DataFrame) -> float:
        a = df_A.values.ravel()
        b = df_B.values.ravel()
        
        # 使用 NumPy 的矢量化操作计算距离
        differences = np.abs(a - b)
        exp_values = np.exp(-differences)
        
        # 计算结果
        result = np.sum(exp_values) / min(len(a), len(b))
        return result

    def Close_Distance(self, df_A: pd.DataFrame, df_B: pd.DataFrame) -> float:
        a = df_A.values.ravel()
        b = df_B.values.ravel()
        
        # 使用 NumPy 的矢量化操作计算距离
        differences = np.abs(a - b)
        exp_values = np.exp(-differences)
        
        # 计算结果
        result = np.sum(exp_values) / min(len(a), len(b))
        return result

    def log_transform(self, data):
        return np.log1p(data)

    def log_centering_transform(self, data):
        # 对数据进行对数变换
        log_data = np.log1p(data)  # 使用log1p避免log(0)的问题

        # 对变换后的数据进行中心化处理
        centered_log_data = log_data - np.mean(log_data, axis=0)

        return centered_log_data

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

    def clear_data(self):
        # 清空数据
        self.df = pd.DataFrame()    
        self.df_list=[]
        self.trans_df_list=[]
        self.calc_df_list=[]
        self.result_df_dict={}
        self.df_name_list = []
        self.trans_function_list = ["log_transform","centering_transform","z_score_normalization","standardize","equalize_hist"]
        self.trans_function_used_dict = {}
        self.distance_function_list = ['Euclidean','Manhattan','Chebyshev','Minkowski','Cosine','Correlation','Jaccard','Dice','Kulsinski','Rogers_Tanimoto','Russell_Rao','Sokal_Michener','Sokal_Sneath','Yule','mutual_info_regression_flattern','mutual_info_regression_unflattern','mutual_info_score_flattern','mutual_info_score_unflattern','calculate_ssim','luminance','contrast','structure','Hsim_Distance','Close_Distance']


    def trans_data(self,func_name_list =["log_transform","equalize_hist"]):
        # self.trans_function_list = ["log_transform","centering_transform","z_score_normalization","standardize","equalize_hist"]
        # self.distance_function_list = ['Euclidean','Manhattan','Chebyshev','Minkowski','Cosine','Correlation','Jaccard','Dice','Kulsinski','Rogers_Tanimoto','Russell_Rao','Sokal_Michener','Sokal_Sneath','Yule','mutual_info_regression_flattern','mutual_info_regression_unflattern','mutual_info_score_flattern','mutual_info_score_unflattern','calculate_ssim','luminance','contrast','structure','Hsim_Distance','Close_Distance']
        for index, text in enumerate(self.df_name_list):
            # index = self.df_name_list.index(text)
            tmp_data = self.df_list[index].to_numpy()
            self.trans_function_used_dict[text] = []

            for func_name in func_name_list:
                if func_name in self.trans_function_list:
                    func = getattr(self, func_name)
                    try:
                        tmp_data = func(tmp_data)
                        self.trans_function_used_dict[text].append(func_name)
                        print(func_name + f" success on {text}")
                    except Exception as e:
                        self.trans_function_used_dict[text].append(func_name + f" Failed with {e}"+ f" on {text}")
                        print(func_name + f" Failed with {e}"+ f" on {text}")
                        
            
            self.trans_df_list.append(pd.DataFrame(tmp_data))
        # print(self.trans_function_used_dict)


    def calc_data(self,func_name_list =["Euclidean","Yule"]):
        # self.distance_function_list = ['Euclidean','Manhattan','Chebyshev','Minkowski','Cosine','Correlation','Jaccard','Dice','Kulsinski','Rogers_Tanimoto','Russell_Rao','Sokal_Michener','Sokal_Sneath','Yule','mutual_info_regression_flattern','mutual_info_regression_unflattern','mutual_info_score_flattern','mutual_info_score_unflattern','calculate_ssim','luminance','contrast','structure','Hsim_Distance','Close_Distance']
        # 针对全部离函数，然后逐对计算距离
        # directory = current_directory
        directory = os.getcwd()
  
        for func_name in func_name_list:
            if func_name in self.distance_function_list:
                func = getattr(self, func_name)
                if func_name not in self.result_df_dict:
                    n = len(self.df_list)
                    # 将 DataFrame 转换为 numpy 数组
                    arrays = np.array([df.values.ravel() for df in self.df_list])        
                    # 创建一个 n x n 的结果矩阵
                    results = np.zeros((n, n))
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
                    self.result_df_dict[func_name] = result_df
                file_path = os.path.join(directory, f'{func_name}.csv')
                print(f'{func_name} file is save to {file_path}')
                try:
                    result_df.to_csv(file_path, sep=',', encoding='utf-8')
                except Exception:
                    pass

        # print(self.result_df_dict)
    def plot_data(self):
        # 自动排列成正方形比例
        # plt.figure()
        
        # 获取选择的数量
        num_selected = len(self.df_name_list)
        
        if num_selected == 0:
            return
        
        # 计算行数和列数
        num_subplots = math.ceil(math.sqrt(num_selected)) ** 2  
        rows = cols = int(math.sqrt(num_subplots))
        # 设置整个图形的大小，使其为正方形
        figsize = (8, 8)  # 例如，设置为10x10英寸的正方形

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # 动态创建子图
        for index, text in enumerate(self.df_name_list):
            row = index // cols
            col = index % cols
            ax = axes[row, col]
            # index = self.df_name_list.index(text)
            tmp_data = self.trans_df_list[index].to_numpy()   

            trans_function_used_str = "\n".join(self.trans_function_used_dict[text])

            ax.imshow(tmp_data, cmap='gray', aspect='auto')
            # ax.set_title(text)  # 设置小标题
            ax.set_title(f"{text}\n{trans_function_used_str}")
        # 隐藏未使用的子图的坐标轴
        for i in range(num_selected, rows * cols):
            row = i // cols
            col = i % cols
            fig.delaxes(axes[row, col])

        # 设置全局窗口标题
        fig.canvas.manager.set_window_title("CPS Data Visualization")
        # plt.suptitle("CPS Data Visualization", fontsize=16)
        plt.tight_layout()
    def silent_plot(self):
        # 保存图像并打印路径
        png_path = os.path.abspath('CPS_Data_Visualization.png')
        pdf_path = os.path.abspath('CPS_Data_Visualization.pdf')
        svg_path = os.path.abspath('CPS_Data_Visualization.svg')

        plt.savefig(png_path, dpi=600)
        plt.savefig(pdf_path)
        plt.savefig(svg_path)

        print('PNG file saved at:', png_path)
        print('PDF file saved at:', pdf_path)
        print('SVG file saved at:', svg_path)
        
        # 关闭当前图像
        plt.close()





def main(data_files_arg=None, functions_arg=None, operation_arg=None):
    main_app = CPS_CLI()
    
    if data_files_arg is None or functions_arg is None or operation_arg is None:
        if len(sys.argv) < 4:
            print("Usage 1: python app_cli.py 'data1.csv data2.csv data3.csv' 'func1 func2 func3' 'silent OR show'",
            '''Usage 2: /n pip install cpsvisualizer /n import cpsvisualizer /n cpsvisualizer.cli('Ag.csv Cu.csv Zn.csv Fe.csv', 'log_transform papa pupu pipi popo equalize_hist Euclidean Yule', 'silent')''')
            
            # print("Usage: python C:/Users/HP/Documents/GitHub/CPS-Visualizer/cpsvisualizer/src/cpsvisualizer/app_cli.py 'Ag.csv Cu.csv  Zn.csv  Fe.csv ' 'log_transform papa pupu pipi popo equalize_hist Euclidean Yule' 'silent OR show'")
            sys.exit(1)

        data_files_arg = sys.argv[1]
        functions_arg = sys.argv[2]
        operation_arg = sys.argv[3]

    data_files = list(dict.fromkeys(data_files_arg.split()))
    functions = list(dict.fromkeys(functions_arg.split()))
    operation = operation_arg

    print('Data Files are : ',data_files)
    # print('Data Files are : ',functions)

    # 拆分 functions 列表
    trans_functions = [func for func in functions if func in main_app.trans_function_list]
    distance_functions = [func for func in functions if func in main_app.distance_function_list]

    # 分别打印输出
    print('Trans Functions are:', trans_functions)
    print('Distance Calculations are:', distance_functions)
    print('Plot Option is : ',operation)
    main_app.open_files(data_files)
    main_app.trans_data(functions)
    main_app.calc_data(functions)
    main_app.plot_data()

    if 'silent' in operation:     
        main_app.silent_plot()
    else:   
        plt.show()
if __name__ == "__main__":
    main()

    # 1551+798 = 2349