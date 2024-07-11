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

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] =  'truetype'


# 切换到当前目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 生成两个不同分布的高维度数据集
np.random.seed(0)
# data_A = np.random.normal(loc=0.0, scale=1.0, size=(100, 50))  # 正态分布
# data_B = np.random.uniform(low=-1.0, high=1.0, size=(150, 50))  # 均匀分布

# 当前目录下的一个Result目录下有Cu65_mat.csv和Zn66_mat.csv两个文件，用pandas读取数据，来替代data_A和data_B
df_A = pd.read_csv('Result/Cu65_mat.csv')
df_B = pd.read_csv('Result/Zn66_mat.csv')

def mutual_info(df_A=pd.DataFrame,df_B=pd.DataFrame):
    data_A = df_A.values
    data_B = df_B.values

    # 计算平均互信息
    average_mutual_info_s = mutual_info_score_unflattern(df_A,df_B)
    average_mutual_info_r = mutual_info_regression_unflattern(df_A,df_B)

    mutual_info_s_flattern = mutual_info_score_flattern(df_A,df_B)
    mutual_info_r_flattern = mutual_info_regression_flattern(df_A,df_B)

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

    print(f"Average Mutual Information (mutual_info_score): {average_mutual_info_s}")
    print(f"Average Mutual Information (mutual_info_regression): {average_mutual_info_r}")
    print(f"Linear Weighted Average Mutual Information: {average_mutual_info_linear}")
    print(f"Log Weighted Average Mutual Information: {average_mutual_info_log}")
    print(f"Exp Weighted Average Mutual Information: {average_mutual_info_exp}")
    print(f"Normalized Weighted Average Mutual Information: {average_mutual_info_normalized}")
    print(f"Mutual Information Score (Flattened): {mutual_info_s_flattern}")
    print(f"Mutual Information Regression (Flattened): {mutual_info_r_flattern}")
          
def mutual_info_score_unflattern(df_A=pd.DataFrame,df_B=pd.DataFrame):
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
    print(f"Mutual Information Score Average: {average_mutual_info_s}")
    return(average_mutual_info_s)

def mutual_info_score_flattern(df_A=pd.DataFrame,df_B=pd.DataFrame):    
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
    print(f"Mutual Information Score (Flattened): {mi_s}")
    return(mi_s)

def mutual_info_regression_unflattern(df_A=pd.DataFrame,df_B=pd.DataFrame):
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
    print(f"Mutual Information Regression Average: {average_mutual_info_r}")
    return(average_mutual_info_r)

def mutual_info_regression_flattern(df_A=pd.DataFrame,df_B=pd.DataFrame):    
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
    print(f"Mutual Information Regression (Flattened): {mi_r[0]}")
    return(mi_r[0])

def calculate_ssim(df_A: pd.DataFrame, df_B: pd.DataFrame):
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
    print(f"SSIM: {ssim_value}")

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

def calculate_ssim_components(df_A: pd.DataFrame, df_B: pd.DataFrame, method='max_range'):
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

    print(f"Luminance: {luminance}, Contrast: {contrast}, Structure: {structure}")

    return luminance, contrast, structure

def luminance(df_A: pd.DataFrame, df_B: pd.DataFrame, method='max_range'):
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

    print(f"Luminance: {luminance}, Contrast: {contrast}, Structure: {structure}")

    return luminance

def contrast(df_A: pd.DataFrame, df_B: pd.DataFrame, method='max_range'):
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

    print(f"Luminance: {luminance}, Contrast: {contrast}, Structure: {structure}")

    return contrast

def structure(df_A: pd.DataFrame, df_B: pd.DataFrame, method='max_range'):
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

    print(f"Luminance: {luminance}, Contrast: {contrast}, Structure: {structure}")

    return structure

def Euclidean(df_A: pd.DataFrame, df_B: pd.DataFrame) -> float:
    return np.linalg.norm(df_A.values.ravel() - df_B.values.ravel())

def Manhattan(df_A: pd.DataFrame, df_B: pd.DataFrame) -> float:
    return np.sum(np.abs(df_A.values.ravel() - df_B.values.ravel()))

def Chebyshev(df_A: pd.DataFrame, df_B: pd.DataFrame) -> float:
    return np.max(np.abs(df_A.values.ravel() - df_B.values.ravel()))

def Minkowski(df_A: pd.DataFrame, df_B: pd.DataFrame, p: float = 3) -> float:
    return np.sum(np.abs(df_A.values.ravel() - df_B.values.ravel()) ** p) ** (1 / p)

def Cosine(df_A: pd.DataFrame, df_B: pd.DataFrame) -> float:
    A_flat = df_A.values.ravel()
    B_flat = df_B.values.ravel()
    return 1 - np.dot(A_flat, B_flat) / (np.linalg.norm(A_flat) * np.linalg.norm(B_flat))

def Correlation(df_A: pd.DataFrame, df_B: pd.DataFrame) -> float:
    A_flat = df_A.values.ravel()
    B_flat = df_B.values.ravel()
    A_mean = A_flat - np.mean(A_flat)
    B_mean = B_flat - np.mean(B_flat)
    return 1 - np.dot(A_mean, B_mean) / (np.linalg.norm(A_mean) * np.linalg.norm(B_mean))

def Jaccard(df_A: pd.DataFrame, df_B: pd.DataFrame) -> float:
    A_flat = df_A.values.ravel().astype(bool)
    B_flat = df_B.values.ravel().astype(bool)
    intersection = np.sum(A_flat & B_flat)
    union = np.sum(A_flat | B_flat)
    return 1 - intersection / union

def Dice(df_A: pd.DataFrame, df_B: pd.DataFrame) -> float:
    A_flat = df_A.values.ravel().astype(bool)
    B_flat = df_B.values.ravel().astype(bool)
    intersection = np.sum(A_flat & B_flat)
    return 1 - (2 * intersection) / (np.sum(A_flat) + np.sum(B_flat))

def Kulsinski(df_A: pd.DataFrame, df_B: pd.DataFrame) -> float:
    A_flat = df_A.values.ravel().astype(bool)
    B_flat = df_B.values.ravel().astype(bool)
    intersection = np.sum(A_flat & B_flat)
    n = len(A_flat)
    return (n - intersection + np.sum(A_flat != B_flat)) / (n + np.sum(A_flat != B_flat))

def Rogers_Tanimoto(df_A: pd.DataFrame, df_B: pd.DataFrame) -> float:
    A_flat = df_A.values.ravel().astype(bool)
    B_flat = df_B.values.ravel().astype(bool)
    n = len(A_flat)
    return (np.sum(A_flat != B_flat) + np.sum(~A_flat & ~B_flat)) / n

def Russell_Rao(df_A: pd.DataFrame, df_B: pd.DataFrame) -> float:
    A_flat = df_A.values.ravel().astype(bool)
    B_flat = df_B.values.ravel().astype(bool)
    return np.sum(A_flat & B_flat) / len(A_flat)

def Sokal_Michener(df_A: pd.DataFrame, df_B: pd.DataFrame) -> float:
    A_flat = df_A.values.ravel().astype(bool)
    B_flat = df_B.values.ravel().astype(bool)
    n = len(A_flat)
    return (np.sum(A_flat == B_flat) + np.sum(~A_flat & ~B_flat)) / n

def Sokal_Sneath(df_A: pd.DataFrame, df_B: pd.DataFrame) -> float:
    A_flat = df_A.values.ravel().astype(bool)
    B_flat = df_B.values.ravel().astype(bool)
    intersection = np.sum(A_flat & B_flat)
    return (2 * intersection) / (np.sum(A_flat) + np.sum(B_flat))

def Yule(df_A: pd.DataFrame, df_B: pd.DataFrame) -> float:
    A_flat = df_A.values.ravel().astype(bool)
    B_flat = df_B.values.ravel().astype(bool)
    n = len(A_flat)
    return (np.sum(A_flat & ~B_flat) + np.sum(~A_flat & B_flat)) / n




distance_list = ['mutual_info_score_unflattern',
                 'mutual_info_score_flattern',
                 'mutual_info_regression_unflattern',
                 'mutual_info_regression_flattern',
                 'calculate_ssim',
                 'luminance', 'contrast', 'structure',
                 "Euclidean", "Manhattan", "Chebyshev", 'Minkowski', 'Cosine', 'Correlation', 'Jaccard', 'Dice', 'Kulsinski', 'Rogers-Tanimoto', 'Russell-Rao', 'Sokal-Michener', 'Sokal-Sneath', 'Yule']



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

def min_max_scaled_normalization(data):
    min_max_scaled = (data - np.min(data)) / (np.max(data) - np.min(data))
    return min_max_scaled

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

mutual_info(df_A,df_B)
# mutual_info_score_unflattern(df_A,df_B)
# mutual_info_score_flattern(df_A,df_B)
# mutual_info_regression_unflattern(df_A,df_B)
# mutual_info_regression_flattern(df_A,df_B)
# visual_diff(df_A,df_B)


calculate_ssim_components(df_A, df_B)
# calculate_ssim(df_A, df_B)