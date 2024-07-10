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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from skimage.filters import threshold_otsu, threshold_local
from skimage import exposure

# 切换到当前目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 生成两个不同分布的高维度数据集
np.random.seed(0)
# data_A = np.random.normal(loc=0.0, scale=1.0, size=(100, 50))  # 正态分布
# data_B = np.random.uniform(low=-1.0, high=1.0, size=(150, 50))  # 均匀分布

# 当前目录下的一个Result目录下有Cu65_mat.csv和Zn66_mat.csv两个文件，用pandas读取数据，来替代data_A和data_B
data_A = pd.read_csv('Result/Cu65_mat.csv').values
data_B = pd.read_csv('Result/Zn66_mat.csv').values

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

print(f"Average Mutual Information (mutual_info_score): {average_mutual_info_s}")
print(f"Average Mutual Information (mutual_info_regression): {average_mutual_info_r}")
print(f"Linear Weighted Average Mutual Information: {average_mutual_info_linear}")
print(f"Log Weighted Average Mutual Information: {average_mutual_info_log}")
print(f"Exp Weighted Average Mutual Information: {average_mutual_info_exp}")
print(f"Normalized Weighted Average Mutual Information: {average_mutual_info_normalized}")

# 可视化，这部分做一下改进，这里的A和B都可以作为二维图像来呈现，就做一个单独A、单独B、AB对比这三个情况的吧？


# 假设 data_A 和 data_B 是二维数组
# 对 data_A 和 data_B 进行对数变换
data_A_log = np.log1p(data_A)
data_B_log = np.log1p(data_B)

# 标准化变换
scaler = StandardScaler()
data_A_log_norm = scaler.fit_transform(data_A_log)
data_B_log_norm = scaler.fit_transform(data_B_log)

# 计算标准化差值
diff_norm = (data_A_log_norm - data_B_log_norm) / (np.abs(data_A_log_norm) + np.abs(data_B_log_norm) + 1e-10)

# 使用自适应阈值进行二值化
block_size = 13  # 可以根据图像大小调整
binary_A_adaptive = data_A_log > threshold_local(data_A_log, block_size, offset=10)
binary_B_adaptive = data_B_log > threshold_local(data_B_log, block_size, offset=10)
binary_diff_adaptive = diff_norm > threshold_local(diff_norm, block_size, offset=10)

# 对数变换后的图像进行直方图均衡化
data_A_log_eq = exposure.equalize_hist(data_A_log)
data_B_log_eq = exposure.equalize_hist(data_B_log)
diff_norm_eq = exposure.equalize_hist(diff_norm)

# 使用自适应阈值进行二值化
binary_A_eq = data_A_log_eq > threshold_local(data_A_log_eq, block_size, offset=10)
binary_B_eq = data_B_log_eq > threshold_local(data_B_log_eq, block_size, offset=10)
binary_diff_eq = diff_norm_eq > threshold_local(diff_norm_eq, block_size, offset=10)

# 可视化
plt.figure(figsize=(20, 20))

# 原始图像
plt.subplot(4, 3, 1)
plt.title('Data A (Log Transformed)')
plt.imshow(data_A_log, aspect='auto', cmap='gray')
plt.colorbar()

plt.subplot(4, 3, 2)
plt.title('Data B (Log Transformed)')
plt.imshow(data_B_log, aspect='auto', cmap='gray')
plt.colorbar()

plt.subplot(4, 3, 3)
plt.title('Normalized Difference (Log Transformed)')
plt.imshow(diff_norm, aspect='auto', cmap='gray')
plt.colorbar()

# 标准化变换
plt.subplot(4, 3, 4)
plt.title('Data A (Standardized)')
plt.imshow(data_A_log_norm, aspect='auto', cmap='gray')
plt.colorbar()

plt.subplot(4, 3, 5)
plt.title('Data B (Standardized)')
plt.imshow(data_B_log_norm, aspect='auto', cmap='gray')
plt.colorbar()

plt.subplot(4, 3, 6)
plt.title('Normalized Difference (Standardized)')
plt.imshow(diff_norm, aspect='auto', cmap='gray')
plt.colorbar()

# 自适应阈值二值化
plt.subplot(4, 3, 7)
plt.title('Data A (Adaptive Threshold)')
plt.imshow(binary_A_adaptive, aspect='auto', cmap='gray')
plt.colorbar()

plt.subplot(4, 3, 8)
plt.title('Data B (Adaptive Threshold)')
plt.imshow(binary_B_adaptive, aspect='auto', cmap='gray')
plt.colorbar()

plt.subplot(4, 3, 9)
plt.title('Normalized Difference (Adaptive Threshold)')
plt.imshow(binary_diff_adaptive, aspect='auto', cmap='gray')
plt.colorbar()

# 直方图均衡化
plt.subplot(4, 3, 10)
plt.title('Data A (Equalized)')
plt.imshow(data_A_log_eq, aspect='auto', cmap='gray')
plt.colorbar()

plt.subplot(4, 3, 11)
plt.title('Data B (Equalized)')
plt.imshow(data_B_log_eq, aspect='auto', cmap='gray')
plt.colorbar()

plt.subplot(4, 3, 12)
plt.title('Normalized Difference (Equalized)')
plt.imshow(diff_norm_eq, aspect='auto', cmap='gray')
plt.colorbar()

plt.tight_layout()
plt.show()