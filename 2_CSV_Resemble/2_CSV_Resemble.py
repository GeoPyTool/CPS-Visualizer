# Intensity Vs Time CPS (counts per second) Visualization

'''
当前目录下有一很多个csv 文件
该文件包含很多列数据，第1列是样本号，第2列是该样本号下的时间，第3列是成分名和成分的CPS值
本代码的目的是将这些成分的时间序列数据进进一步处理
首先要将第一行也删除，然后删除掉第2列，只保留第1列和第3列的数值。
然后第1列的值每变化一次，就将同一行的第3列数值挪到第3列数据原始位置的右侧，以此类推，第1列每变一次值，就往右多挪出来一列，最终有了一个矩形
'''

import os
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt

# 切换到当前目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 获取当前目录下所有的csv文件
csv_files = glob.glob("*.csv")

# 遍历每一个csv文件
for file in csv_files:
    # 读取csv文件
    data = pd.read_csv(file)
    
    # 删除第一行
    data = data.iloc[1:]
    
    # 删除第二列，只保留第一列和第三列和
    data = data.iloc[:, [0, 2]]
    
    # 获取第一列的唯一值
    unique_samples = data.iloc[:, 0].unique()
    
    # 创建一个新的DataFrame用于存储结果
    result_df = pd.DataFrame()
    
    # 遍历每一个唯一的样本号
    for sample in unique_samples:
        # 获取当前样本号对应的所有行
        sample_data = data[data.iloc[:, 0] == sample]
        
        # 获取当前样本号对应的成分CPS值
        cps_values = sample_data.iloc[:, 1].values
        
        # 将成分CPS值添加到结果DataFrame中
        result_df[sample] = pd.Series(cps_values)

    # 删除result_df取消掉index
    result_df = result_df.reset_index(drop=True)

    print(result_df.head())

    # 重置索引并删除第一行
    result_df.reset_index(drop=True, inplace=True)   
    # 转置结果DataFrame，使其成为矩形，与原始照片形状匹配，适合后续可视化
    result_df = result_df.T
        
    # 删除转置后的第一行
    result_df.columns = result_df.iloc[0]
    result_df = result_df[1:]    

    # 替换文件名中的 .csv 为 _mat.csv
    output_file = file.replace('.csv', '_mat.csv')

    # 保存处理后的数据到新的csv文件
    result_df.to_csv(f'../Result/{output_file}', index=False)

    # 删除第一列和第一行
    result_df = result_df.iloc[1:, 1:]
    
    # 对数化处理
    result_df = result_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    result_df = np.log1p(result_df)       
    
    # 将result_df的每个点的值作为灰度值，然后整体保存成一个图片文件
    plt.imshow(result_df, cmap='gray', aspect='auto')
    plt.colorbar()
    img_file = file.replace('.csv', '')
    plt.title(f'Grayscale Image of {file}')
    plt.savefig(f'../Result_img/{img_file}.png')
    plt.close()
