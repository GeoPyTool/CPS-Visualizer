# Intensity Vs Time CPS (counts per second) Visualization

'''
当前目录下有一个 CPS_data.csv 文件
该文件包含很多列数据，第1列是样本号，第2列是该样本号下的时间，后面的是各个成分的列
本代码的目的是将这些成分的时间序列数据进行预处理和
1. 读取 CPS_data.csv 文件
2. 将数据拆分成多个文件，每个文件都包含前2列，以及后面的某1列，文件名就用该列的成分的名字。比如第四列是Si，那么文件名就是Si.csv
'''

import os
import pandas as pd

# 切换到当前目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 初始化一个空的DataFrame用于存储拼接后的数据
combined_df = pd.DataFrame()

# 读取 CPS_data.csv 文件
data = pd.read_csv('combined.csv')
 
# 获取列名
columns = data.columns

# 遍历每一个成分列
for col in columns[3:]:
    # 创建一个新的 DataFrame，只包含前三列和当前成分列
    new_data = data.iloc[:, [0, 1, columns.get_loc(col)]]
    
    # 保存到新的 CSV 文件，文件名为成分列的名字
    new_data.to_csv(f'{col}.csv', index=False)

