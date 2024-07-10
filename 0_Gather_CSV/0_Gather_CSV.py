import os
import pandas as pd

# 获取当前目录下的所有CSV文件
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]

# 初始化一个空的DataFrame用于存储拼接后的数据
combined_df = pd.DataFrame()

# 遍历所有CSV文件
for csv_file in csv_files:
    if 'DE' in csv_file:
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        
        # 删除掉df的列名，设置第三行为列名
        df.columns = df.iloc[2]
        # 然后将前三行删掉
        df = df.iloc[3:]
        # 重置索引以确保索引唯一
        df.reset_index(drop=True, inplace=True)
        # 再删掉df的后三行，因为这些行是重复的
        df = df.iloc[:-3]
        # 打印前五行
        print(df.head())

        # 检查是否已经存在列名为'filename'的列
        if 'filename' not in df.columns:
            # 在最左侧添加一列，列名为'filename'，值为当前CSV文件名
            df.insert(0, 'filename', csv_file) 
        
        # 将当前CSV文件的数据拼接到combined_df中
        combined_df = pd.concat([combined_df, df], ignore_index=True)


# 将拼接后的数据写入一个新的CSV文件
combined_df.to_csv('combined.csv', index=False)

print("All CSV files have been combined into 'combined.csv'.")