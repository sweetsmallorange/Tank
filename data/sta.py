import pandas as pd

# 读取 CSV 文件
file_path = '15-1并联结果.csv'
data = pd.read_csv(file_path, sep=';')

# 根据 nse_tci 列升序排序
sorted_data = data.sort_values(by='nse_tci', ascending=True)

# 选择前 50 行
top_50 = sorted_data.head(50)

# 提取 basin 列
basin_data = top_50['basin']

processed_basins = []
for basin in basin_data:
    # 提取 _ 后面的数字
    number_part = basin.split('_')[-1]

    # 确保数字部分长度为 8 位，不足的部分用 0 补齐
    formatted_number = number_part.zfill(8)

    processed_basins.append(formatted_number)
# 将 basin 列数据存储到 50.txt 文件中
with open('CARTank_50basins_list.txt', 'w') as f:
    for basin in processed_basins:
        f.write(f"{basin}\n")

print("数据已成功存储到 50.txt 文件中。")
