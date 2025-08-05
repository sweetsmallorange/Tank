import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import glob

# 定义模型名称和路径
models = ['Transformer', 'LSTMMSVS2S', 'LSTMS2S']
nse_data = {}

# 1. 读取每个模型的NSE值
for model in models:
    # 构造CSV文件路径
    csv_path = f'/data2/zmz1/data_obs/{model}/pretrain_test_single/station_metrics.csv'

    # 读取CSV文件
    df_model = pd.read_csv(csv_path)

    # 提取NSE列并存储
    nse_data[model] = df_model['NSE'].values

# 2. 合并模型的NSE值到一个DataFrame
nse_df = pd.DataFrame(nse_data)

# 3. 读取SeriesTank模型的NSE值
series_tank_path = f"/data2/zmz1/Tank/runs_paper/attr9_448SeriesTank_Series4_runoff_Transformer_[448basins_list.txt,['CAMELS-US']]_[SeriesTank,varFalse,batch_size2048]_[epochs200]_[15,1][2025062323]/test/448basins/loss_data_(max_nse)*.pkl.txt"

# 使用glob模块找到符合条件的文件
series_tank_files = glob.glob(series_tank_path)

# 读取并合并SeriesTank的NSE值
# for file in series_tank_files:
df_series_tank = pd.read_csv(series_tank_path, delimiter=',')
nse_series_tank = df_series_tank['nse_tci'].values

# 将SeriesTank的NSE值添加到DataFrame中
nse_df['SeriesTank'] = nse_series_tank


# 4. 装配数据用于绘图
nse_values = {
    'model': np.concatenate([
        np.repeat('SeriesTank', len(nse_df['SeriesTank'])),
        np.repeat('Transformer', len(nse_df['Transformer'])),
        np.repeat('LSTMMSVS2S', len(nse_df['LSTMMSVS2S'])),
        np.repeat('LSTMS2S', len(nse_df['LSTMS2S']))
    ]),
    'nse': np.concatenate([
        nse_df['SeriesTank'].values,
        nse_df['Transformer'].values,
        nse_df['LSTMMSVS2S'].values,
        nse_df['LSTMS2S'].values
    ])
}

# 创建最终DataFrame
df = pd.DataFrame(nse_values)

# 设置绘图风格
sns.set(style="whitegrid")
colors = ['#FF9999', '#66B2FF', '#99E6E6', '#FFCC99', '#CCFFCC']
# 设置字体
# plt.rcParams['font.family'] = 'Times New Roman'

# 创建箱式图
plt.figure(figsize=(10, 6))
box_plot = sns.boxplot(x='model', y='nse', data=df, palette=colors, showfliers=False,
                       boxprops=dict(linewidth=2),  # 加粗箱体线条
                       whiskerprops=dict(linewidth=2),  # 加粗须线
                       capprops=dict(linewidth=2),  # 加粗顶部和底部线条
                       medianprops=dict(linewidth=2)
                       )

# colors = ['#FF7F7F', '#3399FF', '#66B2B2', '#FFB266', '#99CC99']
# colors = ['#FF4C4C', '#007ACC', '#339B9B', '#FF9933', '#66B266']

# 颜色映射
colors = {
    'SeriesTank': '#FF4C4C',
    'Transformer': '#007ACC',
    'LSTMMSVS2S': '#339B9B',
    'LSTMS2S': '#FF9933'
}
models = ['SeriesTank', 'Transformer', 'LSTMMSVS2S', 'LSTMS2S']
# 添加点集
for i, model in enumerate(models):
    subset = df[df['model'] == model]
    # 将点集放置在模型的中间偏右位置
    plt.scatter([i + 0.2] * len(subset), subset['nse'], color=colors[model], alpha=0.6, s=30, zorder=4, marker='X')

# 设置标题和标签
# 去掉标题和x轴标签
plt.title('')  # 去掉标题
plt.xlabel('')  # 去掉x轴标签
# 设置y轴标签
plt.ylabel('NSE', fontsize=18)
# 设置纵坐标从0.75开始，每隔0.05分段
plt.ylim(0., 1.0)  # 设置y轴范围
# plt.yticks(np.arange(0.30, 1.0, 0.05))  # 设置y轴刻度
# 设置字体及美观
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# 保存为EPS格式
plt.savefig('hydrology_boxplot_nse.eps', format='eps', bbox_inches='tight')

# 显示图形
plt.show()
