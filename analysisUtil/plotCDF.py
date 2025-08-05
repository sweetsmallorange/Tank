import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# 设置模型和对应文件路径
# models = ["TRM-Tank", "LSTM-S2S", "Pyramidal Transformer", "RR-former"]
models = ["15-1", "30-1", "60-1", "90-1"]
# file_paths = {
#     "TRM-Tank": "/data2/zmz1/Tank/runs_paper/448SeriesTank_Series4_runoff_Transformer_[448basins_list.txt,['CAMELS-US']]_[SeriesTank,varFalse,batch_size2048]_[epochs200]_[15,1][2025031212]/test/448basins/loss_data_(max_nse)*.pkl.txt",
#     "LSTM-S2S": "/data2/zmz1/data_obs/LSTM-S2S/pretrain_test_single/station_metrics.csv",
#     "RR-former": "/data2/zmz1/data_obs/Transformer/pretrain_test_single/station_metrics.csv",
#     "Pyramidal Transformer": "/data2/zmz1/data_obs/Pyramidal Transformer/pretrain_test_single/calc_nse.csv"
# }


# file_paths = {
#     "SeriesTank": "/data2/zmz1/Tank/runs_paper/448SeriesTank_Series4_runoff_Transformer_[448basins_list.txt,['CAMELS-US']]_[SeriesTank,varFalse,batch_size2048]_[epochs200]_[15,1][2025031212]/test/448basins/loss_data_(max_nse)*.pkl.txt",
#     "LSTMS2S": "/data2/zmz1/data_obs/LSTM-S2S/pretrain_test_single/station_metrics.csv",
#     "RR-former": "/data2/zmz1/data_obs/Transformer/pretrain_test_single/station_metrics.csv",
#     "Pyramidal Transformer": "/data2/zmz1/data_obs/Pyramidal Transformer/pretrain_test_single/calc_nse.csv"
# }

file_paths = {
    "15-1": "/data2/zmz1/Tank/runs_paper/448SeriesTank_Series4_runoff_Transformer_[448basins_list.txt,['CAMELS-US']]_[SeriesTank,varFalse,batch_size2048]_[epochs200]_[15,1][2025031212]/test/448basins/loss_data_(max_nse)*.pkl.txt",
    "30-1": "/data2/zmz1/Tank/runs_paper/448SeriesTank_Series4_runoff_Transformer_[448basins_list.txt,['CAMELS-US']]_[SeriesTank,varFalse,batch_size2048]_[epochs200]_[30,1][2025031311]/test/448basins/loss_data_(max_nse)*.pkl.txt",
    "60-1": "/data2/zmz1/Tank/runs_paper/448SeriesTank_Series4_runoff_Transformer_[448basins_list.txt,['CAMELS-US']]_[SeriesTank,varFalse,batch_size2048]_[epochs200]_[60,1][2025031311]/test/448basins/loss_data_(max_nse)*.pkl.txt",
    "90-1": "/data2/zmz1/Tank/runs_paper/448SeriesTank_Series4_runoff_Transformer_[448basins_list.txt,['CAMELS-US']]_[SeriesTank,varFalse,batch_size2048]_[epochs200]_[90,1][2025031616]/test/448basins/loss_data_(max_nse)*.pkl.txt"
}

# column_names = {
#     "TRM-Tank": "kge_tci",
#     "LSTM-S2S": "KGE",
#     "RR-former": "KGE",
#     "Pyramidal Transformer": "kge0"
# }
# column_names = {
#     "TRM-Tank": "nse_tci",
#     "LSTM-S2S": "NSE",
#     "RR-former": "NSE",
#     "Pyramidal Transformer": "nse0"
# }
# column_names = {
#     "15-1": "kge_tci",
#     "30-1": "kge_tci",
#     "60-1": "kge_tci",
#     "90-1": "kge_tci"
# }
column_names = {
    "15-1": "nse_tci",
    "30-1": "nse_tci",
    "60-1": "nse_tci",
    "90-1": "nse_tci"
}

# 创建输出目录
output_dir = "/data2/zmz1/Tank/analysisUtil/cdf"
os.makedirs(output_dir, exist_ok=True)

# 存储 KGE 数据
kge_data = {}

# 读取数据
for model in models:
    file_path = file_paths[model]
    column = column_names[model]

    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".txt"):
            # 使用逗号分隔符读取 .txt 文件
            df = pd.read_csv(file_path, sep=r',', engine='python')
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")

        if column not in df.columns:
            raise KeyError(f"列 '{column}' 不存在于 {model} 的文件中")

        kges = df[column].dropna().values
        if len(kges) != 448:
            print(f"警告: {model} 只读取到 {len(kges)} 个站点，预期为 448 个")
        else:
            print(f"{model}: 成功读取 {len(kges)} 个站点的 KGE 值")
        kge_data[model] = kges

    except Exception as e:
        print(f"读取 {model} 的文件时出错: {e}")
        continue

# 检查所有模型是否都读取了 448 个站点
if any(len(vals) != 448 for vals in kge_data.values()):
    print("部分模型未读取到完整的 448 个站点，继续绘图可能影响准确性。")
else:
    print("所有模型均成功读取 448 个站点的 KGE 值")

# 设置绘图样式
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2.2,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.6,
    'figure.figsize': (10, 6),
    'savefig.dpi': 300,
    'savefig.format': 'png',
    'text.usetex': False
})

fig, ax = plt.subplots()

# 配色方案（Tableau风格）
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 蓝、橙、绿、红

# 插值点数量
x_smooth = np.linspace(0, 1, 1000)

for i, model in enumerate(models):
    if model not in kge_data:
        continue
    kges = kge_data[model]

    # 排序并裁剪 KGE 到合理范围 [0, 1]
    sorted_kges = np.sort(kges)
    sorted_kges = np.clip(sorted_kges, 0, 1)

    # 使用 KDE 平滑 PDF，然后积分得到 CDF
    kde = gaussian_kde(sorted_kges, bw_method=0.05)  # 可调整带宽，如 0.01~0.1
    pdf = kde(x_smooth)
    cdf_smooth = np.cumsum(pdf) / len(pdf)  # 归一化

    # 绘制平滑曲线
    ax.plot(x_smooth, cdf_smooth, label=model, color=colors[i], alpha=0.85)

# 设置图表样式
ax.set_xlim(0, 1)
ax.set_xlabel('NSE Value')
ax.set_ylabel('CDF')
# ax.set_title('CDF of KGE Values Across Models')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)
fig.tight_layout()

# 保存图像
output_path_base = os.path.join(output_dir, "nse_cdf")
# output_path_base = os.path.join(output_dir, "kge_cdf")
plt.savefig(output_path_base + ".eps", format='eps', dpi=1200)
plt.savefig(output_path_base + ".png", format='png', dpi=300)
plt.close()

print(f"图表已保存至: {output_path_base}.eps 和 {output_path_base}.png")