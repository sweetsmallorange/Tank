import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 示例数据
np.random.seed(42)
data = {
    'SeriesTank': np.random.normal(0.93, 0.05, 100),
    'RR-Former': np.random.normal(0.87, 0.03, 100),
    'LSTM-MSV-S2S': np.random.normal(0.83, 0.02, 100),
    'LSTM-S2S': np.random.normal(0.82, 0.02, 100)
}

df = pd.DataFrame(data)

# 将数据转换为长格式，以便绘图
df_long = df.melt(var_name='Model', value_name='Value')

# 设置绘图风格和颜色
sns.set(style="whitegrid")
colors = ['#FF9999', '#66B2FF', '#99E6E6', '#FFCC99', '#CCFFCC']

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(6, 6))

# 绘制箱式图
bplot = sns.boxplot(x='Model', y='Value', data=df_long, palette=colors, ax=ax)

# 获取箱式图每个箱子的颜色
box_colors = {}
for i, box in enumerate(bplot.artists):
    box_color = box.get_facecolor()
    box_colors[df.columns[i]] = box_color

# 在同一x轴位置添加散点
x_position = 0  # 所有点将在x=0的位置上
for model, color in box_colors.items():
    values = df[model]
    ax.scatter([x_position] * len(values), values, color=color, s=20, alpha=0.6)

# 设置标题和标签
ax.set_title('Data Distribution by Model', fontsize=16)
ax.set_xlabel('Model', fontsize=14)
ax.set_ylabel('Value', fontsize=14)

# 移除x轴的刻度标签，因为我们只有一个点集
ax.set_xticks([])

# 显示图形
plt.tight_layout()
plt.show()