import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
from pathlib import Path

def parse_value(value: str):
    return float(value[1:-1])

def load_basin_data(base_dir, basin_id, start_date, end_date):

    data_file = os.path.join(base_dir, basin_id, f'{basin_id}_data.txt')

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"找不到文件: {data_file}")

    # 读取数据
    df = pd.read_csv(data_file, header=None, skiprows=1)

    # 列名映射
    columns = ['start_date', 'obs_tci', 'pet', 'pred_tci', 'pred_ep']
    df.columns = columns[:df.shape[1]]

    # 解析值
    for col in df.columns:
        if col != 'start_date':
            df[col] = df[col].apply(parse_value)

    # 设置时间索引
    df['Date'] = pd.to_datetime(df['start_date'])
    df.set_index('Date', inplace=True)

    # 截取时间段
    df = df.loc[start_date:end_date]

    df = df[['start_date', 'obs_tci', 'pred_tci']]
    df = df.rename(columns={'pred_tci': 'TRM-Tank'})

    return df

def process_models(basin_id, model_names, base_path, df):
    basin_num = basin_id.split('_')[-1].zfill(8)  # 获取数字部分并补零到8位

    for model in model_names:
        folder_path = Path(base_path) / model / "pretrain_test_single" / basin_num
        file_path = folder_path / "obs_pred" / "pred.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        if model != "Pyramidal Transformer":
            df_model = pd.read_csv(file_path, usecols=['start_date', 'pred0'])
            df_model = df_model.rename(columns={'pred0': model})
        else:
            df_model = pd.read_csv(file_path, usecols=['start_date', 'pred_0'])
            df_model = df_model.rename(columns={'pred_0': model})

        df_model['start_date'] = pd.to_datetime(df_model['start_date'])
        df_model.set_index('start_date', inplace=True)

        df = df.join(df_model, how='left')

    return df

def plot_basin_data(df, year, basin_id, output_folder):
    if df.empty:
        print(f"{year} 年无有效数据，跳过绘图。")
        return

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
        'figure.figsize': (12, 6),
        'savefig.dpi': 300,
        'savefig.format': 'png',
        'text.usetex': False,
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0,
        'xtick.color': 'black',
        'ytick.color': 'black',
        'axes.facecolor': '#f9f9f9',
        'savefig.facecolor': 'white'
    })

    fig, ax = plt.subplots()

    # Tableau 配色基础上，特别增强 TRM-Tank 和 Obs 的颜色对比
    colors = ['#4e79a7', '#76b7b2', '#f28e2b', '#e15759', '#9c755f']  # 原配色
    # 特别强化 TRM-Tank 和 obs_tci 的颜色
    colors[3] = '#d62728'  # TRM-Tank -> 更鲜艳的红色
    colors[4] = '#1f77b4'  # obs_tci -> 更深的蓝色（Tableau 蓝）

    # 绘制每条曲线
    ax.plot(df.index, df['Transformer'], label='RR-Former', color=colors[0], alpha=0.9, linewidth=2.0)
    ax.plot(df.index, df['Pyramidal Transformer'], label='Pyramidal Transformer', color=colors[1], alpha=0.9,
            linewidth=2.0)
    ax.plot(df.index, df['LSTM-S2S'], label='LSTM-S2S', color=colors[2], alpha=0.9, linewidth=2.0)

    # 特殊处理：TRM-Tank 和 obs_tci 线条更粗 + 颜色更显眼
    ax.plot(df.index, df['TRM-Tank'], label='TRM-Tank', color=colors[3], alpha=0.95, linewidth=2.0, linestyle='-')
    ax.plot(df.index, df['obs_tci'], label='Runoff Observation', color=colors[4], alpha=0.95, linewidth=2.0,
            linestyle='-', marker='', markevery=10)

    # 时间格式化
    ax.xaxis.set_major_locator(MonthLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))

    # y轴设置
    y_min, y_max = ax.get_ylim()
    y_step = max(1, round((y_max - y_min) / 10))
    y_ticks = np.arange(round(y_min / y_step) * y_step,
                        round(y_max / y_step) * y_step + y_step, y_step)
    ax.set_yticks(y_ticks)

    # 标签和标题
    ax.set_xlabel("Date", fontsize=12, weight='bold')
    ax.set_ylabel("Runoff (mm)", fontsize=12, weight='bold')
    ax.set_title(f"Basin ID: {basin_id}", fontsize=14)

    # 图例
    ax.legend(loc='upper right', frameon=True, edgecolor='black', fancybox=False)

    # 网格与边框美化
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

    fig.autofmt_xdate()  # 自动旋转日期
    fig.tight_layout()

    # 保存图表
    eps_path = os.path.join(output_folder, f"{year}.eps")
    png_path = os.path.join(output_folder, f"{year}.png")

    plt.savefig(eps_path, format='eps', dpi=1200)
    plt.savefig(png_path, format='png', dpi=300)
    plt.close()

    print(f"{year} 年图表已保存至: {output_folder}")

def main():
    # 参数设置
    # base_dir = "/data2/zmz1/Tank/runs_paper/448SeriesTank_Series4_runoff_Transformer_[448basins_list.txt,['CAMELS-US']]_[SeriesTank,varFalse,batch_size2048]_[epochs200]_[15,1][2025031212]/test/448basins/basins"
    base_dir = "/data2/zmz1/Tank/runs_paper/448SeriesTank_Series4_runoff_Transformer_[448basins_list.txt,['CAMELS-US']]_[SeriesTank,varFalse,batch_size2048]_[epochs200]_[90,1][2025031616]/test/448basins/basins"
    base_model_path = "/data2/zmz1/data_obs"
    model_names = ["Pyramidal Transformer", "LSTM-S2S", "Transformer"]
    picture_dir = "picture90"

    years = range(1990, 1999)

    os.makedirs(picture_dir, exist_ok=True)

    basin_ids = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

    for basin_id in basin_ids:
        print(f"正在处理站点: {basin_id}")

        output_folder = os.path.join(picture_dir, basin_id)
        os.makedirs(output_folder, exist_ok=True)

        for year in years:
            start_date = pd.to_datetime(f"{year}-05-01")
            end_date = pd.to_datetime(f"{year}-08-01")

            try:
                df_base = load_basin_data(base_dir, basin_id, start_date, end_date)

                if df_base.empty:
                    print(f"站点 {basin_id} {year} 年无观测数据，跳过...")
                    continue

                df_final = process_models(basin_id, model_names, base_model_path, df_base.copy())
                plot_basin_data(df_final, year, basin_id, output_folder)

            except Exception as e:
                print(f"站点 {basin_id} 处理 {year} 年出错: {e}")

if __name__ == "__main__":
    main()