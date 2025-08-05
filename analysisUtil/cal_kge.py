import os
import pandas as pd
from scipy.stats import pearsonr


def calculate_kge(obs, pred):
    """
    计算KGE值。
    :param obs: 观测值列表或numpy数组。
    :param pred: 预测值列表或numpy数组。
    :return: KGE值。
    """
    mean_obs = obs.mean()
    mean_pred = pred.mean()

    # Pearson相关系数
    r, _ = pearsonr(obs, pred)

    # 平均值比率
    beta = mean_pred / mean_obs

    # 变异系数比率
    cv_obs = obs.std() / mean_obs
    cv_pred = pred.std() / mean_pred
    alpha = cv_pred / cv_obs

    kge = 1 - ((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2) ** 0.5
    return kge


def read_and_calculate_kge(obs_file, pred_file, model):
    """
    读取观测值和预测值，并计算KGE值。
    :param obs_file: 观测值文件路径。
    :param pred_file: 预测值文件路径。
    :return: KGE值。
    """
    if(model != "Pyramidal Transformer"):
        obs_df = pd.read_csv(obs_file, usecols=['start_date', 'obs0'])
        pred_df = pd.read_csv(pred_file, usecols=['start_date', 'pred0'])
    else:
        obs_df = pd.read_csv(obs_file, usecols=['start_date', 'obs_0'])
        pred_df = pd.read_csv(pred_file, usecols=['start_date', 'pred_0'])

    # 按照日期合并两个DataFrame
    merged_df = pd.merge(obs_df, pred_df, on='start_date')

    if merged_df.empty:
        print(f"警告：{obs_file} 和 {pred_file} 之间没有共同的日期。")
        return None

    kge_value = calculate_kge(merged_df['obs0'], merged_df['pred0'])
    return kge_value


def process_models(base_path):
    """
    处理多个模型的数据并计算每个站点的KGE值。
    :param base_path: 数据基础路径。
    """
    model_names = ["LSTMS2S", "Pyramidal Transformer", "Transformer"]
    output_data = {'basin_id': [], 'model': [], 'kge': []}

    for model in model_names:
        model_path = os.path.join(base_path, model, "pretrain_test_single")

        # 遍历每个水文站点
        for basin_id in os.listdir(model_path):
            basin_path = os.path.join(model_path, basin_id, "obs_pred")
            obs_file = os.path.join(basin_path, "obs.csv")
            pred_file = os.path.join(basin_path, "pred.csv")

            if not os.path.exists(obs_file) or not os.path.exists(pred_file):
                print(f"警告：{basin_id} 的观测或预测文件不存在。")
                continue

            kge_value = read_and_calculate_kge(obs_file, pred_file, model)
            if kge_value is not None:
                output_data['basin_id'].append(basin_id)
                output_data['model'].append(model)
                output_data['kge'].append(kge_value)

    # 将结果保存到CSV文件
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(os.path.join(base_path, "kge_values.csv"), index=False)


if __name__ == "__main__":
    data_path = "/data2/zmz1/data_obs"
    process_models(data_path)