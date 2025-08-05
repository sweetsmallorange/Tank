# All metrics needs obs and sim shape: (batch_size, pred_len, tgt_size)

import numpy as np
import torch


def calc_nse(obs: np.array, sim: np.array) -> np.array:
    denominator = np.sum((obs - np.mean(obs, axis=0)) ** 2, axis=0)
    numerator = np.sum((sim - obs) ** 2, axis=0)
    nse = 1 - numerator / denominator

    nse_mean = np.mean(nse)
    # Return mean NSE, and NSE of all locations, respectively
    # return nse_mean, nse[:,0]
    return nse_mean, nse


def calc_new(obs: np.array, sim: np.array):
    sigma2 = np.mean((obs - np.mean(obs)) ** 2)
    mse = np.mean((sim - obs) ** 2)

    first_coff = sigma2 / (sigma2 + 1)
    first = first_coff * (mse / (sigma2 + mse))
    second_coff = 1 / (sigma2 + 1)
    second = second_coff * (mse / (1 + mse))

    test2 = first + second
    return test2


def calc_kge(obs: np.array, sim: np.array):
    mean_obs = np.mean(obs, axis=0)
    mean_sim = np.mean(sim, axis=0)

    std_obs = np.std(obs, axis=0)
    std_sim = np.std(sim, axis=0)

    beta = mean_sim / mean_obs
    alpha = std_sim / std_obs
    numerator = np.mean(((obs - mean_obs) * (sim - mean_sim)), axis=0)
    denominator = std_obs * std_sim
    gamma = numerator / denominator
    kge = 1 - np.sqrt((beta - 1) ** 2 + (alpha - 1) ** 2 + (gamma - 1) ** 2)

    kge_mean = np.mean(kge)
    # Return mean KEG, and KGE of all locations, respectively
    return kge_mean, kge[:, 0]


def calc_tpe(obs: np.array, sim: np.array, alpha):
    sort_index = np.argsort(obs, axis=0)
    obs_sort = np.take_along_axis(obs, sort_index, axis=0)
    sim_sort = np.take_along_axis(sim, sort_index, axis=0)
    top = int(obs.shape[0] * alpha)
    obs_t = obs_sort[-top:, :]
    sim_t = sim_sort[-top:, :]
    numerator = np.sum(np.abs(sim_t - obs_t), axis=0)
    denominator = np.sum(obs_t, axis=0)
    tpe = numerator / denominator

    tpe_mean = np.mean(tpe)
    # Return mean TPE, and TPE of all locations, respectively
    return tpe_mean, tpe[:, 0]


def calc_bias(obs: np.array, sim: np.array):
    numerator = np.sum(sim - obs, axis=0)
    denominator = np.sum(obs, axis=0)
    bias = numerator / denominator

    bias_mean = np.mean(bias)
    # Return mean bias, and bias of all locations, respectively
    return bias_mean, bias[:, 0]


def calc_mse(obs: np.array, sim: np.array):
    mse = np.mean((obs - sim) ** 2, axis=0)

    mse_mean = np.mean(mse)
    # Return mean MSE, and MSE of all locations, respectively
    # return mse_mean, mse[:, 0]
    return mse_mean, mse


def calc_rmse(obs: np.array, sim: np.array):
    mse = np.mean((obs - sim) ** 2, axis=0)
    rmse = np.sqrt(mse)

    rmse_mean = np.mean(rmse)
    # Return mean RMSE, and RMSE of all locations, respectively
    return rmse_mean, rmse[:, 0]


def calc_nrmse(obs: np.array, sim: np.array):
    mse = np.mean((obs - sim) ** 2, axis=0)
    rmse = np.sqrt(mse)
    obs_mean = np.mean(obs, axis=0)
    nrmse = rmse / obs_mean

    nrmse_mean = np.mean(nrmse)
    # Return mean NRMSE, and NRMSE of all locations, respectively
    return nrmse_mean, nrmse[:, 0]


def calc_log_nse(obs: np.array, sim: np.array, eps=1e-6, remove_outliers = True):
    """
    计算对数纳什效率系数 (log-NSE)

    参数:
    obs -- 观测值数组 (形状: [时间步长, 空间位置])
    sim -- 模拟值数组 (形状需与obs相同)

    返回:
    log_nse_mean -- 所有空间位置log-NSE的平均值
    log_nse      -- 每个空间位置的log-NSE值 (一维数组)
    """

    # 复制数组以避免修改原始数据
    obs = obs.copy()
    sim = sim.copy()

    # 防止出现负值或零，加上一个极小值
    obs += eps
    sim += eps

    # 移除异常值（NaN、inf）
    if remove_outliers:
        valid_mask = np.isfinite(obs) & np.isfinite(sim) & (obs > 0) & (sim > 0)
        obs[~valid_mask] = np.nan
        sim[~valid_mask] = np.nan


    # 取自然对数
    log_obs = np.log(obs)
    log_sim = np.log(sim)

    # 计算均值（忽略 NaN）
    log_obs_mean = np.nanmean(log_obs, axis=0, keepdims=True)

    # 计算分子和分母
    numerator = np.nansum((log_sim - log_obs) ** 2, axis=0)
    denominator = np.nansum((log_obs - log_obs_mean) ** 2, axis=0)

    # 避免除以零
    with np.errstate(divide='ignore', invalid='ignore'):
        log_nse = 1 - (numerator / denominator)

    # 替换无穷值为 NaN
    log_nse = np.where(np.isinf(log_nse), np.nan, log_nse)

    # 返回平均 log-NSE 和每个站点的结果
    log_nse_mean = np.nanmean(log_nse)

    return log_nse_mean, log_nse


def calc_nse_torch(obs, sim):
    '''
    先实际值，再预测值
    '''
    with torch.no_grad():
        denominator = torch.sum((obs - torch.mean(obs, dim=0)) ** 2, dim=0)
        numerator = torch.sum((sim - obs) ** 2, dim=0)
        nse = torch.tensor(1).to(sim.device) - numerator / denominator

        nse_mean = torch.mean(nse)
        # Return mean NSE, and NSE of all locations, respectively
        return nse_mean, nse[:, 0]
