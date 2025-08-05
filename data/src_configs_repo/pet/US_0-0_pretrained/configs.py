import os
from src.utils.config_injector.ForcingRunoffDatasetConfigUtils import init_dss_cfg
import global_variables


class ForcingRunoffDatasetConfig:
    dss_cfg = init_dss_cfg(os.path.dirname(__file__), ["CAMELS-US"])


class PretrainConfig:
    gpus = [0]  # TODO: multi-gpu list, the first gpu is the main gpu
    n_epochs = 100  # TODO
    start_epoch = 100  # start_epoch should be zero when checkpoint_path is ""
    checkpoint_path = "/data1/du/scalable-prediction-platform/runs/Transformer[NAR]_lossMSE_n100_bs2048_lr0.0001_none_seed2333_start0_cp@[64-4-4-4-256-0.1]_22[15]|22[1]_past15_pred7_mask0_upfTrue_uffTrue_uptTrue@23c61de8cb14c808be663c248600b014/(min_mse)_72_0.16584219980659448.pkl"  # start_epoch should be zero when checkpoint_path is "


class PretrainTestConfig:
    saved_root = global_variables.run_root + "/Transformer[NAR]_lossMSE_n100_bs2048_lr0.0001_none_seed2333_start100_cp(min_mse)_72_0.16584219980659448.pkl@[64-4-4-4-256-0.1]_22[15]|22[1]_past15_pred7_mask0_upfTrue_uffTrue_uptTrue@1d3747c256408a1a80dfbde2cec9b476"
    result_saving_root = saved_root + "/test_results"
    gpus = [0]  # TODO: multi-gpu list, the first gpu is the main gpu
