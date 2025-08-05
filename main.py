import argparse
import shutil
from pathlib import Path
from typing import Dict
import pandas as pd
import os
import torch

from utils.tools import SeedMethods
from exp.exp_main import Exp_Main
from utils.utils import str_to_bool

import warnings

# warnings.filterwarnings("ignore")  # NOTE:忽略warning

# NOTE: 默认的seed
fix_seed = 1234

# fixed settings for all experiments
GLOBAL_SETTINGS = {
    # tank模型使用已废弃
    # 'tank_nums': 4,
    # 'factors': {'bottom_factor': 0, 'side_factor': 0, 'thres': 0, 'h_initial': 0},
    # tank模型使用已废弃

    # 注释掉的已经实现可输入参数
    # 'batch_size': 512,
    # 'dropout': 0.1,
    # 'epochs': 50,
    # 'n_heads': 4,
    # 'd_model': 64,
    'd_ff': 256,
    'hidden_size': 256,
    # 'learning_rate': 1e-3,
    # 'src_size': 5,  # 原5，united时候是7
    'tgt_size': 2,
    # 'static_size': 27,  # NOTE:[us:27,gb:22]
    # 'past_len': 1,
    # 'pred_len': 1,
}

# 使用kraxxx风格时候需要在这里设置时间
# 其他时候在下面加上'forcing_type':{}即可
DATE_CHOICE = {
    # NOTE: maurer
    'maurer': {
        'train_start': pd.to_datetime("2001-10-01", format="%Y-%m-%d"),
        'train_end': pd.to_datetime("2008-09-30", format="%Y-%m-%d"),
        'val_start': pd.to_datetime("1999-10-01", format="%Y-%m-%d"),
        'val_end': pd.to_datetime("2001-09-30", format="%Y-%m-%d"),
        'test_start': pd.to_datetime("1989-10-01", format="%Y-%m-%d"),
        'test_end': pd.to_datetime("1999-09-30", format="%Y-%m-%d")
    }
}


def get_args() -> Dict:
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    # NOTE:basic config
    parser.add_argument('--stage', type=str, required=True, choices=["train", "evaluate", "test", "finetune",
                                                                     "finetune_test", "train_single"])
    parser.add_argument('--model_id', type=str, required=True, help='model id')
    parser.add_argument('--model', type=str, required=True, help='model name, options: [Tank, SAC,Trm,LSTM]')
    parser.add_argument('--sub_model', type=str, required=True, help='sub model of SAC model:[Ep]')
    parser.add_argument('--seed', type=int, required=True, default=fix_seed, help="Random seed")
    parser.add_argument('--run_dir', type=str,
                        help="For evaluation and test stage. Path to run directory.之前train保存的位置")
    parser.add_argument('--finetune_dir', type=str, default="",
                        help="finetune saving dir")
    parser.add_argument('--sh_file', type=str, help="run .sh file")
    parser.add_argument('--local_run_dir', type=str, default='', help="run dir(local),default=''")
    parser.add_argument('--global_run_dir', type=str, default='', help="run dir(global),default=''")

    # NOTE:
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="learning_rate")
    parser.add_argument('--dropout', type=float, default=0.1, help="dropout")
    parser.add_argument('--epochs', type=int, default=50, help="epochs")
    parser.add_argument('--past_len', type=int, default=1, help="past_len")
    parser.add_argument('--pred_len', type=int, default=1, help="pred_len")
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=4, help="n_heads")
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--src_size', type=int, default=5, help="also called dynamic_size")
    parser.add_argument('--static_size', type=int, default=27, help="static_size,[us:27,gb:22]")
    parser.add_argument('--num_layers', type=int, default=2, help="")
    # parser.add_argument('--d_ff', type=int, default=256, help="")
    # parser.add_argument('--hidden_size', type=int, default=256, help="feedforward hidden size")
    # parser.add_argument('--tgt_size', type=int, default=2, help="")
    parser.add_argument('--output_size', type=int, default=2, help="n_heads")

    # NOTE:data loader
    parser.add_argument('--camels_root', type=str, default='/home/zhuwu/Dataset/CAMELS', help='CAMELS root dir')
    parser.add_argument('--forcing_type', type=str, default='daymet', help='Types of Meteorological Forcing Properties')
    parser.add_argument('--basins_list_path', type=str, default='/data2/zmz1/Tank/data/448basins_list.txt',
                        help='basins_list path')
    parser.add_argument('--freq', type=str, default='d',
                        help='freq for time features encoding, options:'
                             '[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly],'
                             ' you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--drop_last', type=str_to_bool, nargs='?', const=True, required=True,
                        help='whether to drop the part that is less than a batch size')
    parser.add_argument('--use_static', type=str_to_bool, nargs='?', default=True,
                        help='whether to enable static in x')
    parser.add_argument('--use_runoff', type=str_to_bool, nargs='?', default=False,
                        help='whether to enable runoff in x')
    parser.add_argument('--dss_config_path', type=str, default='/data2/zw/sac/data/src_configs_repo/noUS',
                        help='path for camels dataset YAML configs')
    parser.add_argument('--dss_config_ds', nargs='+', default='"CAMELS-AUS" "CAMELS-BR" "CAMELS-GB"',
                        help='An array includes the specific region of the Camels dataset used, in the form of'
                             '["CAMELS-US"]. They must be provided under the path directory.')

    # NOTE:optimization
    parser.add_argument('--num_workers', type=int, default=12, help='data loader num workers')
    parser.add_argument('--loss', type=str, default='nse', help='loss function')
    parser.add_argument('--loss_all', type=str_to_bool, nargs='?', const=True, required=True,
                        help='whether to enable calculate loss for all days')
    parser.add_argument('--use_var', type=str_to_bool, nargs='?', const=True, required=True,
                        help='是否仅用parameter来学习参数')
    parser.add_argument('--warm_up', type=str_to_bool, nargs='?', const=True, required=True,
                        help='whether to enable warm_up')
    parser.add_argument('--data_path', type=str, default='', help='use another datapath')

    # NOTE:GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true',
                        help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # NOTE:finetune and test
    parser.add_argument('--finetune_name', type=str, default=None, help='')
    parser.add_argument('--finetune_mode', type=str, default=None, help='')
    parser.add_argument('--test_name', type=str, default=None, help='')
    parser.add_argument('--test_mode', type=str, default=None, help='')
    parser.add_argument('--train_mode', type=str, default=None, help='')

    cfg = vars(parser.parse_args())

    # Validation checks
    if (cfg["stage"] == "train") and (cfg["seed"] is None):
        # generate random seed for this run
        # cfg["seed"] = int(np.random.uniform(low=0, high=1e6))
        raise ValueError("need seed")

    if (cfg["stage"] in ["evaluate", "test"]) and (cfg["run_dir"] is None):
        raise ValueError("In evaluation stage a run directory (--run_dir) has to be specified")

    # combine global settings with user config
    cfg.update(GLOBAL_SETTINGS)
    cfg.update(DATE_CHOICE['maurer'])
    # if cfg['forcing_type'] in ['united', 'separated', 'yr_u', 'yr_s']:  # ADD
    #     pass
    # else:
    #     cfg.update(DATE_CHOICE[cfg['forcing_type']])

    if cfg["stage"] == "train":
        # print config to terminal
        for key, val in cfg.items():
            print(f"{key}: {val}")

    # convert path to PosixPath object
    cfg["camels_root"] = Path(cfg["camels_root"])
    if cfg["run_dir"] is not None:
        cfg["run_dir"] = Path(cfg["run_dir"])
    return cfg


if __name__ == '__main__':
    # initializing
    print("pid:", os.getpid())

    cfg = get_args()
    print("cfg:\n", cfg)

    SeedMethods.seed_torch(seed=cfg['seed'])

    # initializing
    exp = Exp_Main(cfg)
    if cfg['stage'] == 'train':
        if cfg['train_mode'] == 'single':
            exp.train_single(cfg)
        else:
            exp.train(cfg)
    elif cfg['stage'] == 'test':
        if cfg['test_mode'] == 'single':
            exp.test(cfg)
        else:
            raise NotImplementedError("all,暂时不用")
    elif cfg['stage'] == 'finetune':
        if cfg['finetune_mode'] == 'all':
            exp.train(cfg)
        else:
            exp.finetune_single(cfg)
            # exp.finetune(cfg)
    # 这两个函数比较久远，可能已经无用
    # elif cfg['stage'] == 'finetune_test':
    #     exp.finetune_test(cfg)
    # elif cfg['stage'] == 'train_single':
    #     exp.train_single_pre(cfg)
