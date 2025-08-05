import datetime
import json
import logging
import os
import shutil
from pathlib import Path

import pandas as pd
import torch
import numpy as np


class Exp_Basic(object):
    def __init__(self, cfg):
        self.cfg = cfg

        if cfg['stage'] == 'train' or cfg['stage'] == 'train_single':
            self.run_cfg = None
            self.device = self._acquire_device()
            if cfg['train_mode'] !='single':
                self.model = self._build_model(self.cfg).to(self.device)
        elif cfg['stage'] == 'test':
            self.run_cfg = self.read_cfg()
            self.device = self._acquire_device()
            self.model = self._build_model(self.run_cfg).to(self.device)
        elif cfg['stage'] == 'finetune':
            self.run_cfg = self.read_cfg()
            self.device = self._acquire_device()
            if cfg['finetune_mode'] == 'all':
                self.model = self._build_model(self.run_cfg).to(self.device)
        elif cfg['stage'] == 'finetune_test':
            self.run_cfg = self.read_cfg()
            self.finetune_cfg = self.read_finetune_cfg()
            self.device = self._acquire_device()
            # print(self.run_cfg)
            # print(self.finetune_cfg)
            # self.model = self._build_model(self.run_cfg).to(self.device)

    def _build_model(self, cfg):
        raise NotImplementedError("未设置model")

    def _acquire_device(self):
        if self.cfg['use_gpu']:
            if self.cfg['use_multi_gpu']:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cfg['devices'])
            device = torch.device('cuda:{}'.format(self.cfg['gpu']))
            self.cfg['device'] = device  # ADD
            print('Use GPU: cuda:{}'.format(self.cfg['gpu']))
        else:
            device = torch.device('cpu')
            self.cfg['device'] = device  # ADD
            print('Use CPU')
        # device = torch.device('cpu')
        # self.cfg['device'] = device
        # print('Use CPU')
        return device

    def setup_train(self):
        now = datetime.datetime.now()
        year = f"{now.year}".zfill(4)
        day = f"{now.day}".zfill(2)
        month = f"{now.month}".zfill(2)
        hour = f"{now.hour}".zfill(2)
        minute = f"{now.minute}".zfill(2)
        stage = self.cfg['stage']

        if 'united' in self.cfg["forcing_type"] or 'separated' in self.cfg["forcing_type"]:
            # 文件名过长， 需要剪短
            # run_name = f'run_{self.cfg["model_id"]}_' \
            #            f'[{self.cfg["dss_config_path"].split("/")[-1]},{self.cfg["dss_config_ds"]}]_' \
            #            f'[{self.cfg["model"]},var{self.cfg["use_var"]},batch_size{self.cfg["batch_size"]}]_' \
            #            f'[epochs{self.cfg["epochs"]}]_' \
            #            f'[{self.cfg["past_len"]},{self.cfg["pred_len"]}]_' \
            #            f'[{self.cfg["learning_rate"]},warmUp{self.cfg["warm_up"]}]_' \
            #            f'[{self.cfg["loss"]}_all{self.cfg["loss_all"]}]_' \
            #            f'[dp{self.cfg["dropout"]}]_' \
            #            f'seed{self.cfg["seed"]}_'
            run_name = f'{self.cfg["model_id"]}_' \
                       f'[{self.cfg["dss_config_path"].split("/")[-1]},{self.cfg["dss_config_ds"]}]_' \
                       f'[{self.cfg["model"]},var{self.cfg["use_var"]},batch_size{self.cfg["batch_size"]}]_' \
                       f'[epochs{self.cfg["epochs"]}]_' \
                       f'[{self.cfg["past_len"]},{self.cfg["pred_len"]}]' \
                       f'[{year}{month}{day}{hour}]'




        else:
            run_name = f'run_{self.cfg["model_id"]}_' \
                       f'[{self.cfg["basins_list_path"].split("/")[-1].split("_")[0]},{self.cfg["forcing_type"]}]_' \
                       f'[{self.cfg["model"]},var{self.cfg["use_var"]},batch_size{self.cfg["batch_size"]}]_' \
                       f'[epochs{self.cfg["epochs"]}]_' \
                       f'[{self.cfg["past_len"]},{self.cfg["pred_len"]}]_' \
                       f'[{self.cfg["learning_rate"]},warmUp{self.cfg["warm_up"]}]_' \
                       f'[{self.cfg["loss"]}_all{self.cfg["loss_all"]}]_' \
                       f'[dp{self.cfg["dropout"]}]_' \
                       f'seed{self.cfg["seed"]}_'

        self.cfg['run_dir'] = Path(__file__).absolute().parent.parent / "runs_671_new" / run_name  # 需要手动调整
        if self.cfg['local_run_dir'] != '':
            print('local', self.cfg['local_run_dir'])
            self.cfg['run_dir'] = Path(__file__).absolute().parent.parent / self.cfg[
                'local_run_dir'] / run_name  # 需要手动调
        if not self.cfg["run_dir"].is_dir():
            self.cfg['config_dir'] = self.cfg['run_dir'] / 'config'
            self.cfg['data_dir'] = self.cfg['run_dir'] / 'data'
            self.cfg['train_dir'] = self.cfg['run_dir'] / 'train'
            self.cfg['test_dir'] = self.cfg['run_dir'] / 'test'
            # self.cfg['finetune_dir'] = self.cfg['run_dir'] / 'finetune_dir' #todo finetune_dir错误
            self.cfg['finetune_dir'] = self.cfg['run_dir'] / 'finetune'
            # self.cfg['cached_data_root'] = self.cfg['run_dir'] / 'cache'

            self.cfg["run_dir"].mkdir(parents=True)
            self.cfg["config_dir"].mkdir(parents=True)
            self.cfg["data_dir"].mkdir(parents=True)
            self.cfg["train_dir"].mkdir(parents=True)
            self.cfg["test_dir"].mkdir(parents=True)
            self.cfg["finetune_dir"].mkdir(parents=True)
        else:
            self.cfg['config_dir'] = self.cfg['run_dir'] / 'config'
            self.cfg['data_dir'] = self.cfg['run_dir'] / 'data'
            self.cfg['train_dir'] = self.cfg['run_dir'] / 'train'
            self.cfg['test_dir'] = self.cfg['run_dir'] / 'test'
            self.cfg['finetune_dir'] = self.cfg['run_dir'] / 'finetune'


        shutil.copy2(self.cfg["sh_file"], self.cfg['train_dir'])
        # 将水文站点的列表复制到目标目录中去，以便站点被覆盖后恢复
        # for item in os.listdir(self.cfg["dss_config_path"]):
        #     yml_root = os.path.join(self.cfg["dss_config_path"], item)
        shutil.copy2(self.cfg["dss_config_path"], self.cfg['data_dir'])

        # dump a copy of cfg to run directory
        with (self.cfg["config_dir"] / 'cfg.json').open('w') as fp:
            temp_cfg = {}
            for key, val in self.cfg.items():
                if isinstance(val, Path):
                    temp_cfg[key] = str(val)
                elif isinstance(val, pd.Timestamp):
                    temp_cfg[key] = val.strftime("%Y-%m-%d")
                elif isinstance(val, torch.device):
                    temp_cfg[key] = str(val)
                else:
                    temp_cfg[key] = val
            # json.dump(temp_cfg, fp, sort_keys=True, indent=4)
            json.dump(temp_cfg, fp, sort_keys=False, indent=4)
        print(f"save cfg in {self.cfg['config_dir']} ok")

    def setup_finetune(self):
        if self.cfg['finetune_mode'] != 'all' and self.cfg['finetune_mode'] != 'yr_single' and self.cfg[
            'finetune_mode']!='single':
            basins_list = pd.read_csv(self.cfg["basins_list_path"], header=None, dtype=str)[0].values.tolist()

        # self.cfg['run_dir'] = self.run_cfg['run_dir']
        if self.cfg["run_dir"].is_dir():
            self.cfg['finetune_dir'] = Path(self.cfg['run_dir'] / 'finetune')

            if self.cfg['finetune_name'] is not None:
                self.cfg['finetune_dir'] = Path(self.cfg['finetune_dir']) / self.cfg['finetune_name']
            else:
                raise ValueError("need test name(special dir in test_dir)")

            if self.cfg["finetune_dir"].is_dir():
                # logging.warning("parent finetune_dir exists")
                raise RuntimeError("finetune_dir exists!!! may overwrite!!!")
            else:
                self.cfg["finetune_dir"].mkdir(parents=True)

            self.cfg['data_dir'] = Path(self.cfg['finetune_dir']) / 'data'
            self.cfg["data_dir"].mkdir(parents=True)
        else:
            raise RuntimeError(f"no {self.cfg['run_dir']}")

        # save shell file
        # shutil.copytree("/data2/zw/sac_paper/models", self.cfg['finetune_dir'] / "models_backup")
        shutil.copy2(self.cfg["sh_file"], self.cfg['finetune_dir'])

        # dump a copy of cfg to run directory
        with (self.cfg["config_dir"] / 'finetune_cfg.json').open('w') as fp:
            temp_cfg = {}
            for key, val in self.cfg.items():
                if isinstance(val, Path):
                    temp_cfg[key] = str(val)
                elif isinstance(val, pd.Timestamp):
                    temp_cfg[key] = val.strftime("%Y-%m-%d")
                elif isinstance(val, torch.device):
                    temp_cfg[key] = str(val)
                else:
                    temp_cfg[key] = val
            # json.dump(temp_cfg, fp, sort_keys=True, indent=4)
            json.dump(temp_cfg, fp, sort_keys=False, indent=4)
        print(f"save finetune_cfg in {self.cfg['config_dir']} ok")

    def read_cfg(self):
        self.cfg['config_dir'] = self.cfg['run_dir'] / 'config'
        # self.cfg['data_dir'] = self.cfg['run_dir'] / 'data'
        with open(self.cfg['config_dir'] / "cfg.json", 'r') as fp:
            run_cfg = json.load(fp)
            for key, val in run_cfg.items():
                if 'dir' in key:
                    # print(key,val) #TEST
                    run_cfg[key] = Path(val)
                elif 'start' in key or 'end' in key:
                    run_cfg[key] = pd.to_datetime(val, format="%Y-%m-%d")
                else:
                    run_cfg[key] = val
        print(f"read run_cfg from {self.cfg['config_dir']} ok")

        if self.cfg['stage'] == 'test':
            if self.cfg['test_name'] is not None:
                print(self.cfg['finetune_name'])
                if self.cfg['finetune_name'] is not None:
                    run_cfg['test_dir'] = Path(run_cfg['test_dir']) / (
                            self.cfg['test_name'] + '+' + self.cfg['finetune_name'])
                    run_cfg['data_dir'] = run_cfg['finetune_dir'] / self.cfg['finetune_name'] / 'data'
                else:
                    run_cfg['test_dir'] = Path(run_cfg['test_dir']) / self.cfg['test_name']

                if run_cfg["test_dir"].is_dir():
                    pass
                else:
                    run_cfg["test_dir"].mkdir(parents=True)
                # save shell file

                #这个地方老是报错现在给注释掉
                # shutil.copy2(self.cfg["sh_file"], run_cfg['test_dir'])
            else:
                raise ValueError("need test name(special dir in test_dir)")

        # if self.cfg['stage'] == 'finetune':
        #     if self.cfg['finetune_name'] is not None:
        #         self.cfg['finetune_dir'] = Path(self.cfg['finetune_dir']) / self.cfg['finetune_name']
        #         # save shell file
        #         shutil.copy2(self.cfg["sh_file"], self.cfg['finetune_dir'])
        #     else:
        #         raise ValueError("need test name(special dir in test_dir)")

        return run_cfg

    def read_finetune_cfg(self):
        self.cfg['config_dir'] = self.cfg['run_dir'] / 'config'
        self.cfg['data_dir'] = self.cfg['run_dir'] / 'data'
        with open(self.cfg['config_dir'] / 'finetune_cfg.json', 'r') as fp:
            finetune_cfg = json.load(fp)
            for key, val in finetune_cfg.items():
                if 'dir' in key:
                    finetune_cfg[key] = Path(val)
                elif 'start' in key or 'end' in key:
                    finetune_cfg[key] = pd.to_datetime(val, format="%Y-%m-%d")
                else:
                    finetune_cfg[key] = val
        print(f"read finetune_cfg from {self.cfg['config_dir']} ok")

        # save shell file
        # shutil.copy2(self.cfg["sh_file"], finetune_cfg['finetune_dir'])
        return finetune_cfg

    def _get_data(self, **kwargs):
        pass

    def vali(self, **kwargs):
        pass

    def train_epoch(self, **kwargs):
        pass

    def train(self, **kwargs):
        pass

    def test(self, **kwargs):
        pass
