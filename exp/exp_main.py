import importlib
import os
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

from data.dataset import data_provider, data_provider_finetune
from exp.exp_basic import Exp_Basic
from utils.lr_strategies import SchedulerFactory
from utils.metrics import calc_nse_torch, calc_nse, calc_mse, calc_tpe, calc_kge, calc_rmse, calc_log_nse
from utils.nseloss import NSELoss
from utils.tools import count_parameters
from utils.utils import BestModelLog
from data.united.ForcingRunoffDatasetConfigUtils import init_dss_cfg

warnings.filterwarnings('ignore')

# TODO
from data.dataset import MODELLIST


class Exp_Main(Exp_Basic):
    def __init__(self, cfg):
        super(Exp_Main, self).__init__(cfg)

    def _build_model(self, cfg):
        # Define model type
        models = importlib.import_module("models")
        Model = getattr(models, cfg["model"])
        model = Model(cfg)  # to_device在exp_basic.py里
        return model

    def _get_data(self, cfg, stage, single=False):
        data_set, data_loader = data_provider(cfg, stage, single)
        return data_set, data_loader

    def _get_data_finetune(self, cfg, stage, basin):
        data_set, data_loader = data_provider_finetune(cfg, stage, basin)
        return data_set, data_loader

    def _select_optimizer(self, scheduler_paras, model=None):
        if model is not None:
            model_optim = optim.Adam(model.parameters(), lr=self.cfg['learning_rate'])
            model_scheduler = SchedulerFactory.get_scheduler(model_optim, **scheduler_paras)
            return model_optim, model_scheduler

        model_optim = optim.Adam(self.model.parameters(), lr=self.cfg['learning_rate'])
        model_scheduler = SchedulerFactory.get_scheduler(model_optim, **scheduler_paras)
        return model_optim, model_scheduler

    def _select_criterion(self, which="nse"):
        if "mse" == which:
            criterion = nn.MSELoss()
        elif "nse" == which:
            criterion = NSELoss()
        else:
            return None
        return criterion

    def vali_train(self, model, vali_loader, device):
        # run_cfg = self.read_cfg()
        # self.model.eval()
        # self.model.eval()
        model.eval()
        mse = nn.MSELoss()
        cnt = 0
        mse_mean = 0
        nse_mean = 0
        mse_mean_allD = 0
        nse_mean_allD = 0

        with torch.no_grad():
            for x_seq, x_seq_mark, y_seq_past, y_seq_future, _, _ in vali_loader:
                x_seq, x_seq_mark, y_seq_past, y_seq_future = \
                    x_seq.to(self.device), x_seq_mark.to(self.device), y_seq_past.to(self.device), \
                        y_seq_future.to(self.device)

                batch_size = y_seq_past.shape[0]
                tgt_len = y_seq_past.shape[1] + y_seq_future.shape[1]
                tgt_size = y_seq_future.shape[2]
                pred_len = y_seq_future.shape[1]

                if self.cfg['sub_model'] in ['var_plus', 'var_plus_s', 'var_plus_sp'] or self.cfg[
                    'sub_model'] == 'lstm' or self.cfg['model'] in MODELLIST:
                    train_x_mean, train_y_mean = vali_loader.dataset.get_means()
                    train_x_std, train_y_std = vali_loader.dataset.get_stds()
                    # print(train_x_mean, train_x_std)  # TEST
                    y_hat_tci, y_hat_ep = model(x_seq, x_seq_mark, train_x_mean, train_x_std, train_y_mean,
                                                train_y_std)
                else:
                    y_hat_tci, y_hat_ep = model(x_seq, x_seq_mark)  # TEST:noNorm
                y_hat_tci_future = y_hat_tci[:, -pred_len:, :]  #
                y_hat_ep_future = y_hat_ep[:, -pred_len:, :]  #

                nse_value = np.array([calc_nse_torch(y_seq_future[:, :, 0:1], y_hat_tci_future)[0].cpu(),
                                      calc_nse_torch(y_seq_future[:, :, 1:2], y_hat_ep_future)[0].cpu()])

                mse_value = np.array(
                    [mse(y_hat_tci_future, y_seq_future[:, :, 0:1]).item(),
                     mse(y_hat_ep_future, y_seq_future[:, :, 1:2]).item()])

                cnt += 1
                nse_mean = nse_mean + (nse_value - nse_mean) / cnt  # Welford’s method
                mse_mean = mse_mean + (mse_value - mse_mean) / cnt  # Welford’s method

        # model.train()
        return mse_mean, nse_mean

    def vali(self, model, vali_loader, criterion):
        # run_cfg = self.read_cfg()
        # self.model.eval()
        model.eval()

        mse = nn.MSELoss()

        pred_tci = []
        pred_ep = []
        obs = []

        nse_all = []
        mse_all = []

        with torch.no_grad():
            for x_seq, x_seq_mark, y_seq_past, y_seq_future, _, _ in vali_loader:
                x_seq, x_seq_mark, y_seq_past, y_seq_future = \
                    x_seq.to(self.device), x_seq_mark.to(self.device), y_seq_past.to(self.device), \
                        y_seq_future.to(self.device)

                batch_size = y_seq_past.shape[0]
                tgt_len = y_seq_past.shape[1] + y_seq_future.shape[1]
                tgt_size = y_seq_future.shape[2]
                past_len = y_seq_past.shape[1]
                pred_len = y_seq_future.shape[1]

                if self.cfg['sub_model'] in ['var_plus', 'var_plus_s', 'var_plus_sp'] or self.cfg[
                    'sub_model'] == 'lstm' or self.cfg['model'] in MODELLIST:
                    train_x_mean, train_y_mean = vali_loader.dataset.get_means()
                    train_x_std, train_y_std = vali_loader.dataset.get_stds()
                    # print(train_x_mean, train_x_std)  # TEST
                    y_hat_tci, y_hat_ep = model(x_seq, x_seq_mark, train_x_mean, train_x_std, train_y_mean,
                                                train_y_std)
                else:
                    y_hat_tci, y_hat_ep = model(x_seq, x_seq_mark)  # TEST:noNorm
                y_hat_tci_future = y_hat_tci[:, -pred_len:, :]  #
                y_hat_ep_future = y_hat_ep[:, -pred_len:, :]  #

                basin_tci = vali_loader.dataset.local_rescale(
                    y_hat_tci_future.cpu().numpy(), variable='output')[:, :, 0:1]
                basin_ep = vali_loader.dataset.local_rescale(
                    y_hat_ep_future.cpu().numpy(), variable='output')[:, :, 1:2]

                obs_tci = vali_loader.dataset.local_rescale(
                    y_seq_future.cpu().numpy(), variable='output')[:, :, 0:1]
                obs_ep = vali_loader.dataset.local_rescale(
                    y_seq_future.cpu().numpy(), variable='output')[:, :, 1:2]

                nse_basin = np.array([calc_nse(obs_tci, basin_tci)[0],
                                      calc_nse(obs_ep, basin_ep)[0]])
                mse_basin = np.array(
                    [calc_mse(obs_tci, basin_tci)[0],
                     calc_mse(obs_ep, basin_ep)[0]])

                nse_all.append(nse_basin)
                mse_all.append(mse_basin)

        mse_value = np.mean(mse_all, axis=0)
        nse_value = np.mean(nse_all, axis=0)
        print(mse_value, nse_value)
        # model.train()
        return mse_value, nse_value

    def train_epoch(self, model, data_loader, optimizer, scheduler, loss_func, device):
        """
        Train model for a single epoch.

        param model: A torch.nn.Module implementing the Transformer model.
        param data_loader: A PyTorch DataLoader, providing the trainings data in mini batches.
        param optimizer: One of PyTorch optimizer classes.
        param scheduler: scheduler of learning rate.
        param loss_func: The loss function to minimize.
        param device: device for data and models
        """
        # set model to train mode (important for dropout)
        model.train()
        cnt = 0
        loss_mean = 0
        for x_seq, x_seq_mark, y_seq_past, y_seq_future, _, y_stds in data_loader:  # origin:但其实没用到的
            # delete previously stored gradients from the model
            optimizer.zero_grad()
            # push data to GPU (if available)
            x_seq, x_seq_mark, y_seq_past, y_seq_future, y_stds = x_seq.to(device), x_seq_mark.to(
                device), y_seq_past.to(device), y_seq_future.to(device), y_stds.to(device)

            batch_size = y_seq_past.shape[0]
            past_len = y_seq_past.shape[1]
            pred_len = y_seq_future.shape[1]
            tgt_len = past_len + pred_len
            tgt_size = y_seq_future.shape[2]

            if self.cfg['model'] in MODELLIST:
                # get model predictions
                train_x_mean, train_y_mean = data_loader.dataset.get_means()
                train_x_std, train_y_std = data_loader.dataset.get_stds()
                y_hat_tci, y_hat_ep = model(x_seq, x_seq_mark, train_x_mean, train_x_std, train_y_mean, train_y_std)
                # TEST:noNorm
                y_hat = torch.cat((y_hat_tci, y_hat_ep), dim=2)
                y_hat_tci_future = y_hat_tci[:, -pred_len:, :]  #
                y_hat_ep_future = y_hat_ep[:, -pred_len:, :]  #
            # elif self.cfg['model'] in ['RR']:
            #
            #     dec_inputs = torch.zeros((batch_size, tgt_len, tgt_size)).to(device)
            #     dec_inputs[:, :-pred_len, :] = y_seq_past
            #     y_hat_tci, y_hat_ep = model(x_seq, x_seq_mark, train_x_mean, train_x_std, train_y_mean, train_y_std)
            #     # TEST:noNorm
            #     y_hat = torch.cat((y_hat_tci, y_hat_ep), dim=2)
            #     y_hat_tci_future = y_hat_tci[:, -pred_len:, :]  #
            #     y_hat_ep_future = y_hat_ep[:, -pred_len:, :]  #


            # calculate loss
            if type(loss_func).__name__ == "NSELoss":
                y_stds = y_stds.to(device)
                if self.cfg["loss_all"]:
                    # 全部天数进行loss
                    # y_true = torch.cat((y_seq_past, y_seq_future), dim=1)

                    loss_pred_tci = loss_func(y_hat_tci[:, -pred_len:, :], y_seq_future[:, :, 0:1], y_stds[:, 0:1])
                    loss_pred_ep = loss_func(y_hat_ep[:, -pred_len:, :], y_seq_future[:, :, 1:2], y_stds[:, 1:2])

                    loss_past_tci = loss_func(y_hat_tci[:, :past_len, :], y_seq_past[:, :, 0:1], y_stds[:, 0:1])
                    loss_past_ep = loss_func(y_hat_tci[:, :past_len, :], y_seq_past[:, :, 1:2], y_stds[:, 1:2])

                    loss = loss_pred_tci * 1.0 + loss_past_tci * 1.0 + loss_pred_ep * 0.1 + loss_past_ep * 0.1
                else:
                    # 仅预测天进行loss
                    # TEST-加权！！！
                    # print("y_stds.shape", y_stds.shape)  # TEST
                    if self.cfg['sub_model'] == 'nn_pro_loss':
                        raise NotImplementedError("已被取消")
                    elif self.cfg['sub_model'] in ['var_plus', 'var_plus_s', 'var_plus_sp'] or self.cfg[
                        'sub_model'] == 'lstm' or self.cfg[
                        "model"] in MODELLIST:
                        # test-3:不反归一化or在sac里面归一化
                        loss_tci = loss_func(y_hat_tci[:, -pred_len:, :], y_seq_future[:, :, 0:1], y_stds[:, 0:1])
                        loss_ep = loss_func(y_hat_ep[:, -pred_len:, :], y_seq_future[:, :, 1:2], y_stds[:, 1:2])

                        # loss = loss_tci * 0.9 + loss_ep * 0.1  # TODO: 0.9-0.1
                        # loss = loss_tci * 1.0 + loss_ep * 0.1  # TODO: 1-0.1
                        loss = loss_tci * 1.0 + loss_ep * 0.0  # TODO: 1-0
                        # loss = loss_tci * 1.0 + loss_ep * 1.0  # TODO: 1-1
                    else:
                        loss_tci = loss_func(y_hat_tci[:, -pred_len:, :], y_seq_future[:, :, 0:1], y_stds[:, 0:1])
                        loss_ep = loss_func(y_hat_ep[:, -pred_len:, :], y_seq_future[:, :, 1:2], y_stds[:, 1:2])
                        # loss = loss_tci * 0.9 + loss_ep * 0.1
                        loss = loss_tci * 1.0 + loss_ep * 0.1
                        if 'ep' in self.cfg['sub_model']:  # TODO：TEST:后面记得去掉
                            loss = loss_ep

            else:  # NOTE：MSE
                if self.cfg["loss_all"]:
                    raise NotImplementedError("还未考虑")
                else:
                    if self.cfg['sub_model'] == 'nn_pro_loss':
                        raise NotImplementedError("已被取消")
                    elif self.cfg['sub_model'] in ['var_plus', 'var_plus_s', 'var_plus_sp'] or self.cfg[
                        'sub_model'] == 'lstm' or self.cfg[
                        "model"] in MODELLIST:
                        # test-3:不反归一化or在sac里面归一化
                        loss_tci = loss_func(y_hat_tci[:, -pred_len:, :], y_seq_future[:, :, 0:1])
                        loss_ep = loss_func(y_hat_ep[:, -pred_len:, :], y_seq_future[:, :, 1:2])

                        loss = loss_tci * 1.0 + loss_ep * 0.0  # TODO: 1-0
                    else:
                        loss_tci = loss_func(y_hat_tci[:, -pred_len:, :], y_seq_future[:, :, 0:1])
                        loss_ep = loss_func(y_hat_ep[:, -pred_len:, :], y_seq_future[:, :, 1:2])
                        loss = loss_tci * 1.0 + loss_ep * 0.0  # TODO

            # calculate gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # calculate mean loss
            cnt += 1
            loss_mean = loss_mean + (loss.item() - loss_mean) / cnt  # Welford’s method
        scheduler.step()
        print('loss_all:', self.cfg["loss_all"])
        return loss_mean

    def train(self, cfg):
        if self.cfg['stage'] == 'train':
            self.setup_train()
            # saving dir
            saving_root = cfg['run_dir'] / 'train'
            if not saving_root.is_dir():
                saving_root.mkdir()
            log_file = saving_root / "log_train.csv"
        else:
            raise NotImplementedError("wrong")

        train_data, train_loader = self._get_data(cfg=self.cfg, stage='train')
        vali_data, vali_loader = self._get_data(cfg=self.cfg, stage='val')

        # "type" chose in [none, warm_up, cos_anneal, exp_decay]  # TODO
        if cfg["warm_up"]:
            scheduler_paras = {"scheduler_type": "warm_up", "warm_up_epochs": cfg['epochs'] * 0.25, "decay_rate": 0.99}
        else:
            scheduler_paras = {"scheduler_type": "none"}
        optimizer, scheduler = self._select_optimizer(scheduler_paras)
        loss_func = self._select_criterion(self.cfg['loss'])

        print(f"----------------train------------------------\n"
              f"Parameters count:{count_parameters(self.model)}")

        with open(log_file, "wt") as f:
            f.write(f"parameters_count:{count_parameters(self.model)}\n")
            f.write("epoch,train_loss_iterated,train_mse,val_mse,train_nse,val_nse\n")

        mse_log = BestModelLog(self.model, saving_root, "min_mse", high_better=False)
        nse_log = BestModelLog(self.model, saving_root, "max_nse", high_better=True)
        nse_log2 = BestModelLog(self.model, saving_root, "max_nse_train", high_better=True)
        # newest_log = BestModelLog(model, saving_root, "newest", high_better=True)
        epoch_log = BestModelLog(self.model, saving_root, "epoch", high_better=True, log_all=False)  # NOTE:不再存储所有的epoch

        t1 = time.time()
        for i in range(cfg['epochs']):
            print(f"Training progress: {i} / {cfg['epochs']}")
            train_loss_iterated = self.train_epoch(self.model, train_loader, optimizer, scheduler, loss_func,
                                                   self.device)
            print("train_loss_iterated", train_loss_iterated)  # TEST

            #  NOTE:注释后可以加速训练
            mse_train, nse_train = self.vali_train(self.model, train_loader, self.device)
            mse_train = 0.0
            nse_train = 0.0
            mse_val, nse_val = self.vali(self.model, vali_loader, self.device)
            # mse_train, nse_train = mse_val, nse_val  # NOTE: no val train
            # mse_val, nse_val = self.vali_train(self.model, vali_loader, self.device)

            print(f"train mse:{mse_train},train nse: {nse_train}\n"
                  f"val mse:{mse_val},val nse: {nse_val}\n"
                  f"------------------------------------")

            # mse_val, nse_val = mse_val[0], nse_val[0]  # TEST: loss分开算了，这里选径流最好
            with open(log_file, "at") as f:
                f.write(f"{i},{train_loss_iterated},{mse_train},{mse_val},{nse_train},{nse_val}\n")
            mse_log.update(self.model, mse_val[0], i)
            nse_log.update(self.model, nse_val[0], i)
            # nse_log2.update(self.model, nse_train[0], i)  #ADD:12月26号为了加速注释了
            epoch_log.update(self.model, i, i)

            # 最后一个epoch简易验证测试集
            if i == (cfg['epochs'] - 1):
                test_data, test_loader = self._get_data(cfg=self.cfg, stage='test')
                mse_test, nse_test = self.vali(self.model, test_loader, self.device)
                print("last epochs:")
                print(f"test mse:{mse_test},test nse: {nse_test}\n"
                      f"------------------------------------")

        t2 = time.time()
        print(f"Training used time:{t2 - t1}")

        return self.model

    def test(self, cfg):
        '''
        实际是test_single，就是利用全站点训练了一个模型，然后在每个站点上面测试
        '''
        # run_cfg = self.read_cfg()

        self.cfg['test_dir'] = Path(self.run_cfg['test_dir'])
        self.cfg['data_dir'] = Path(self.run_cfg['data_dir'])

        test_data_dict, test_loader_dict = self._get_data(cfg=self.cfg, stage='test', single=True)


        # 可以选择多个训练模型进行test，这边节省时间，仅选择(max_nse)*.pkl
        # best_epochs = ["(max_nse)*.pkl", "(max_nse_train)*.pkl", "(min_mse)*.pkl", "(epoch)_*.pkl"]  # 选vali最优
        best_epochs = ["(max_nse)*.pkl"]  # 选vali最优 TODO
        for best_epoch in best_epochs:
            # if self.cfg['finetune_name'] is not None:
            #     p = self.run_cfg['finetune_dir'] / self.cfg['finetune_name']
            #     best_path = list(p.glob(f"{best_epoch}"))
            # else:
            if self.cfg['finetune_name'] is None:  # 普通测试的读取模型代码
                best_path = list(self.run_cfg['train_dir'].glob(f"{best_epoch}"))
                print(best_path[0])

                self.model.load_state_dict(torch.load(best_path[0], map_location=self.device))

            mse = nn.MSELoss()

            test_result = pd.DataFrame(columns=['basin', 'nse_tci', 'mse_tci', 'nse_ep', 'mse_ep',
                                                'kge_tci', 'rmse_tci', 'tpe-2%_tci','log-nse_tci'])
            tci_result_everyDay = pd.DataFrame(columns=['basin', 'pred_day', 'nse_tci', 'mse_tci'])

            self.model.eval()
            with torch.no_grad():
                for basin, test_loader in test_loader_dict.items():#NOTE:origin，united
                # for full_index in test_data_dict:  # NOTE:yr_s and separated
                #     basin = full_index  # NOTE:yr_s
                #     test_loader = test_loader_dict[full_index]  # NOTE:yr_s

                    if self.cfg['finetune_name'] is not None:  # finetune后测试的读取模型代码
                        p = self.run_cfg['finetune_dir'] / self.cfg['finetune_name'] / basin
                        best_path = list(p.glob(f"{best_epoch}"))
                        if len(best_path) == 0:
                            continue
                        self.model.load_state_dict(torch.load(best_path[0], map_location=self.device))

                    basin_pred_tci = []
                    basin_pred_ep = []
                    basin_obs = []

                    for x_seq, x_seq_mark, y_seq_past, y_seq_future, _, _ in test_loader:
                        x_seq, x_seq_mark, y_seq_past, y_seq_future = \
                            x_seq.to(self.device), x_seq_mark.to(self.device), y_seq_past.to(self.device), \
                                y_seq_future.to(self.device)

                        past_len = y_seq_past.shape[1]
                        pred_len = y_seq_future.shape[1]
                        assert pred_len == self.run_cfg["pred_len"]

                        # pred runoff
                        if self.cfg['sub_model'] in ['var_plus', 'var_plus_s', 'var_plus_sp'] or self.cfg[
                            'sub_model'] == 'lstm' or self.cfg['model'] in MODELLIST:
                            train_x_mean, train_y_mean = test_loader.dataset.get_means()
                            train_x_std, train_y_std = test_loader.dataset.get_stds()

                            y_hat_tci, y_hat_ep = self.model(x_seq, x_seq_mark, train_x_mean, train_x_std, train_y_mean,
                                                             train_y_std)
                        else:
                            y_hat_tci, y_hat_ep = self.model(x_seq, x_seq_mark)  # TEST:noNorm

                        y_hat_tci_future = y_hat_tci[:, -pred_len:, :]
                        y_hat_ep_future = y_hat_ep[:, -pred_len:, :]

                        # batch数据汇总
                        basin_pred_tci.append(y_hat_tci_future)
                        basin_pred_ep.append(y_hat_ep_future)
                        basin_obs.append(y_seq_future)  # NOTE：这里是用train的mean和std还原的obs，可能不太准确

                    #TEST:现在var_plus输出的也是归一化后的数据了
                    # NOTE:model输出数据,归一化还原
                    basin_tci_all = torch.cat(basin_pred_tci)
                    basin_ep_all = torch.cat(basin_pred_ep)
                    basin_pred_all = torch.cat((basin_tci_all, basin_ep_all), dim=2).cpu().numpy()


                    basin_tci_all = test_loader.dataset.local_rescale(basin_pred_all, variable='output')[:, :, 0:1]
                    basin_ep_all = test_loader.dataset.local_rescale(basin_pred_all, variable='output')[:, :, 1:2]

                    # 原始数据
                    if self.cfg['forcing_type'] in ['united', 'separated'] or 'yr' in cfg['forcing_type']:
                        basin_obs_all = torch.cat(basin_obs).cpu().numpy()  # NOTE：这里是用train的mean和std还原的obs，可能不太准确
                        basin_obs_tci_all = test_loader.dataset.local_rescale(
                            basin_obs_all, variable='output')[:, :, 0:1]
                        basin_obs_ep_all = test_loader.dataset.local_rescale(
                            basin_obs_all, variable='output')[:, :, 1:2]
                    else:
                        basin_obs_all = torch.cat(basin_obs).cpu().numpy()  # NOTE：这里是用train的mean和std还原的obs，可能不太准确

                        basin_obs_tci_all = test_loader.dataset.local_rescale(
                            basin_obs_all, variable='output')[:, :, 0:1]
                        basin_obs_ep_all = test_loader.dataset.local_rescale(
                            basin_obs_all, variable='output')[:, :, 1:2]

                    # 存预测和原始数据
                    if self.cfg['forcing_type'] in ['united', 'separated'] or 'yr' in self.cfg['forcing_type']:
                        pd_range = pd.date_range(
                            pd.to_datetime(test_loader.dataset.start_dates[2]) + pd.Timedelta(days=past_len),
                            pd.to_datetime(test_loader.dataset.end_dates[2]) - pd.Timedelta(days=pred_len - 1))

                        predict_result = pd.DataFrame(
                            data={'obs_tci': basin_obs_tci_all.reshape(-1, pred_len).tolist(),
                                  'pet': basin_obs_ep_all.reshape(-1, pred_len).tolist(),
                                  'pred_tci': basin_tci_all.reshape(-1, pred_len).tolist(),
                                  'pred_ep': basin_ep_all.reshape(-1, pred_len).tolist()},
                            index=pd_range)
                    else:
                        date_index = test_loader.dataset.date_index_dict[basin]
                        date_index = date_index[past_len:len(date_index) - pred_len + 1]

                        predict_result = pd.DataFrame(
                            data={'obs_tci': basin_obs_tci_all.reshape(-1, pred_len).tolist(),
                                  'pet': basin_obs_ep_all.reshape(-1, pred_len).tolist(),
                                  'pred_tci': basin_tci_all.reshape(-1, pred_len).tolist(),
                                  'pred_ep': basin_ep_all.reshape(-1, pred_len).tolist()},
                            index=date_index)

                    # bestepoch在window中无法使用，需要进行替换，通过将 * 替换成其他的即可
                    # 由于我们仅测试一个bestepoch，所以这边不需要进行区分
                    # best_epoch = best_epoch.replace('*',"_")
                    # basin_dir = self.cfg["test_dir"] / f"detail_{best_epoch}" / basin
                    basin_dir = self.cfg["test_dir"] / 'basins' / basin


                    if not os.path.isdir(basin_dir):
                        basin_dir.mkdir(parents=True)


                    predict_result.to_csv(basin_dir / f"{basin}_data.txt",
                                          encoding="utf-8", sep=',', index_label="start_date")

                    # 每个basin的损失函数
                    nse_value = np.array([calc_nse(basin_obs_tci_all, basin_tci_all)[0],
                                          calc_nse(basin_obs_ep_all, basin_ep_all)[0]])  # TEST,loss分开
                    mse_value = np.array(
                        [calc_mse(basin_tci_all, basin_obs_tci_all)[0],
                         calc_mse(basin_ep_all, basin_obs_ep_all)[0]])

                    # kge
                    kge_value = np.array([calc_kge(basin_obs_tci_all, basin_tci_all)[0],
                                          calc_kge(basin_obs_ep_all, basin_ep_all)[0]])  # TEST,loss分开
                    # rmse
                    rmse_value = np.array([calc_rmse(basin_obs_tci_all, basin_tci_all)[0],
                                          calc_rmse(basin_obs_ep_all, basin_ep_all)[0]])  # TEST,loss分开
                    # tpe-2%
                    tpe_value = np.array([calc_tpe(basin_obs_tci_all, basin_tci_all, 0.02)[0],
                                          calc_tpe(basin_obs_ep_all, basin_ep_all, 0.02)[0]])  # TEST,loss分开
                    # log-nse_tci
                    log_nse_value = np.array([calc_log_nse(basin_obs_tci_all, basin_tci_all)[0],
                                          calc_log_nse(basin_obs_ep_all, basin_ep_all)[0]])  # TEST,loss分开

                    print(nse_value, mse_value)  # TEST
                    element = pd.DataFrame({'basin': [basin],'nse_tci': [nse_value[0]], 'mse_tci': [mse_value[0]],
                                            'nse_ep': [nse_value[1]], 'mse_ep': [mse_value[1]], 'kge_tci':[kge_value[0]],
                                            'rmse_tci':[rmse_value[0]], 'tpe-2%_tci': [tpe_value[0]], 'log-nse_tci':[log_nse_value[0]]})
                    test_result = pd.concat([test_result, element])

                    # 在新的panda库中append方法已经被弃用
                    # test_result = test_result.append([{'basin': [basin],
                    #                                                'nse_tci': [nse_value[0]], 'mse_tci': [mse_value[0]],
                    #                                                'nse_ep': [nse_value[1]], 'mse_ep': [mse_value[1]]}])

                    # 上面是7天均值，下面这个是7天分开：
                    id = [[], []]
                    for i in range(1, pred_len + 1):
                        id[0].append(basin)
                        id[1].append(f'day{i}')
                    nse_value_everyDay = np.array(calc_nse(basin_obs_tci_all, basin_tci_all)[1])
                    mse_value_everyDay = np.array(calc_mse(basin_obs_tci_all, basin_tci_all)[1])


                    # tci_result_everyDay = tci_result_everyDay.append(pd.DataFrame({'basin': id[0],
                    #                                                                'pred_day': id[1],
                    #                                                                'nse_tci': nse_value_everyDay.flatten(),
                    #                                                                'mse_tci': mse_value_everyDay.flatten()}))
                    tci_result_everyDay = pd.concat([tci_result_everyDay, pd.DataFrame({'basin': id[0],
                                                                                   'pred_day': id[1],
                                                                                   'nse_tci': nse_value_everyDay.flatten(),
                                                                                   'mse_tci': mse_value_everyDay.flatten()})])


                # 存损失函数
                print(test_result)  # TEST
                test_result.to_csv(self.cfg["test_dir"] / f"loss_data_{best_epoch}.txt", encoding="utf-8", sep=',',
                                   index_label='basin')
                tci_result_everyDay.to_csv(self.cfg["test_dir"] / f"loss_data_everyDay_{best_epoch}.txt",
                                           encoding="utf-8", sep=',', index_label='basin')

                results_value = test_result.values
                tci_nse = results_value[:, 1]
                tci_mse = results_value[:, 2]
                ep_nse = results_value[:, 3]
                ep_mse = results_value[:, 4]
                tci_kge = results_value[:, 5]
                tci_rmse = results_value[:, 6]
                tci_tpe = results_value[:, 7]
                tci_log_nse = results_value[:, 8]
                # 将tci_nse中的所有负值全置0
                tci_nse_NoNegativeValue = results_value[:, 1].copy()
                tci_nse_NoNegativeValue[tci_nse_NoNegativeValue < 0] = 0


                with open(self.cfg["test_dir"] / f"test_statistics_{best_epoch}.txt", 'w+') as f:
                    f.write("tci:\n")
                    f.write(f"nse_median:{np.median(tci_nse)},nse_mean:{np.mean(tci_nse)}\n")
                    f.write(f"NoNegativeValue nse_median:{np.median(tci_nse_NoNegativeValue)},nse_mean:{np.mean(tci_nse_NoNegativeValue)}\n")
                    f.write(f"mse_median:{np.median(tci_mse)},mse_mean:{np.mean(tci_mse)}\n")
                    f.write(f"kge_median:{np.median(tci_kge)},kge_mean:{np.mean(tci_kge)}\n")
                    f.write(f"rmse_median:{np.median(tci_rmse)},rmse_mean:{np.mean(tci_rmse)}\n")
                    f.write(f"tpe_median:{np.median(tci_tpe)},tpe_mean:{np.mean(tci_tpe)}\n")
                    f.write(f"log-nse_median:{np.median(tci_log_nse)},log-nse_mean:{np.mean(tci_log_nse)}\n")
                    f.write("ep:\n")
                    f.write(f"nse_median:{np.median(ep_nse)},nse_mean:{np.mean(ep_nse)}\n")
                    f.write(f"mse_median:{np.median(ep_mse)},mse_mean:{np.mean(ep_mse)}\n")



                # tci_result_everyDay_value = tci_result_everyDay.values
                # tci_nse = tci_result_everyDay_value[:, 1]
                # tci_mse = tci_result_everyDay_value[:, 2]
                # with open(self.cfg["test_dir"] / f"test_statistics_everyDay_{best_epoch}.txt", 'w+') as f:
                #     f.write("tci:\n")
                #     f.write(f"nse_median:{np.median(tci_nse, axis=0)},nse_mean:{np.mean(tci_nse, axis=0)}\n")
                #     f.write(f"mse_median:{np.median(tci_mse, axis=0)},mse_mean:{np.mean(tci_mse, axis=0)}\n")
                tci_nse_mean = tci_result_everyDay.groupby('pred_day').mean(numeric_only=True)
                tci_nse_median = tci_result_everyDay.groupby('pred_day').median(numeric_only=True)
                # df = pd.DataFrame(columns=['pred_day', 'statistics', 'nse_tci', 'mse_tci'])
                test = pd.concat([tci_nse_mean, tci_nse_median]).reset_index(drop=False)
                s = []
                for i in range(1, pred_len + 1):
                    s.append('mean')
                for i in range(1, pred_len + 1):
                    s.append('median')
                test.insert(0, 'statistics', s)
                test.set_index(['statistics', 'pred_day'], inplace=True)
                test.to_csv(self.cfg["test_dir"] / f"test_statistics_everyDay_{best_epoch}.txt", encoding="utf-8",
                            sep=',')

        return

    def finetune_single(self, cfg):
        self.setup_finetune()

        train_datas, train_loaders = self._get_data(cfg=self.cfg, stage='train')
        vali_data, val_loaders = self._get_data(cfg=self.cfg, stage='val')
        test_datas, test_loaders = self._get_data(cfg=self.cfg, stage='test')

        # basins_list = pd.read_csv(self.run_cfg["basins_list_path"], header=None, dtype=str)[0].values.tolist()
        # basins_list = pd.read_csv(self.cfg["basins_list_path"], header=None, dtype=str)[0].values.tolist()
        for full_index in train_datas:
            print(full_index)
            train_loader = train_loaders[full_index]
            val_loader = val_loaders[full_index]
            test_loader = test_loaders[full_index]

            best_epoch = "(max_nse)*.pkl"  # 选vali最优
            run_cfg = self.read_cfg()
            best_path = list(run_cfg['train_dir'].glob(f"{best_epoch}"))
            print(best_path)

            models = importlib.import_module("models")
            Model = getattr(models, cfg["model"])
            self.model = None
            self.model = Model(self.run_cfg).to(self.device)
            self.model.load_state_dict(torch.load(best_path[0], map_location=self.device))
            # "type" chose in [none, warm_up, cos_anneal, exp_decay]  # TODO
            if cfg["warm_up"]:
                scheduler_paras = {"scheduler_type": "warm_up", "warm_up_epochs": cfg['epochs'] * 0.25,
                                   "decay_rate": 0.99}
            else:
                scheduler_paras = {"scheduler_type": "none"}
            optimizer, scheduler = self._select_optimizer(scheduler_paras)
            loss_func = self._select_criterion(self.cfg['loss'])

            print(f"----------------train------------------------\n"
                  f"Parameters count:{count_parameters(self.model)}")
            saving_root = self.cfg['finetune_dir'] / full_index
            if not saving_root.is_dir():
                saving_root.mkdir()
            log_file = saving_root / "log_train.csv"
            with open(log_file, "wt") as f:
                f.write(f"parameters_count:{count_parameters(self.model)}\n")
                f.write("epoch,train_loss_iterated,train_mse,val_mse,train_nse,val_nse\n")

            mse_log = BestModelLog(self.model, saving_root, "min_mse", high_better=False)
            nse_log = BestModelLog(self.model, saving_root, "max_nse", high_better=True)
            # newest_log = BestModelLog(model, saving_root, "newest", high_better=True)
            epoch_log = BestModelLog(self.model, saving_root, "epoch", high_better=True,
                                     log_all=False)  # NOTE:不再存储所有的epoch

            t1 = time.time()
            for i in range(cfg['epochs']):
                print(f"Training progress: {i} / {cfg['epochs']}")
                train_loss_iterated = self.train_epoch(self.model, train_loader, optimizer, scheduler, loss_func,
                                                       self.device)
                print("train_loss_iterated", train_loss_iterated)  # TEST
                # mse_train, nse_train is not need to be calculated (via eval_model function),
                # and you can comment the next line to speed up
                mse_train, nse_train = '', ''
                mse_train, nse_train = self.vali_train(self.model, train_loader, self.device)  # NOTE:注释后可以加速训练
                mse_val, nse_val = self.vali(self.model, val_loader, self.device)

                print(f"train mse:{mse_train},train nse: {nse_train}\n"
                      f"val mse:{mse_val},val nse: {nse_val}\n"
                      f"------------------------------------")

                # mse_val, nse_val = mse_val[0], nse_val[0]  # TEST: loss分开算了，这里选径流最好

                with open(log_file, "at") as f:
                    f.write(f"{i},{train_loss_iterated},{mse_train},{mse_val},{nse_train},{nse_val}\n")
                mse_log.update(self.model, mse_val[0], i)
                nse_log.update(self.model, nse_val[0], i)
                epoch_log.update(self.model, i, i)

                # 最后一个epoch简易验证测试集
                if i == (cfg['epochs'] - 1):
                    mse_test, nse_test = self.vali(self.model, test_loader, self.device)
                    print("last epochs:")
                    print(f"test mse:{mse_test},test nse: {nse_test}\n"
                          f"------------------------------------")

            t2 = time.time()
            print(f"Training used time:{t2 - t1}")

        return self.model

    # 下面都是旧的代码，很可能已经无用了
    # 下面都是旧的代码，很可能已经无用了
    # 下面都是旧的代码，很可能已经无用了
    # def train_single(self, cfg):
    #     self.setup_train()
    #
    #     train_datas, train_loaders = self._get_data(cfg=self.cfg, stage='train')
    #     vali_datas, val_loaders = self._get_data(cfg=self.cfg, stage='val')
    #     test_datas, test_loaders = self._get_data(cfg=self.cfg, stage='test')
    #
    #     # basins_list = pd.read_csv(self.run_cfg["basins_list_path"], header=None, dtype=str)[0].values.tolist()
    #     # basins_list = pd.read_csv(self.cfg["basins_list_path"], header=None, dtype=str)[0].values.tolist()
    #     for full_index in train_datas:
    #         train_loader = train_loaders[full_index]
    #         val_loader = val_loaders[full_index]
    #         test_loader = test_loaders[full_index]
    #         models = importlib.import_module("models")
    #         Model = getattr(models, cfg["model"])
    #         self.model = Model(cfg).to(self.device)
    #         self.model.apply(weights_init_function)
    #         # "type" chose in [none, warm_up, cos_anneal, exp_decay]  # TODO
    #         if cfg["warm_up"]:
    #             scheduler_paras = {"scheduler_type": "warm_up", "warm_up_epochs": cfg['epochs'] * 0.25,
    #                                "decay_rate": 0.99}
    #         else:
    #             scheduler_paras = {"scheduler_type": "none"}
    #         optimizer, scheduler = self._select_optimizer(scheduler_paras)
    #         loss_func = self._select_criterion(self.cfg['loss'])
    #
    #         print(f"----------------train------------------------\n"
    #               f"Parameters count:{count_parameters(self.model)}")
    #         saving_root = self.cfg['train_dir'] / full_index
    #         if not saving_root.is_dir():
    #             saving_root.mkdir()
    #         log_file = saving_root / "log_train.csv"
    #         with open(log_file, "wt") as f:
    #             f.write(f"parameters_count:{count_parameters(self.model)}\n")
    #             f.write("epoch,train_loss_iterated,train_mse,val_mse,train_nse,val_nse\n")
    #
    #         mse_log = BestModelLog(self.model, saving_root, "min_mse", high_better=False)
    #         nse_log = BestModelLog(self.model, saving_root, "max_nse", high_better=True)
    #         # newest_log = BestModelLog(model, saving_root, "newest", high_better=True)
    #         epoch_log = BestModelLog(self.model, saving_root, "epoch", high_better=True,
    #                                  log_all=False)  # NOTE:不再存储所有的epoch
    #
    #         t1 = time.time()
    #         for i in range(cfg['epochs']):
    #             print(f"Training progress: {i} / {cfg['epochs']}")
    #             train_loss_iterated = self.train_epoch(self.model, train_loader, optimizer, scheduler, loss_func,
    #                                                    self.device)
    #             print("train_loss_iterated", train_loss_iterated)  # TEST
    #             # mse_train, nse_train is not need to be calculated (via eval_model function),
    #             # and you can comment the next line to speed up
    #             mse_train, nse_train = '', ''
    #             mse_train, nse_train = self.vali_train(self.model, train_loader, self.device)  # NOTE:注释后可以加速训练
    #             mse_val, nse_val = self.vali(self.model, val_loader, self.device)
    #
    #             print(f"train mse:{mse_train},train nse: {nse_train}\n"
    #                   f"val mse:{mse_val},val nse: {nse_val}\n"
    #                   f"------------------------------------")
    #
    #             # mse_val, nse_val = mse_val[0], nse_val[0]  # TEST: loss分开算了，这里选径流最好
    #
    #             with open(log_file, "at") as f:
    #                 f.write(f"{i},{train_loss_iterated},{mse_train},{mse_val},{nse_train},{nse_val}\n")
    #             mse_log.update(self.model, mse_val[0], i)
    #             nse_log.update(self.model, nse_val[0], i)
    #             epoch_log.update(self.model, i, i)
    #
    #             # 最后一个epoch简易验证测试集
    #             if i == (cfg['epochs'] - 1):
    #                 mse_test, nse_test = self.vali(self.model, test_loader, self.device)
    #                 print("last epochs:")
    #                 print(f"test mse:{mse_test},test nse: {nse_test}\n"
    #                       f"------------------------------------")
    #
    #         t2 = time.time()
    #         print(f"Training used time:{t2 - t1}")
    #
    #     return self.model

    def finetune(self, cfg):
        self.setup_finetune()
        models = importlib.import_module("models")
        Model = getattr(models, cfg["model"])

        best_epoch = "(max_nse)*.pkl"  # 选vali最优
        # best_epoch = "(epoch)_198*.pkl"  # 自定义
        run_cfg = self.read_cfg()
        best_path = list(run_cfg['train_dir'].glob(f"{best_epoch}"))
        print(best_path)

        basins_list = pd.read_csv(self.run_cfg["basins_list_path"], header=None, dtype=str)[0].values.tolist()
        # basins_list = pd.read_csv(self.cfg["basins_list_path"], header=None, dtype=str)[0].values.tolist()
        for i, basin in enumerate(basins_list):
            train_data, train_loader = self._get_data_finetune(cfg=self.cfg, stage='train', basin=basin)
            vali_data, vali_loader = self._get_data_finetune(cfg=self.cfg, stage='val', basin=basin)
            test_data, test_loader = self._get_data_finetune(cfg=self.cfg, stage='test', basin=basin)

            self.model = Model(self.run_cfg).to(self.device)
            self.model.load_state_dict(torch.load(best_path[0], map_location=self.device))
            # "type" chose in [none, warm_up, cos_anneal, exp_decay]  # TODO
            if cfg["warm_up"]:
                scheduler_paras = {"scheduler_type": "warm_up", "warm_up_epochs": cfg['epochs'] * 0.25,
                                   "decay_rate": 0.99}
            else:
                scheduler_paras = {"scheduler_type": "none"}
            optimizer, scheduler = self._select_optimizer(scheduler_paras)
            loss_func = self._select_criterion(self.cfg['loss'])

            print(f"----------------train------------------------\n"
                  f"Parameters count:{count_parameters(self.model)}")
            saving_root = cfg['run_dir'] / 'finetune' / basin
            if not saving_root.is_dir():
                saving_root.mkdir()
            log_file = saving_root / "log_train.csv"
            with open(log_file, "wt") as f:
                f.write(f"parameters_count:{count_parameters(self.model)}\n")
                f.write("epoch,train_loss_iterated,train_mse,val_mse,train_nse,val_nse\n")

            mse_log = BestModelLog(self.model, saving_root, "min_mse", high_better=False)
            nse_log = BestModelLog(self.model, saving_root, "max_nse", high_better=True)
            # newest_log = BestModelLog(model, saving_root, "newest", high_better=True)
            epoch_log = BestModelLog(self.model, saving_root, "epoch", high_better=True,
                                     log_all=False)  # NOTE:不再存储所有的epoch

            t1 = time.time()
            for i in range(cfg['epochs']):
                print(f"Training progress: {i} / {cfg['epochs']}")
                train_loss_iterated = self.train_epoch(self.model, train_loader, optimizer, scheduler, loss_func,
                                                       self.device)
                print("train_loss_iterated", train_loss_iterated)  # TEST
                # mse_train, nse_train is not need to be calculated (via eval_model function),
                # and you can comment the next line to speed up
                mse_train, nse_train = '', ''
                mse_train, nse_train = self.vali_train(self.model, train_loader, self.device)  # NOTE:注释后可以加速训练
                if basin not in ['02125000', '06154410', '06291500', '12383500', '12388400']:
                    mse_val, nse_val = self.vali(self.model, vali_loader, self.device)
                else:
                    mse_val, nse_val = mse_train, nse_train
                    print("no val!!!")
                # mse_val, nse_val = self.vali_train(self.model, vali_loader, self.device)

                print(f"train mse:{mse_train},train nse: {nse_train}\n"
                      f"val mse:{mse_val},val nse: {nse_val}\n"
                      f"------------------------------------")

                mse_val, nse_val = mse_val[0], nse_val[0]  # TEST: loss分开算了，这里选径流最好
                with open(log_file, "at") as f:
                    f.write(f"{i},{train_loss_iterated},{mse_train},{mse_val},{nse_train},{nse_val}\n")
                mse_log.update(self.model, mse_val, i)
                nse_log.update(self.model, nse_val, i)
                epoch_log.update(self.model, i, i)

                # 最后一个epoch简易验证测试集
                if i == (cfg['epochs'] - 1):
                    mse_test, nse_test = self.vali(self.model, test_loader, self.device)
                    print("last epochs:")
                    print(f"test mse:{mse_test},test nse: {nse_test}\n"
                          f"------------------------------------")

            t2 = time.time()
            print(f"Training used time:{t2 - t1}")

        return self.model

    def finetune_test(self, cfg):
        '''
        实际是test_single
        实际是test_single
        实际是test_single
        '''
        models = importlib.import_module("models")
        Model = getattr(models, cfg["model"])

        mse = nn.MSELoss()

        best_epoch = "(max_nse)*.pkl"  # 选vali最优
        # best_epoch = "(epoch)_198*.pkl"  # 自定义
        basins_list = pd.read_csv(self.run_cfg["basins_list_path"], header=None, dtype=str)[0].values.tolist()

        # basins_list = pd.read_csv(self.cfg["basins_list_path"], header=None, dtype=str)[0].values.tolist()
        for i, basin in enumerate(basins_list):
            print(f"----------------finetune_test: {basin}------------------------")
            test_data, test_loader = self._get_data_finetune(cfg=self.cfg, stage='test', basin=basin)

            best_path = list((self.finetune_cfg['finetune_dir'] / basin).glob(f"{best_epoch}"))
            print(best_path)
            self.model = Model(self.run_cfg).to(self.device)
            self.model.load_state_dict(torch.load(best_path[0], map_location=self.device))

            test_result = pd.DataFrame(columns=['basin', 'nse_tci', 'mse_tci', 'nse_ep', 'mse_ep'])

            self.model.eval()
            with torch.no_grad():
                basin_pred_tci = []
                basin_pred_ep = []
                for x_seq, x_seq_mark, y_seq_past, y_seq_future, static_values, _ in test_loader:
                    x_seq, x_seq_mark, y_seq_past, y_seq_future, static_values = \
                        x_seq.to(self.device), x_seq_mark.to(self.device), y_seq_past.to(self.device), \
                            y_seq_future.to(self.device), static_values.to(self.device)

                    past_len = y_seq_past.shape[1]
                    pred_len = y_seq_future.shape[1]
                    assert pred_len == self.run_cfg["pred_len"]

                    # pred runoff
                    # if self.run_cfg["sub_model"] == 'var_plus' or self.run_cfg["sub_model"] == "var_plus_cycle" \
                    #         or self.cfg['sub_model'] == 'lstm' or self.cfg["model"] in [
                    #     "Trm", "LSTM", "RRS", "RR"]:
                    if self.run_cfg["sub_model"] in ['var_plus', 'var_plus_s', 'var_plus_sp'] or self.run_cfg[
                        "sub_model"] == "var_plus_cycle" \
                            or self.cfg['sub_model'] == 'lstm' or self.cfg["model"] in MODELLIST:
                        train_x_mean, train_y_mean = test_loader.dataset.get_means()
                        train_x_std, train_y_std = test_loader.dataset.get_stds()
                        # print(train_x_mean, train_x_std)  # TEST
                        y_hat_tci, y_hat_ep = self.model(x_seq, x_seq_mark, train_x_mean, train_x_std, train_y_mean,
                                                         train_y_std)
                    else:
                        y_hat_tci, y_hat_ep = self.model(x_seq, x_seq_mark)  # TEST:noNorm
                    # y_hat_tci, y_hat_ep = self.model(x_seq, x_seq_mark)
                    y_hat_tci_future = y_hat_tci[:, -pred_len:, :]
                    y_hat_ep_future = y_hat_ep[:, -pred_len:, :]

                    # batch数据汇总
                    basin_pred_tci.append(y_hat_tci_future)
                    basin_pred_ep.append(y_hat_ep_future)
                    # basin_obs.append(y_seq_future) #NOTE：这里是用train的mean和std还原的obs，应该不太准确

                # model输出数据,归一化还原
                basin_tci_all = torch.cat(basin_pred_tci)
                basin_ep_all = torch.cat(basin_pred_ep)
                basin_pred_all = torch.cat((basin_tci_all, basin_ep_all), dim=2).cpu().numpy()

                basin_tci_all = test_loader.dataset.local_rescale(basin_pred_all, variable='output')[:, :, 0:1]
                basin_ep_all = test_loader.dataset.local_rescale(basin_pred_all, variable='output')[:, :, 1:2]

                # 原始数据
                # basin_obs_all = torch.cat(basin_obs)#NOTE：这里是用train的mean和std还原的obs，应该不太准确
                basin_obs_all = test_loader.dataset.y_origin[basin]
                basin_obs_all = basin_obs_all[past_len:len(basin_obs_all) - pred_len + 1]
                basin_obs_tci_all = basin_obs_all[:, 0:1].reshape(-1, 1, 1)
                basin_obs_ep_all = basin_obs_all[:, 1:2].reshape(-1, 1, 1)

                # 存预测和原始数据
                date_index = test_loader.dataset.date_index_dict[basin]
                date_index = date_index[past_len:len(date_index) - pred_len + 1]

                # print(basin_obs_tci_all.shape, basin_obs_ep_all.shape)  # TEST
                # print(basin_tci_all.shape, basin_ep_all.shape)  # TEST
                # print(date_index.shape)  # TEST

                predict_result = pd.DataFrame(
                    data={'obs_tci': basin_obs_tci_all.flatten(),
                          'pet': basin_obs_ep_all.flatten(),
                          'pred_tci': basin_tci_all.flatten(),
                          'pred_ep': basin_ep_all.flatten()},
                    index=date_index)
                # print(self.run_cfg) #TEST
                basin_dir = self.run_cfg["test_dir"] / "finetune" / f"detail_{best_epoch}" / basin
                if not os.path.isdir(basin_dir):
                    basin_dir.mkdir(parents=True)
                else:
                    raise "finetune_test exists!!!"
                predict_result.to_csv(basin_dir / f"{basin}_data.txt",
                                      encoding="utf-8", sep=',', index_label="start_date")

                # 每个basin的损失函数
                nse_value = np.array([calc_nse(basin_obs_tci_all, basin_tci_all)[0],
                                      calc_nse(basin_obs_ep_all, basin_ep_all)[0]])  # TEST,loss分开
                mse_value = np.array(
                    [calc_mse(basin_tci_all, basin_obs_tci_all)[0],
                     calc_mse(basin_ep_all, basin_obs_ep_all)[0]])
                print(nse_value, mse_value)  # TEST
                test_result = test_result.append(pd.DataFrame({'basin': [basin],
                                                               'nse_tci': [nse_value[0]], 'mse_tci': [mse_value[0]],
                                                               'nse_ep': [nse_value[1]], 'mse_ep': [mse_value[1]]}))
        # 存损失函数
        print(test_result)  # TEST
        test_result.to_csv(self.run_cfg["test_dir"] / "finetune" / f"loss_data_{best_epoch}.txt", encoding="utf-8",
                           sep=',')

        results_value = test_result.values
        tci_nse = results_value[:, 1]
        tci_mse = results_value[:, 2]
        ep_nse = results_value[:, 3]
        ep_mse = results_value[:, 4]
        with open(self.run_cfg["test_dir"] / "finetune" / f"test_statistics_{best_epoch}.txt", 'w+') as f:
            f.write("tci:\n")
            f.write(f"nse_median:{np.median(tci_nse)},nse_mean:{np.mean(tci_nse)}\n")
            f.write(f"mse_median:{np.median(tci_mse)},mse_mean:{np.mean(tci_mse)}\n")
            f.write("ep:\n")
            f.write(f"nse_median:{np.median(ep_nse)},nse_mean:{np.mean(ep_nse)}\n")
            f.write(f"mse_median:{np.median(ep_mse)},mse_mean:{np.mean(ep_mse)}\n")

        print("---------tci---------\n"
              f"nse_median:{np.median(tci_nse)},nse_mean:{np.mean(tci_nse)}\n",
              f"mse_median:{np.median(tci_mse)},mse_mean:{np.mean(tci_mse)}\n")
        print("---------ep:---------\n"
              f"nse_median:{np.median(ep_nse)},nse_mean:{np.mean(ep_nse)}\n",
              f"mse_median:{np.median(ep_mse)},mse_mean:{np.mean(ep_mse)}\n")
        return

    def train_single_pre(self, cfg):
        self.setup_train()

        basins_list = pd.read_csv(self.cfg["basins_list_path"], header=None, dtype=str)[0].values.tolist()
        # basins_list = pd.read_csv(self.cfg["basins_list_path"], header=None, dtype=str)[0].values.tolist()
        for i, basin in enumerate(basins_list):
            train_data, train_loader = self._get_data_finetune(cfg=self.cfg, stage='train', basin=basin)
            vali_data, vali_loader = self._get_data_finetune(cfg=self.cfg, stage='val', basin=basin)
            test_data, test_loader = self._get_data_finetune(cfg=self.cfg, stage='test', basin=basin)

            # "type" chose in [none, warm_up, cos_anneal, exp_decay]  # TODO
            if cfg["warm_up"]:
                scheduler_paras = {"scheduler_type": "warm_up", "warm_up_epochs": cfg['epochs'] * 0.25,
                                   "decay_rate": 0.99}
            else:
                scheduler_paras = {"scheduler_type": "none"}
            optimizer, scheduler = self._select_optimizer(scheduler_paras)
            loss_func = self._select_criterion(self.cfg['loss'])

            print(f"----------------train------------------------\n"
                  f"Parameters count:{count_parameters(self.model)}")
            saving_root = cfg['run_dir'] / 'train' / basin
            if not saving_root.is_dir():
                saving_root.mkdir()
            log_file = saving_root / "log_train.csv"
            with open(log_file, "wt") as f:
                f.write(f"parameters_count:{count_parameters(self.model)}\n")
                f.write("epoch,train_loss_iterated,train_mse,val_mse,train_nse,val_nse\n")

            mse_log = BestModelLog(self.model, saving_root, "min_mse", high_better=False)
            nse_log = BestModelLog(self.model, saving_root, "max_nse", high_better=True)
            nse_log2 = BestModelLog(self.model, saving_root, "max_nse_train", high_better=True)
            # newest_log = BestModelLog(model, saving_root, "newest", high_better=True)
            epoch_log = BestModelLog(self.model, saving_root, "epoch", high_better=True,
                                     log_all=False)  # NOTE:不再存储所有的epoch

            t1 = time.time()
            for i in range(cfg['epochs']):
                print(f"Training progress: {i} / {cfg['epochs']}")
                train_loss_iterated = self.train_epoch(self.model, train_loader, optimizer, scheduler, loss_func,
                                                       self.device)
                print("train_loss_iterated", train_loss_iterated)  # TEST
                # mse_train, nse_train is not need to be calculated (via eval_model function),
                # and you can comment the next line to speed up
                # mse_train, nse_train = '', ''
                # if self.cfg['sub_model'] == 'var_plus':
                #     mse_train, nse_train = self.vali(self.model, train_loader, self.device)  # NOTE:注释后可以加速训练
                # else:
                mse_train, nse_train = self.vali_train(self.model, train_loader, self.device)  # NOTE:注释后可以加速训练
                if basin not in ['02125000', '06154410', '06291500', '12383500', '12388400']:
                    mse_val, nse_val = self.vali(self.model, vali_loader, self.device)
                else:
                    mse_val, nse_val = mse_train, nse_train
                # mse_val, nse_val = self.vali_train(self.model, vali_loader, self.device)

                print(f"train mse:{mse_train},train nse: {nse_train}\n"
                      f"val mse:{mse_val},val nse: {nse_val}\n"
                      f"------------------------------------")

                # mse_val, nse_val = mse_val[0], nse_val[0]  # TEST: loss分开算了，这里选径流最好
                with open(log_file, "at") as f:
                    f.write(f"{i},{train_loss_iterated},{mse_train},{mse_val},{nse_train},{nse_val}\n")
                mse_log.update(self.model, mse_val[0], i)
                nse_log.update(self.model, nse_val[0], i)
                nse_log2.update(self.model, nse_train[0], i)
                epoch_log.update(self.model, i, i)

                # 最后一个epoch简易验证测试集
                if i == (cfg['epochs'] - 1):
                    mse_test, nse_test = self.vali(self.model, test_loader, self.device)
                    print("last epochs:")
                    print(f"test mse:{mse_test},test nse: {nse_test}\n"
                          f"------------------------------------")

            t2 = time.time()
            print(f"Training used time:{t2 - t1}")

        return self.model
