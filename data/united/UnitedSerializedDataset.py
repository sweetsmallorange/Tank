import json
from bisect import bisect_right
from copy import copy

import numpy as np
from torch.utils.data import Dataset

from data.united.ForcingRunoffDataset import ForcingRunoffDataset
from data.united.AbsDataset import AbsDataset
from utils.ZScoreNormalization import ZScoreNormalization

class UnitedSerializedDataset(Dataset, AbsDataset):
    """UnitedSerializedDataset extends from torch.utils.data.Dataset and rely on ForcingRunoffDataset instance,
    providing service:
    1. instead of real serialization, we build index for __getitem__, which means virtual serialization.
    """

    def build_index(self, forcing, runoff, data_index):
        assert len(forcing) == len(runoff)
        for idx, full_index in enumerate(forcing):
            forcing_one = forcing[full_index]
            runoff_one = runoff[full_index]
            # runoff_one=runoff_one.dropna() #note:yr ADD new
            data_stamp_one = data_index[full_index]  # NOTE:ADD

            # z-score
            self.x_dict[full_index] = ZScoreNormalization.normalization(forcing_one, self.x_mean,
                                                                        self.x_std)
            self.data_stamp_dict[full_index] = data_stamp_one  # NOTE:ADD
            self.y_dict[full_index] = ZScoreNormalization.normalization(runoff_one, self.y_mean, self.y_std)#NOTE:camels
            # NOTE:yr TEST
            if np.isnan(runoff_one).any():
                print(f"nan in build_index, {full_index} 不符合条件")
            # self.y_dict[full_index] = runoff_one #NOTE:yr
            self.y_std_dict[full_index] = runoff_one.std(axis=0).tolist()  # ADD

            self.basins_ls.append(full_index)
            avail_len = len(forcing_one) - self.kernel.past_len - self.kernel.pred_len + 1
            self.length_ls.append(avail_len)
            self.length_dict[full_index] = avail_len

        self.index_ls = [0]
        # basin依次加在一起
        for i in range(len(self.length_ls)):
            v = self.index_ls[i] + self.length_ls[i]
            self.index_ls.append(v)
        # basin的样本数全部全部加在一起
        for length in self.length_ls:
            self.num_samples += length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        basin_idx = bisect_right(self.index_ls, idx) - 1
        local_idx = idx - self.index_ls[basin_idx]
        basin = self.basins_ls[basin_idx]
        x_seq = self.x_dict[basin][local_idx: local_idx + self.kernel.past_len + self.kernel.pred_len, :]
        x_seq_mark = self.data_stamp_dict[basin][
                     local_idx: local_idx + self.kernel.past_len + self.kernel.pred_len, :]  # ADD

        y_seq_past = self.y_dict[basin][local_idx: local_idx + self.kernel.past_len, :]
        y_seq_future = self.y_dict[basin][
                       local_idx + self.kernel.past_len: local_idx + self.kernel.past_len + self.kernel.pred_len, :]

        y_stds_basin = np.array(self.y_std_dict[basin])  # add

        return x_seq, x_seq_mark, y_seq_past, y_seq_future, self.kernel.forcing_align_dict[
            basin], y_stds_basin,  # change, add y_stds_basin

    # add
    def get_means(self):
        return self.x_mean, self.y_mean

    def get_stds(self):
        return self.x_std, self.y_std

    def local_rescale(self, feature: np.ndarray, variable: str) -> np.ndarray:
        if variable == 'inputs':
            feature = feature * self.x_std + self.x_mean
        elif variable == 'output':
            feature = feature * self.y_std + self.y_mean
        else:
            raise RuntimeError(f"Unknown variable type {variable}")
        return feature

    # add

    def __init__(self, dss_cfg, cfg, stage, x_mean=None, y_mean=None, x_std=None, y_std=None, freq='h',
                 y_stds_dict=None):
        """Initialization
        :param stage: "train", "val" or "test"
        :param x_mean: should be provided if stage != "train".
        :param y_mean: should be provided if stage != "train".
        :param x_std: should be provided if stage != "train".
        :param y_std: should be provided if stage != "train".
        """
        self.kernel = ForcingRunoffDataset.get_dataset(dss_cfg, cfg)  # 从上面移下来了
        self.stage = stage
        self.x_mean = x_mean
        self.y_mean = y_mean
        self.x_std = x_std
        self.y_std = y_std
        self.basins_ls = list()
        self.length_ls = list()
        self.length_dict = dict()
        self.num_samples = 0
        self.index_ls = None

        self.x_dict = dict()  # forcing(w/wo runoff) ---normalized---> x
        self.y_dict = dict()  # runoff,pet ---normalized---> y
        self.data_stamp_dict = dict()  # date_index ---> data_stamp # ADD
        self.y_std_dict = dict()  # y-std ADD
        if stage == "train":
            # origin:forcing_mean, forcing_std
            # 计算出来的x_mean, x_std的长度为16（forcing，static，runoff）
            self.x_mean, self.x_std = self.calc_dict_mean_and_std(self.kernel.forcing_train)
            # origin:runoff_mean, runoff_std
            # 计算出来的y_mean, y_std的长度为2（runoff，pet）
            self.y_mean, self.y_std = self.calc_dict_mean_and_std(self.kernel.runoff_train) # NOTE:camels
            # self.y_mean, self.y_std = self.calc_dict_zero_mean_and_std(self.kernel.runoff_train) # NOTE:yr
            self.build_index(self.kernel.forcing_train, self.kernel.runoff_train, self.kernel.date_stamp_train)
            # ADD
            # TEST ？？？？为什么要这么做



            fea_mask_bool = np.isnan(self.x_mean)
            self.x_mean[fea_mask_bool] = 0.0
            self.x_std[fea_mask_bool] = 1
            fea_mask_bool = np.isnan(self.y_mean)
            self.y_mean[fea_mask_bool] = 0.0
            self.y_std[fea_mask_bool] = 1


            # TEST
            # Saving training mean and training std
            np.savetxt(cfg['data_dir'] / "x_means.csv", self.x_mean)
            np.savetxt(cfg['data_dir'] / "x_stds.csv", self.x_std)
            np.savetxt(cfg['data_dir'] / "y_means.csv", self.y_mean)
            np.savetxt(cfg['data_dir'] / "y_stds.csv", self.y_std)
            with open(cfg['data_dir'] / "y_stds_dict.json", "wt") as f:  # 暂时不用
                json.dump(self.y_std_dict, f)
            # ADD
        elif stage == "val":
            self.build_index(self.kernel.forcing_val, self.kernel.runoff_val, self.kernel.date_stamp_val)
        elif stage == "test":
            self.build_index(self.kernel.forcing_test, self.kernel.runoff_test, self.kernel.date_stamp_test)
        else:
            raise RuntimeError(f"Illegal stage: {stage}")

        self.y_origin = copy(self.y_dict)  # TEST:
