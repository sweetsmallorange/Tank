from copy import copy

import numpy as np
from torch.utils.data import Dataset

from data.united.ForcingRunoffDataset import ForcingRunoffDataset
from data.united.AbsDataset import AbsDataset
from utils.ZScoreNormalization import ZScoreNormalization


class SeparatedSerializedDatasetFactory(AbsDataset):
    kernel = None  # change

    @classmethod
    def get_datasets(cls, dss_cfg, cfg, norm_separated,
                     x_mean=None, y_mean=None,  # change
                     x_std=None, y_std=None):  # change
        """Initialization

        :param dss_cfg,
        :param cfg,
        :param norm_separated: the mean and standard deviation for normalization are obtained from its own training set
        :param x_mean: should be provided if norm_separated == False.
        :param y_mean: should be provided if norm_separated == False.
        :param x_std: should be provided if norm_separated == False.
        :param y_std: should be provided if norm_separated == False.
        """
        cls.kernel = ForcingRunoffDataset.get_dataset(dss_cfg, cfg)  # add
        datasets_train = dict()
        datasets_val = dict()
        datasets_test = dict()
        forcing_align_dict = cls.kernel.forcing_align_dict
        for full_index in cls.kernel.forcing_train.keys():
            # forcing ---normalized---> x
            # runoff ---normalized---> y
            # Training data
            forcing_train = cls.kernel.forcing_train[full_index]
            runoff_train = cls.kernel.runoff_train[full_index]
            data_stamp_train = cls.kernel.date_stamp_train[full_index]
            if norm_separated:
                x_mean, x_std = cls.calc_array_mean_and_std(forcing_train)
                y_mean, y_std = cls.calc_array_mean_and_std(runoff_train)
                # TEST
                fea_mask_bool = np.isnan(x_mean)
                x_mean[fea_mask_bool] = 0.0
                x_std[fea_mask_bool] = 1
                fea_mask_bool = np.isnan(y_mean)
                y_mean[fea_mask_bool] = 0.0
                y_std[fea_mask_bool] = 1
                # TEST
                # Set the standard deviation of the static features to 1
                x_std[(0.0000001 >= x_std) & (x_std >= -0.0000001)] = 1
                # NOTE: finetune_single的时候应该需要，因为每个站点需要存自己的均值和方差 TODO 需要自己调整
                # ADD
                t_dir = cfg['data_dir'] / full_index
                if not t_dir.is_dir():
                    t_dir.mkdir(parents=True)
                np.savetxt(cfg['data_dir'] /full_index/ "x_means.csv", x_mean)
                np.savetxt(cfg['data_dir'] / full_index/"x_stds.csv", x_std)
                np.savetxt(cfg['data_dir'] / full_index/"y_means.csv", y_mean)
                np.savetxt(cfg['data_dir'] / full_index/"y_stds.csv", y_std)
                # with open(cfg['data_dir'] / "y_stds_dict.json", "wt") as f:  # 暂时不用
                #     json.dump(y_std_dict, f)
                # # ADD
            elif x_mean is None and y_mean is None and x_std is None and y_std is None:
                # NOTE：yr_s测试的时候不要注释，其他时候分清楚是不是finetune_single
                # ADD
                # 如果finetune_single用这个：  TODO fine_tune 测试的时候都需要调整
                ########################################################################################################
                x_mean=np.loadtxt(cfg['data_dir'] / full_index / "x_means.csv", dtype="float32")
                x_std=np.loadtxt(cfg['data_dir'] / full_index / "x_stds.csv", dtype="float32")
                y_mean=np.loadtxt(cfg['data_dir'] / full_index / "y_means.csv", dtype="float32")
                y_std=np.loadtxt(cfg['data_dir'] / full_index / "y_stds.csv", dtype="float32")
                ########################################################################################################
                # 如果不是，用这个：TODO fine_tune 测试的时候都需要调整
                ########################################################################################################
                # x_mean = np.loadtxt(cfg['data_dir']  / "x_means.csv", dtype="float32")
                # x_std = np.loadtxt(cfg['data_dir']  / "x_stds.csv", dtype="float32")
                # y_mean = np.loadtxt(cfg['data_dir']  / "y_means.csv", dtype="float32")
                # y_std = np.loadtxt(cfg['data_dir'] / "y_stds.csv", dtype="float32")
                ########################################################################################################
                # with open(cfg['data_dir'] / "y_stds_dict.json", "wt") as f:  # 暂时不用
                #     json.dump(y_std_dict, f)
                # ADD

            x_train = ZScoreNormalization.normalization(forcing_train, x_mean, x_std)
            y_train = ZScoreNormalization.normalization(runoff_train, y_mean, y_std)

            # Validation data
            forcing_val = cls.kernel.forcing_val[full_index]
            runoff_val = cls.kernel.runoff_val[full_index]
            data_stamp_val = cls.kernel.date_stamp_val[full_index]
            x_val = ZScoreNormalization.normalization(forcing_val, x_mean, x_std)
            y_val = ZScoreNormalization.normalization(runoff_val, y_mean, y_std)

            # Test data
            forcing_test = cls.kernel.forcing_test[full_index]
            runoff_test = cls.kernel.runoff_test[full_index]
            data_stamp_test = cls.kernel.date_stamp_test[full_index]
            x_test = ZScoreNormalization.normalization(forcing_test, x_mean, x_std)
            y_test = ZScoreNormalization.normalization(runoff_test, y_mean, y_std)

            start_dates = cls.kernel.start_dates_dict[full_index]
            end_dates = cls.kernel.end_dates_dict[full_index]

            datasets_train[full_index] = SeparatedSerializedDataset(x_train, y_train, data_stamp_train,
                                                                    forcing_align_dict[full_index],
                                                                    cls.kernel.past_len, cls.kernel.pred_len,
                                                                    start_dates, end_dates, x_mean=x_mean, x_std=x_std,
                                                                    y_mean=y_mean, y_std=y_std, y_origin=forcing_train)
            datasets_val[full_index] = SeparatedSerializedDataset(x_val, y_val, data_stamp_val,
                                                                  forcing_align_dict[full_index],
                                                                  cls.kernel.past_len, cls.kernel.pred_len,
                                                                  start_dates, end_dates, x_mean=x_mean, x_std=x_std,
                                                                  y_mean=y_mean, y_std=y_std, y_origin=forcing_val)
            datasets_test[full_index] = SeparatedSerializedDataset(x_test, y_test, data_stamp_test,
                                                                   forcing_align_dict[full_index],
                                                                   cls.kernel.past_len, cls.kernel.pred_len,
                                                                   start_dates, end_dates, x_mean=x_mean, x_std=x_std,
                                                                   y_mean=y_mean, y_std=y_std, y_origin=forcing_test)

        return datasets_train, datasets_val, datasets_test


class SeparatedSerializedDataset(Dataset):
    """

    """

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x_seq = self.x_data[idx: idx + self.past_len + self.pred_len, :]
        x_seq_mark = self.data_stamp[idx: idx + self.past_len + self.pred_len, :]  # ADD
        y_seq_past = self.y_data[idx: idx + self.past_len, :]
        y_seq_future = self.y_data[idx + self.past_len: idx + self.past_len + self.pred_len, :]
        y_std = self.y_std  # add

        return x_seq, x_seq_mark, y_seq_past, y_seq_future, self.flag_columns, y_std  # add

    def local_rescale(self, feature, variable: str):
        if variable == 'inputs':
            feature = feature * self.x_std + self.x_mean
        elif variable == 'output':
            feature = feature * self.y_std + self.y_mean
        else:
            raise RuntimeError(f"Unknown variable type {variable}")
        return feature

    def get_means(self):
        return self.x_mean, self.y_mean

    def get_stds(self):
        return self.x_std, self.y_std

    def __init__(self, x_data, y_data, data_stamp, flag_columns, past_len, pred_len, start_dates, end_dates,
                 x_mean=None, y_mean=None, x_std=None, y_std=None, y_origin=None):
        self.x_data = x_data
        self.y_data = y_data
        self.data_stamp = data_stamp  # ADD
        self.flag_columns = flag_columns
        self.past_len = past_len
        self.pred_len = pred_len
        self.start_dates = start_dates
        self.end_dates = end_dates
        self.length = len(self.x_data) - past_len - pred_len + 1

        self.x_mean = x_mean
        self.y_mean = y_mean
        self.x_std = x_std
        self.y_std = y_std

        self.y_origin = y_origin  # TEST:
