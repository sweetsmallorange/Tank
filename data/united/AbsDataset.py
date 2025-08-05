import numpy as np


class AbsDataset:
    """AbsDataset
    1. calculate mean and standard deviation.
    2. normalization / inverse normalization.
    """

    @staticmethod
    def calc_dict_mean_and_std(data_dict: dict):
        data_all = np.concatenate(list(data_dict.values()), axis=0)  # CAN NOT serializable
        nan_mean = np.nanmean(data_all, axis=0).astype("float32")
        nan_std = np.nanstd(data_all, axis=0).astype("float32")
        return nan_mean, nan_std

    @staticmethod
    def calc_array_mean_and_std(data_all: np.ndarray):
        nan_mean = np.nanmean(data_all, axis=0).astype("float32")
        nan_std = np.nanstd(data_all, axis=0).astype("float32")
        return nan_mean, nan_std

    @staticmethod
    def calc_dict_zero_mean_and_std(data_dict: dict):
        data_all = np.concatenate(list(data_dict.values()), axis=0)
        nan_mean = np.nanmean(data_all, axis=0).astype("float32")
        nan_mean=np.zeros_like(nan_mean).astype("float32")
        nan_std = np.nanstd(data_all, axis=0).astype("float32")
        nan_std = np.zeros_like(nan_std).astype("float32")
        return nan_mean, nan_std
