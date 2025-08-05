import yaml
from pathlib import Path

from data.united.AbsDataset import AbsDataset
from utils.timefeatures import time_features


def init_dss_cfg(find_root, used_ds):
    dss_cfg = dict()

    selected_yml_paths = list()
    for ds in used_ds:
        selected_yml_paths.append(list(Path(find_root).glob(f"[[]{ds}[]]Selected*.yml"))[0])
    selected_yml_paths = sorted(selected_yml_paths, key=lambda x: x.name)
    for selected_yml_path in selected_yml_paths:
        f_selected = open(selected_yml_path, "rb")
        left = str(selected_yml_path).find("[")
        right = str(selected_yml_path).find("]")
        dataset_name = str(selected_yml_path)[left + 1:right]
        f_split = open(f"{find_root}/[{dataset_name}]Split.yml", "rb")
        yaml_selected = yaml.load(f_selected, Loader=yaml.FullLoader)
        yaml_split = yaml.load(f_split, Loader=yaml.FullLoader)
        dss_cfg[dataset_name] = {
            "basins": yaml_selected["basins"],
            "start_date": yaml_selected["start_date"],
            "end_date": yaml_selected["end_date"],
            "train_start": yaml_split["train_start"],
            "train_end": yaml_split["train_end"],
            "val_start": yaml_split["val_start"],
            "val_end": yaml_split["val_end"],
            "test_start": yaml_split["test_start"],
            "test_end": yaml_split["test_end"]
        }
        f_split.close()
        f_selected.close()

    return dss_cfg


# import global_variables
import numpy as np
import pandas as pd
import torch
from pathlib import Path


# from src.utils.config_injector.ConfigInjector import ConfigInjector
# from utils.tools import get_hash_code


class ForcingRunoffDataset:
    @staticmethod
    def fill_nan(df: pd.DataFrame):
        # df = df.interpolate(method="spline", order=3).fillna(value=df.mean(axis=0))
        df = df.interpolate(method="linear").fillna(value=df.mean(axis=0))
        return df

    @staticmethod
    def get_missing_rate(df):
        missing_rate = df.isnull().sum(axis=0).sum(axis=0) / (len(df) * len(df.columns))
        return missing_rate

    @staticmethod
    def align_forcing_columns(df, full_columns) -> [pd.DataFrame, np.ndarray]:
        align_flag = list()
        for c in full_columns:
            if c not in df.columns:
                align_flag.append(float("-inf"))
                df.loc[:, c] = np.nan
            # elif df[c].isnull().sum(0) / len(df[c]) > 0.2:
            #     # TODO: 缺失率过大的列，当作该列不存在
            #     print(f"A lot of data is missing in feature {c}, thus we treat it as empty feature.")
            #     align_flag.append(float("-inf"))
            #     df.loc[:, c] = np.nan
            else:
                align_flag.append(0.0)

        df = df[full_columns]
        align_flag = np.array(align_flag).astype("float32")
        return df, align_flag

    @classmethod
    def load_forcing(cls, forcing_root: Path, basins: list, start_date: str, end_date: str, full_columns: list):
        forcing = dict()
        forcing_align_flags = dict()
        dataset_name = forcing_root.parent.name
        for basin in basins:
            full_index = dataset_name + "_" + basin
            file = forcing_root / f"{basin}.csv"
            df = pd.read_csv(file, sep=",", header=0)
            df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
            df = df.set_index("date")
            # Only load selected dates
            df = df[start_date:end_date]
            # Fill nan by interpolation
            print(
                f"{full_index}: {cls.get_missing_rate(df) * 100:.6f}% (forcing) data is missing, will be interpolated.")
            df = cls.fill_nan(df)
            # Align forcing columns (filled with nan in empty features)
            aligned_df, align_flag = cls.align_forcing_columns(df, full_columns)
            forcing[full_index] = aligned_df
            forcing_align_flags[full_index] = align_flag
        return forcing, forcing_align_flags

    @classmethod
    def load_runoff(cls, runoff_root: Path, basins: list, start_date: str, end_date: str):
        runoff = dict()
        dataset_name = runoff_root.parent.name
        for basin in basins:
            full_index = dataset_name + "_" + basin
            file = runoff_root / f"{basin}.csv"
            df = pd.read_csv(file, sep=",", header=0)
            df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
            df = df.set_index("date")
            # Only load selected dates
            df = df[start_date:end_date]
            # Fill nan by interpolation
            print(
                f"{full_index}: {cls.get_missing_rate(df) * 100:.6f}% (runoff) data is missing, will be interpolated.")
            df = cls.fill_nan(df)
            runoff[full_index] = df
        return runoff

    @staticmethod
    def load_static(static_root: Path, basins: list):
        static = dict()
        dataset_name = static_root.parent.name
        runoff_file = static_root / "static_attributes.csv"
        df = pd.read_csv(runoff_file, sep=",", header=0, dtype={0: str})
        df = df.set_index("basin")
        for (idx, row) in df.iterrows():
            if idx in basins:
                static[dataset_name + "_" + idx] = pd.DataFrame(row).T
        return static

    def set_hash_code(self):
        self.hash_code = get_hash_code(self.dss_cfg) + get_hash_code(self.cfg)

    def load_data(self):
        # print(self.dss_cfg)
        camels_root = self.camels_root
        for dataset_name in self.dss_cfg:
            dataset_root = camels_root / dataset_name
            forcing_root = dataset_root / "forcing"
            runoff_root = dataset_root / "runoff"
            static_root = dataset_root / "static"
            basins = self.dss_cfg[dataset_name]["basins"]
            start_date = self.dss_cfg[dataset_name]["start_date"]
            end_date = self.dss_cfg[dataset_name]["end_date"]
            forcing, forcing_align_flags = self.load_forcing(forcing_root, basins, start_date, end_date,
                                                             self.full_columns)
            self.forcing_dict.update(forcing)
            self.forcing_align_dict.update(forcing_align_flags)
            self.runoff_dict.update(self.load_runoff(runoff_root, basins, start_date, end_date))
            self.static_dict.update(self.load_static(static_root, basins))
            print(f"=============== Dataset: {dataset_name} loaded complete. ===============")

    def split_data(self):
        for dataset_name in self.dss_cfg:
            train_start = pd.to_datetime(self.dss_cfg[dataset_name]["train_start"], format="%Y-%m-%d")
            train_end = pd.to_datetime(self.dss_cfg[dataset_name]["train_end"], format="%Y-%m-%d")
            val_start = pd.to_datetime(self.dss_cfg[dataset_name]["val_start"], format="%Y-%m-%d")
            val_end = pd.to_datetime(self.dss_cfg[dataset_name]["val_end"], format="%Y-%m-%d")
            test_start = pd.to_datetime(self.dss_cfg[dataset_name]["test_start"], format="%Y-%m-%d")
            test_end = pd.to_datetime(self.dss_cfg[dataset_name]["test_end"], format="%Y-%m-%d")
            for full_index in self.forcing_dict:
                if full_index.startswith(dataset_name):
                    self.start_dates_dict[full_index] = [train_start, val_start, test_start]
                    self.end_dates_dict[full_index] = [train_end, val_end, test_end]
                    temp = self.forcing_dict[full_index]

                    if self.use_static:
                        static_feature = self.static_dict[full_index]
                        rep_static_features = pd.DataFrame(np.repeat(static_feature.values, temp.shape[0], axis=0))
                        rep_static_features.columns = static_feature.columns
                        rep_static_features.index = temp.index
                        temp = pd.concat([temp, rep_static_features], axis=1)

                    if self.use_runoff:
                        train_start2 = train_start - pd.DateOffset(days=1) - pd.DateOffset(days=self.past_len - 1)
                        train_end2 = train_end - pd.DateOffset(days=1)
                        val_start2 = val_start - pd.DateOffset(days=1) - pd.DateOffset(days=self.past_len - 1)
                        val_end2 = val_end - pd.DateOffset(days=1)
                        test_start2 = test_start - pd.DateOffset(days=1) - pd.DateOffset(days=self.past_len - 1)
                        test_end2 = test_end - pd.DateOffset(days=1)

                        runoff = self.runoff_dict[full_index]
                        train_runoff = runoff.loc[train_start2: train_end2]
                        val_runoff = runoff.loc[val_start2: val_end2]
                        test_runoff = runoff.loc[test_start2: test_end2]

                        train_start = train_runoff.index[0] + pd.DateOffset(days=1)
                        train_end = train_runoff.index[-1] + pd.DateOffset(days=1)
                        val_start = val_runoff.index[0] + pd.DateOffset(days=1)
                        val_end = val_runoff.index[-1] + pd.DateOffset(days=1)
                        test_start = test_runoff.index[0] + pd.DateOffset(days=1)
                        test_end = test_runoff.index[-1] + pd.DateOffset(days=1)

                        self.forcing_train[full_index] = temp.loc[train_start:train_end]
                        self.forcing_train[full_index] = pd.concat(
                            [self.forcing_train[full_index].reset_index(drop=True),
                             train_runoff.reset_index(drop=True)],
                            axis=1).values.astype("float32")

                        self.forcing_val[full_index] = temp.loc[val_start:val_end]
                        self.forcing_val[full_index] = pd.concat(
                            [self.forcing_val[full_index].reset_index(drop=True), val_runoff.reset_index(drop=True)],
                            axis=1).values.astype("float32")

                        self.forcing_test[full_index] = temp.loc[test_start:test_end]
                        self.forcing_test[full_index] = pd.concat(
                            [self.forcing_test[full_index].reset_index(drop=True), test_runoff.reset_index(drop=True)],
                            axis=1).values.astype("float32")

                        # temp = self.runoff_dict[full_index]
                        self.runoff_train[full_index] = pd.concat((runoff.loc[train_start:train_end],
                                                                   temp.loc[train_start:train_end,
                                                                   'pet']), axis=1).values.astype("float32")
                        self.runoff_val[full_index] = runoff.loc[val_start:val_end].values.astype("float32")
                        self.runoff_test[full_index] = runoff.loc[test_start:test_end].values.astype("float32")

                    else:
                        self.forcing_train[full_index] = temp.loc[train_start:train_end].values.astype("float32")
                        self.forcing_val[full_index] = temp.loc[val_start:val_end].values.astype("float32")
                        self.forcing_test[full_index] = temp.loc[test_start:test_end].values.astype("float32")

                        temp = self.runoff_dict[full_index]
                        self.runoff_train[full_index] = temp.loc[train_start:train_end].values.astype("float32")
                        self.runoff_val[full_index] = temp.loc[val_start:val_end].values.astype("float32")
                        self.runoff_test[full_index] = temp.loc[test_start:test_end].values.astype("float32")

                    # time_feature部分
                    self.date_index_train[full_index] = time_features(temp.loc[train_start:train_end].index,
                                                                      freq=self.freq).transpose(1, 0).astype("float32")

                    self.date_index_val[full_index] = time_features(temp.loc[val_start:val_end].index,
                                                                    freq=self.freq).transpose(1, 0).astype("float32")

                    self.date_index_test[full_index] = time_features(temp.loc[test_start:test_end].index,
                                                                     freq=self.freq).transpose(1, 0).astype("float32")

            print(f"=============== Dataset: {dataset_name} split complete. ===============")

    def __init__(self, dss_cfg, cfg, x_mean=None, y_mean=None, x_std=None, y_std=None, y_stds_dict=None):
        # print(self.dss_cfg)
        self.cfg = cfg
        self.past_len = cfg["past_len"]
        self.pred_len = cfg["pred_len"]
        self.camels_root = cfg["camels_root"]
        self.data_dir = cfg["data_dir"]
        self.freq = cfg["freq"]
        self.use_runoff = cfg["use_runoff"]
        self.dynamic_size = cfg["src_size"]
        self.static_size = cfg["static_size"]
        self.use_static = True
        self.full_columns = ["prcp", "pet", "srad", "tmax", "tmin", "vprp", "aet"]
        self.dss_cfg = dss_cfg
        self.hash_code = None

        # intermediate data
        self.forcing_dict = dict()
        self.forcing_align_dict = dict()
        self.runoff_dict = dict()
        self.static_dict = dict()

        # split data
        self.start_dates_dict = dict()  # [train_start, val_start, test_start]
        self.end_dates_dict = dict()  # [train_end, val_end, test_end]
        self.forcing_train = dict()
        self.forcing_val = dict()
        self.forcing_test = dict()
        self.runoff_train = dict()
        self.runoff_val = dict()
        self.runoff_test = dict()
        self.date_index_train = dict()
        self.date_index_val = dict()
        self.date_index_test = dict()

    @classmethod
    def get_dataset(cls, dss_cfg, cfg):
        instance = cls(dss_cfg, cfg)
        # instance.set_hash_code()
        # cache_path = Path(instance.data_dir) / "cache" / f"{instance.hash_code}_serialized.pkl"
        # if cache_path.exists():
        #     print(f"Use cached dataset in: {cache_path}.")
        #     instance = torch.load(cache_path)
        #     return instance
        instance.load_data()
        instance.split_data()
        instance.forcing_dict = "Duty done, set as String, for the sake of saving memory."
        instance.runoff_dict = "Duty done, set as String, for the sake of saving memory."
        instance.static_dict = "Duty done, set as String, for the sake of saving memory."
        # cache_path.parent.mkdir(exist_ok=True, parents=True)
        # torch.save(instance, cache_path)
        return instance


class SeparatedSerializedDatasetFactory(AbsDataset):
    kernel = None

    @classmethod
    def get_datasets(cls, dss_cfg, cfg, norm_separated,
                     forcing_mean=None, runoff_mean=None,
                     forcing_std=None, runoff_std=None):
        """Initialization

        :param dss_cfg,
        :param cfg,
        :param norm_separated: the mean and standard deviation for normalization are obtained from its own training set
        :param forcing_mean: should be provided if norm_separated == False.
        :param runoff_mean: should be provided if norm_separated == False.
        :param forcing_std: should be provided if norm_separated == False.
        :param runoff_std: should be provided if norm_separated == False.
        """
        cls.kernel = ForcingRunoffDataset.get_dataset(dss_cfg, cfg)
        datasets_train = dict()
        datasets_val = dict()
        datasets_test = dict()
        forcing_align_dict = cls.kernel.forcing_align_dict


cfg2 = {
    "stage": "train",
    "model_id": "SAC_VAR(plus_mean_std)_loss(1-0,ng)_date(compare)_encoder(2)_softmaxAhead_normal(2,8nomask+9)_100epochs_bz1024",
    "model": "Sac",
    "sub_model": "var_plus",
    "seed": 1234,
    "run_dir": "/home/zhuwu/Deepening/Sac/runs_671_new/run_SAC_VAR(plus_mean_std)_loss(1-0,ng)_date(compare)_encoder(2)_softmaxAhead_normal(2,8nomask+9)_100epochs_bz1024_[671basins,daymet]_[Sac,varFalse,batch_size1024]_[epochs100]_[60,1]_[0.001,warmUpTrue]_[nse_allFalse]_[dp0.1]_seed1234_202302011622",
    "finetune_dir": "",
    "sh_file": "/home/zhuwu/Deepening/Sac/scripts/sac/var_plus_basins_train.sh",
    "learning_rate": 0.001,
    "dropout": 0.1,
    "epochs": 100,
    "past_len": 60,
    "pred_len": 1,
    "static_size": 27,
    "batch_size": 1024,
    "n_heads": 4,
    "d_model": 64,
    "camels_root": Path("/data2/zw/dataset/Uniform-CAMELS"),
    "forcing_type": "daymet",
    "basins_list_path": "/home/zhuwu/Dataset/CAMELS/list/671basins_list.txt",
    "freq": "d",
    "drop_last": False,
    "use_runoff": True,
    "num_workers": 12,
    "loss": "nse",
    "loss_all": False,
    "use_var": False,
    "warm_up": True,
    "data_path": "",
    "use_gpu": True,
    "gpu": 0,
    "use_multi_gpu": False,
    "devices": "0,1,2,3",
    "tank_nums": 4,
    "d_ff": 256,
    "hidden_size": 256,
    "src_size": 5,
    "tgt_size": 2,
    "train_start": "1980-10-01",
    "train_end": "1995-09-30",
    "val_start": "2010-10-01",
    "val_end": "2014-09-30",
    "test_start": "1995-10-01",
    "test_end": "2010-09-30",
    "device": "cuda:0",
    "data_dir": Path("/data2/zw/sac")
}

dss_cfg2 = init_dss_cfg("/data2/zw/sac/data/src_configs_repo/noUS", ["CAMELS-AUS"])
# dss_cfg2
ForcingRunoffDataset.get_dataset(dss_cfg2, cfg2)
# SeparatedSerializedDatasetFactory.get_datasets(dss_cfg2, cfg2, False)
