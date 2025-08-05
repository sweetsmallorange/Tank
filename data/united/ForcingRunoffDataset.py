# import global_variables
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from utils.timefeatures import time_features
# from src.utils.config_injector.ConfigInjector import ConfigInjector
from utils.tools import get_hash_code
import os

# pkl文件在linux上面能够打开，放在window上不能打开了，所以要转化一下
# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath
class DaymetHydroReader():
    camels_root = None  # needs to be set in class method "init_root"
    forcing_root = None
    discharge_root = None
    # forcing_cols = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
    features = ["PRCP(mm/day)", "SRAD(W/m2)", "Tmax(C)", "Tmin(C)", "Vp(Pa)"]
    discharge_cols = ["basin", "Year", "Mnth", "Day", "QObs", "flag"]
    target = ["QObs(mm/d)"]

    @classmethod
    def init_root(cls, camels_root):  # often be rewritten
        cls.camels_root = Path(camels_root)
        cls.forcing_root = cls.camels_root / 'basin_timeseries_v1p2_metForcing_obsFlow' / 'basin_dataset_public_v1p2' / 'basin_mean_forcing' / 'maurer'
        cls.discharge_root = cls.camels_root / 'basin_timeseries_v1p2_metForcing_obsFlow' / 'basin_dataset_public_v1p2' / 'usgs_streamflow'
        cls.pet_root = cls.camels_root / 'pet_harg' / 'maurer'

    def __init__(self, basin: str):
        self.basin = basin
        self.area = None
        df = self._load_data()
        # df = self._process_invalid_data(df)  # TODO 处理nan值，后续测试需要注释掉
        self.df_x = df.iloc[:, :-1]  # Datetime as index
        self.df_y = df.iloc[:, -1:]  # Datetime as index

    def get_df_x(self):
        return self.df_x

    def get_df_y(self):
        return self.df_y

    def _load_data(self):
        df_forcing = self._load_forcing()
        # df_forcing['tmean'] = (df_forcing.iloc[:, 2] + df_forcing.iloc[:, 3]) / 2
        df_discharge = self._load_discharge()
        df_pet = self._load_pet()
        df_forcing["PET"] = df_pet
        df = pd.concat([df_forcing, df_discharge], axis=1)

        return df

    def _load_pet(self):
        files = list(self.pet_root.glob(f"%d.csv" % (self.basin)))
        if len(files) == 0:
            raise RuntimeError(f"No forcing file found for Basin {self.basin}")
        elif len(files) >= 2:
            raise RuntimeError(f"Redundant forcing files found for Basin {self.basin}")
        else:
            file_path = files[0]

        df = pd.read_csv(file_path)
        df.index = pd.to_datetime(df.iloc[:, 0], format="%Y-%m-%d")
        df = df.drop(columns=df.columns[0])
        # df.index = df.index.strftime("%Y/%m/%d")


        return df

    # Loading meteorological data
    def _load_forcing(self):
        # files = list(self.forcing_root.glob(f"**/{self.basin}_*.txt"))
        # 根据你所使用的数据选择tempS
        # if dataset == 'daymet':
        #     tempS = 'cida'
        # elif dataset == 'nldas_extended':
        #     tempS = 'nldas'
        # elif dataset == 'maurer_extended':
        #     tempS = 'maurer'
        tempS = 'maurer'
        files = list(self.forcing_root.glob(f"**/%08d_lump_%s_forcing_leap.txt" % (self.basin, tempS)))
        if len(files) == 0:
            raise RuntimeError(f"No forcing file found for Basin {self.basin}")
        elif len(files) >= 2:
            raise RuntimeError(f"Redundant forcing files found for Basin {self.basin}")
        else:
            file_path = files[0]

        # read-in data and convert date to datetime index
        df = pd.read_csv(file_path, sep=r"\s+", header=3)  # \s+ means matching any whitespace character
        dates = df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str)
        df.index = pd.to_datetime(dates, format="%Y/%m/%d")

        # Line 2 (starting at 0) of the file is the area value
        with open(file_path) as fp:
            # readline is faster than readines, if only read two lines
            fp.readline()
            fp.readline()
            content = fp.readline().strip()
            area = int(content)
        self.area = area

        return df[self.features]

    # Loading runoff data
    def _load_discharge(self):
        # files = list(self.discharge_root.glob(f"**/{self.basin}_*.txt"))
        files = list(self.discharge_root.glob(f"**/%08d_streamflow_qc.txt" % (self.basin)))
        if len(files) == 0:
            raise RuntimeError(f"No discharge file found for Basin {self.basin}")
        elif len(files) >= 2:
            raise RuntimeError(f"Redundant discharge files found for Basin {self.basin}")
        else:
            file_path = files[0]

        df = pd.read_csv(file_path, sep=r"\s+", header=None, names=self.discharge_cols)
        dates = df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str)
        df.index = pd.to_datetime(dates, format="%Y/%m/%d")

        # normalize discharge from cubic feed per second to mm per day
        assert len(self.target) == 1
        df[self.target[0]] = 28316846.592 * df["QObs"] * 86400 / (self.area * 10 ** 6)

        return df[self.target]

    # Processing invalid data
    def _process_invalid_data(self, df: pd.DataFrame):
        # Delete all row, where exits NaN (only discharge has NaN in this dataset)
        # len_raw = len(df)
        # df = df.dropna()
        # len_drop_nan = len(df)
        # if len_raw > len_drop_nan:
        #     print(f"Deleted {len_raw - len_drop_nan} records because of NaNs {self.basin}")
        #
        # # Deletes all records, where no discharge was measured (-999)
        # df = df.drop((df[df['QObs(mm/d)'] < 0]).index)
        # len_drop_neg = len(df)
        # if len_drop_nan > len_drop_neg:
        #     print(f"Deleted {len_drop_nan - len_drop_neg} records because of negative discharge {self.basin}")

        # Update NaN to 0
        df.fillna(0, inplace=True)

        return df



class ForcingRunoffDataset:
    # 线性插值
    @staticmethod
    def fill_nan(df: pd.DataFrame):
        # df = df.interpolate(method="spline", order=3).fillna(value=df.mean(axis=0))
        df = df.interpolate(method="linear").fillna(value=df.mean(axis=0))
        return df
    # 计算缺失率
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
                df.loc[:, c] = np.nan  # origin
                # df.loc[:, c] = 0.0
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

    @staticmethod
    def get_hydro_readerForCamels(camels_root, basin):
        DaymetHydroReader.init_root(camels_root)
        reader = DaymetHydroReader(basin)
        return reader

    @classmethod
    def load_forcingAndRunoff(cls, camels_root: Path, basins: list, start_date: str, end_date: str, full_columns: list):
        forcing = dict()
        runoff = dict()
        forcing_align_flags = dict()
        dataset_name = "maurer"
        for basin in basins:
            full_index = dataset_name + "_" + str(basin)
            reader = cls.get_hydro_readerForCamels(camels_root, basin)
            df_x = reader.get_df_x()
            df_y = reader.get_df_y()

            df_x = df_x[start_date:end_date]
            df_y = df_y[start_date:end_date]
            # Fill nan by interpolation
            print(
                f"{full_index}: {cls.get_missing_rate(df_x) * 100:.6f}% (forcing) data is missing, will be interpolated.")
            print(
                f"{full_index}: {cls.get_missing_rate(df_y) * 100:.6f}% (runoff) data is missing, will be interpolated.")

            # 前面已经使用函数进行nan值去除，这边无需继续fill_nan，加了也无所谓
            #使用线性插值的方法填充 DataFrame 中的缺失值
            df_x = cls.fill_nan(df_x)
            df_y = cls.fill_nan(df_y)

            # Align forcing columns (filled with nan in empty features)
            # 实际作用就只是新增了aet列，并且全部的值为nan
            # 如果列需要添加到 df 中，则align_flag中对应位置为 -inf，如果df中的列已经存在，align_flag则对应位置为 0.0
            aligned_df, align_flag = cls.align_forcing_columns(df_x, full_columns)
            forcing[full_index] = aligned_df
            runoff[full_index] = df_y
            forcing_align_flags[full_index] = align_flag
        return forcing, forcing_align_flags, runoff


    @staticmethod
    def load_static(static_root: Path, basins: list):
        # load static data  27个静态属性
        # attrnewLst = ['p_mean', 'pet_mean', 'p_seasonality', 'frac_snow', 'aridity', 'high_prec_freq', 'high_prec_dur',
        #               'low_prec_freq', 'low_prec_dur', 'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest',
        #               'lai_max', 'lai_diff', 'gvf_max', 'gvf_diff', 'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity',
        #               'soil_conductivity','max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'carbonate_rocks_frac',
        #               'geol_permeability']
        # 论文对比 需要9个静态属性
        attrnewLst = ['p_mean', 'frac_snow', 'aridity', 'high_prec_freq', 'high_prec_dur',
                      'elev_mean', 'slope_mean', 'area_gages2', 'geol_permeability']
        keyLst = ['topo', 'clim', 'hydro', 'vege', 'soil', 'geol']
        static_file_path = static_root / 'camels_attributes_v2.0' / 'camels_attributes_v2.0'
        static = dict()

        static_df = pd.DataFrame()
        static_df['gauge_id'] = basins

        for key in keyLst:
            dataFile = os.path.join(static_file_path, 'camels_' + key + '.txt')
            dataTemp = pd.read_csv(dataFile, sep=';')
            varLstTemp = list(dataTemp.columns[1:])

            for field in varLstTemp:
                # if pd.api.types.is_string_dtype(dataTemp[field]):
                if dataTemp[field].apply(lambda x: isinstance(x, str)).any():
                    value, ref = pd.factorize(dataTemp[field], sort=True)
                    dataTemp[field] = value


            static_df = pd.merge(static_df, dataTemp, on='gauge_id', how='outer')
        static_df.rename(columns={'gauge_id': 'basin'}, inplace=True)
        static_df = static_df.set_index("basin")
        static_df = static_df[attrnewLst]

        dataset_name = "maurer"
        # turn list to dict
        for index, row in static_df.iterrows():
            static[dataset_name + "_" + str(index)] = pd.DataFrame(row).T

        return static

    def set_hash_code(self):
        self.hash_code = get_hash_code(self.dss_cfg) + "_" + get_hash_code(self.cfg)

    def load_data(self):
        # print(self.dss_cfg)
        camels_root = self.camels_root
        # tank数据集仅针对maurer
        dataset_name = "maurer"
        basins = self.dss_cfg["basins"]
        start_date = self.dss_cfg["start_date"]
        end_date = self.dss_cfg["end_date"]


        forcing, forcing_align_flags, runoff = self.load_forcingAndRunoff(camels_root, basins, start_date, end_date,
                                                         self.full_columns)

        # runoff = self.load_runoff(runoff_root, basins, start_date, end_date)
        self.forcing_dict.update(forcing)
        self.forcing_align_dict.update(forcing_align_flags)
        self.runoff_dict.update(runoff)
        if self.use_static:
            static = self.load_static(camels_root, basins)
            self.static_dict.update(static)
            print(f"=============== Dataset: {dataset_name} loaded complete. ===============")

    def split_data(self):
        # for dataset_name in self.cfg:
        train_start = pd.to_datetime(self.cfg["train_start"], format="%Y-%m-%d")
        train_end = pd.to_datetime(self.cfg["train_end"], format="%Y-%m-%d")
        val_start = pd.to_datetime(self.cfg["val_start"], format="%Y-%m-%d")
        val_end = pd.to_datetime(self.cfg["val_end"], format="%Y-%m-%d")
        test_start = pd.to_datetime(self.cfg["test_start"], format="%Y-%m-%d")
        test_end = pd.to_datetime(self.cfg["test_end"], format="%Y-%m-%d")
        for full_index in self.forcing_dict:
            # if full_index.startswith(dataset_name):
            self.start_dates_dict[full_index] = [train_start, val_start, test_start]
            self.end_dates_dict[full_index] = [train_end, val_end, test_end]

            # temp 将forcing、static、runoff全部合在一起了
            temp = self.forcing_dict[full_index]

            if self.use_static:
                static_feature = self.static_dict[full_index]
                rep_static_features = pd.DataFrame(np.repeat(static_feature.values, temp.shape[0], axis=0))
                rep_static_features.columns = static_feature.columns
                rep_static_features.index = temp.index
                temp = pd.concat([temp, rep_static_features], axis=1)

            #
            if self.use_runoff:
                train_start2 = train_start - pd.DateOffset(days=self.pred_len) - pd.DateOffset(
                    days=self.past_len - self.pred_len)
                train_end2 = train_end - pd.DateOffset(days=self.pred_len)
                val_start2 = val_start - pd.DateOffset(days=self.pred_len) - pd.DateOffset(
                    days=self.past_len - self.pred_len)
                val_end2 = val_end - pd.DateOffset(days=self.pred_len)
                test_start2 = test_start - pd.DateOffset(days=self.pred_len) - pd.DateOffset(
                    days=self.past_len - self.pred_len)
                test_end2 = test_end - pd.DateOffset(days=self.pred_len)

                runoff = self.runoff_dict[full_index]
                # TEST:yr
                if (np.isnan(runoff).any()[0]):
                    print("nan in split_data")
                train_runoff = runoff.loc[train_start2: train_end2]
                val_runoff = runoff.loc[val_start2: val_end2]
                test_runoff = runoff.loc[test_start2: test_end2]

                train_start = train_runoff.index[0] + pd.DateOffset(days=self.pred_len)
                train_end = train_runoff.index[-1] + pd.DateOffset(days=self.pred_len)
                val_start = val_runoff.index[0] + pd.DateOffset(days=self.pred_len)
                val_end = val_runoff.index[-1] + pd.DateOffset(days=self.pred_len)
                test_start = test_runoff.index[0] + pd.DateOffset(days=self.pred_len)
                test_end = test_runoff.index[-1] + pd.DateOffset(days=self.pred_len)

                self.start_dates_dict[full_index] = [train_start, val_start, test_start]
                self.end_dates_dict[full_index] = [train_end, val_end, test_end]

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
                # 将径流runoff（tci）和pet蒸发合并在一起
                self.runoff_train[full_index] = pd.concat((runoff.loc[train_start:train_end],
                                                           temp.loc[train_start:train_end, 'PET']), # NOTE:camels
                                                           # temp.loc[train_start:train_end, 'evp']),  # NOTE:yr
                                                          axis=1).values.astype("float32")
                self.runoff_val[full_index] = pd.concat((runoff.loc[val_start:val_end],
                                                         temp.loc[val_start:val_end, 'PET'],),  # NOTE:camels
                                                         # temp.loc[val_start:val_end, 'evp'],),  # NOTE:yr
                                                        axis=1).values.astype("float32")
                self.runoff_test[full_index] = pd.concat((runoff.loc[test_start:test_end],
                                                          temp.loc[test_start:test_end, 'PET'],),  # NOTE:camels
                                                          # temp.loc[test_start:test_end, 'evp'],),  # NOTE:yr
                                                         axis=1).values.astype("float32")
                # self.runoff_train[full_index] = pd.concat((runoff.loc[train_start:train_end],
                #                                           temp.loc[train_start:train_end, 'evp']),  # NOTE:yr-dropna
                #                                          axis=1).dropna().values.astype("float32")
                # self.runoff_val[full_index] = pd.concat((runoff.loc[val_start:val_end],
                #                                          temp.loc[val_start:val_end, 'evp'],),  # NOTE:yr-dropna
                #                                         axis=1).dropna().values.astype("float32")
                # self.runoff_test[full_index] = pd.concat((runoff.loc[test_start:test_end],
                #                                           temp.loc[test_start:test_end, 'evp'],),  # NOTE:yr-dropna
                #                                          axis=1).dropna().values.astype("float32")
            else:
                self.forcing_train[full_index] = temp.loc[train_start:train_end].values.astype("float32")
                self.forcing_val[full_index] = temp.loc[val_start:val_end].values.astype("float32")
                self.forcing_test[full_index] = temp.loc[test_start:test_end].values.astype("float32")

                # origin
                temp = self.forcing_dict[full_index]
                # self.runoff_train[full_index] = temp.loc[train_start:train_end].values.astype("float32")
                # self.runoff_val[full_index] = temp.loc[val_start:val_end].values.astype("float32")
                # self.runoff_test[full_index] = temp.loc[test_start:test_end].values.astype("float32")
                # origin
                # change
                runoff = self.runoff_dict[full_index]
                self.runoff_train[full_index] = pd.concat((runoff.loc[train_start:train_end],
                                                           temp.loc[train_start:train_end, 'PET']),  # NOTE:camels
                                                           # temp.loc[train_start:train_end, 'evp']),  # NOTE:yr
                                                          axis=1).values.astype("float32")
                self.runoff_val[full_index] = pd.concat((runoff.loc[val_start:val_end],
                                                         temp.loc[val_start:val_end, 'PET'],),  # NOTE:camels
                                                         # temp.loc[val_start:val_end, 'evp'],),  # NOTE:yr
                                                        axis=1).values.astype("float32")
                self.runoff_test[full_index] = pd.concat((runoff.loc[test_start:test_end],
                                                          temp.loc[test_start:test_end, 'PET'],),  # NOTE:camels
                                                          # temp.loc[test_start:test_end, 'evp'],),  # NOTE:yr
                                                         axis=1).values.astype("float32")
                # change

            # time_feature部分， 将时间转变为-0.5 ~ 0.5
            # self.date_index = df_y.index
            # self.date_index_dict[basin] = df_y.index  # df_x这时候已经没有datetime的index了
            self.date_stamp_train[full_index] = time_features(temp.loc[train_start:train_end].index,
                                                              freq=self.freq).transpose(1, 0).astype("float32")

            self.date_stamp_val[full_index] = time_features(temp.loc[val_start:val_end].index,
                                                            freq=self.freq).transpose(1, 0).astype("float32")

            self.date_stamp_test[full_index] = time_features(temp.loc[test_start:test_end].index,
                                                             freq=self.freq).transpose(1, 0).astype("float32")

        print(f"=============== Dataset: split complete. ===============")

    def __init__(self, dss_cfg, cfg, x_mean=None, y_mean=None, x_std=None, y_std=None, y_stds_dict=None):
        # print(self.dss_cfg)
        self.cfg = cfg
        self.past_len = cfg["past_len"]
        self.pred_len = cfg["pred_len"]
        self.camels_root = cfg["camels_root"]
        self.data_dir = cfg["data_dir"]
        self.freq = cfg["freq"]
        self.use_runoff = cfg["use_runoff"]
        self.dynamic_size = cfg["src_size"]  # origin:dynamic
        self.static_size = cfg["static_size"]
        # self.use_static = True
        # 在US-pet上面跑的话，按照论文需要使用静态属性，否则位置编码的通道将会报错
        # 将use_static的bool值加入到cfg["use_static"]中
        # self.use_static = False # NOTE:TEST:涝峪口
        self.use_static = cfg["use_static"]

        self.full_columns = ["PET", "PRCP(mm/day)", "SRAD(W/m2)", "Tmax(C)", "Tmin(C)", "Vp(Pa)"]  # NOTE:camels

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
        self.date_stamp_train = dict()
        self.date_stamp_val = dict()
        self.date_stamp_test = dict()

    @classmethod
    def get_dataset(cls, dss_cfg, cfg):
        instance = cls(dss_cfg, cfg)
        # instance.set_hash_code()
        #由于文件名称过长，在windows中无法存储下来，所以取消这边hash的计算方式
        cache_path = Path(cfg['data_dir']) / "cache" / f"{instance.hash_code}_serialized.pkl"

        if cache_path.exists():
            print(f"Use cached dataset in: {cache_path}.")
            instance = torch.load(cache_path)
            return instance
        instance.load_data()
        instance.split_data()
        instance.forcing_dict = "Duty done, set as String, for the sake of saving memory."
        instance.runoff_dict = "Duty done, set as String, for the sake of saving memory."
        instance.static_dict = "Duty done, set as String, for the sake of saving memory."
        cache_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(instance, cache_path)
        return instance
