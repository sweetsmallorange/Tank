import json
from copy import copy

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
import torch

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from bisect import bisect_right
from abc import ABCMeta, abstractmethod

from data.united.ForcingRunoffDatasetConfigUtils import init_dss_cfg
from data.united.UnitedSerializedDataset import UnitedSerializedDataset
from data.united.SeparatedSerializedDataset import SeparatedSerializedDatasetFactory
from utils.timefeatures import time_features

MODELLIST = ['FusionModel', 'Hymod', 'Sac', 'S2S',
             'RR', 'RRS', 'Tank', "Xaj", 'AdTank',
             'ArTank', 'CARTank', 'SMSTank', 'Awbm',
             'LSTM', 'Gr4j', 'DeepTank_LSTM', 'OriginalTank',
             'OrdinaryTank', 'TransformableTank', 'SeriesTank',
             'SeriesTankv2']


class AbstractReader(metaclass=ABCMeta):
    """Abstract data reader.

    Its subclasses need to ensure conversion from raw data to pandas.DataFrame,
    and process invalid data item
    """

    @abstractmethod
    def _load_data(self, *args, **kwargs):
        """
        Subclasses must implement loading inputs and target data.
        """
        pass

    @abstractmethod
    def _process_invalid_data(self, *args, **kwargs):
        """
        Subclasses must implement how to process invalid data item.
        """
        pass

    @abstractmethod
    def get_df_x(self):
        """
        Subclasses must return inputs data with a form of pandas.DataFrame.
        """
        pass

    @abstractmethod
    def get_df_y(self):
        """
        Subclasses must return target data with a form of pandas.DataFrame.
        """
        pass


class AbstractStaticReader:
    """Abstract static data reader.

    1. Reads data from a static attributes file (.csv).
    2. Select used attributes and do normalization.
    3. Need to ensure conversion from static attributes to pandas.DataFrame.
    """

    @abstractmethod
    def get_df_static(self, basin):
        """
        Subclasses must return static data with a form of pandas.DataFrame for a specific basin
        """
        pass


# TODO: 需要蒸发量，蒸发量在output文件里
class DaymetHydroReader(AbstractReader):
    camels_root = None  # needs to be set in class method "init_root"
    output_root = None

    need = ["SWE", "RAIM", "TAIR", "PRCP", "PET", "ET", "OBS_RUN"]
    features = ["PRCP", "PET", "SWE", "RAIM", "TAIR"]  # TEST
    # features_runoff = [ "PRCP", "PET","OBS_RUN", "SWE", "RAIM", "TAIR"]  # TEST
    target = ["OBS_RUN", "PET"]  # TEST

    @classmethod
    def init_root(cls, camels_root):  # often be rewritten
        cls.camels_root = Path(camels_root)
        cls.output_root = cls.camels_root / "model_output_daymet" / "model_output" / "flow_timeseries" / "daymet"
        # cls.discharge_root = cls.camels_root / "usgs_streamflow"

    def __init__(self, basin: str):
        self.basin = basin
        self.area = None
        df = self._load_data()
        df = self._process_invalid_data(df)
        # if runoff == True:
        #     self.df_x = df[self.features_runoff]
        # else:
        self.df_x = df[self.features]  # Datetime as index
        # print(self.df_x.head()) #TEST
        self.df_y = df[self.target]  # Datetime as index

    def get_df_x(self):
        return self.df_x

    def get_df_y(self):
        return self.df_y

    def _load_data(self):
        df = self._load_need()

        return df

    # Loading sac data
    def _load_need(self):
        files = list(self.output_root.glob(f"**/{self.basin}_*_model_output.txt"))
        if len(files) == 0:
            raise RuntimeError(f"No discharge file found for Basin {self.basin}")
        # elif len(files) >= 2:
        #     raise RuntimeError(f"Redundant discharge files found for Basin {self.basin}")
        # else:
        #     file_path = files[0]
        df_total = pd.read_csv(files[0], sep=r"\s+")
        dates = df_total.YR.map(str) + "/" + df_total.MNTH.map(str) + "/" + df_total.DY.map(str)
        df_total.index = pd.to_datetime(dates, format="%Y/%m/%d")
        # assert len(self.need) == 3

        for i in range(1, len(files)):
            try:
                df = pd.read_csv(files[i], sep=r"\s+")
                dates = df.YR.map(str) + "/" + df.MNTH.map(str) + "/" + df.DY.map(str)
                df.index = pd.to_datetime(dates, format="%Y/%m/%d")
                # assert len(self.need) == 3
                df_total = df_total + df
            except EmptyDataError:
                print(f"No columns to parse from file {files[i]}")

        return df_total[self.need] / len(files)

    # Processing invalid data
    def _process_invalid_data(self, df: pd.DataFrame):
        # Delete all row, where exits NaN (only discharge has NaN in this dataset)
        len_raw = len(df)
        df = df.dropna()
        len_drop_nan = len(df)
        if len_raw > len_drop_nan:
            print(f"Deleted {len_raw - len_drop_nan} records because of NaNs {self.basin}")

        # Deletes all records, where no discharge was measured (-999)
        df = df.drop((df[df['OBS_RUN'] < 0]).index)
        len_drop_neg = len(df)
        if len_drop_nan > len_drop_neg:
            print(f"Deleted {len_drop_nan - len_drop_neg} records because of negative discharge {self.basin}")

        return df


# TODO: 需要蒸发量！！！
class MaurerExtHydroReader(DaymetHydroReader):
    camels_root = None  # needs to be set in class method "init_root"
    forcing_root = None
    discharge_root = None

    # need = ["SWE", "RAIM", "TAIR", "PRCP", "PET", "ET", "OBS_RUN"]
    need = ["PET", "OBS_RUN"]
    # features = ["PRCP", "PET", "SWE", "RAIM", "TAIR"]  # TEST
    # features_runoff = [ "PRCP", "PET","OBS_RUN", "SWE", "RAIM", "TAIR"]  # TEST
    forcing_cols = ["Year", "Mnth", "Day", "Hr", "dayl(s)", "prcp(mm/day)", "srad(W/m2)", "swe(mm)", "tmax(C)",
                    "tmin(C)", "vp(Pa)"]
    features = ["prcp(mm/day)", "srad(W/m2)", "tmax(C)", "tmin(C)", "vp(Pa)"]
    target = ["OBS_RUN", "PET"]  # TEST

    @classmethod
    def init_root(cls, camels_root):
        cls.camels_root = Path(camels_root)
        cls.forcing_root = cls.camels_root / "basin_mean_forcing" / "maurer_extended"
        cls.discharge_root = cls.camels_root / "usgs_streamflow"
        cls.output_root = cls.camels_root / "model_output_maurer" / "model_output" / "flow_timeseries" / "maurer"

    def _load_need(self):
        files = list(self.output_root.glob(f"**/{self.basin}_*_model_output.txt"))
        if len(files) == 0:
            raise RuntimeError(f"No discharge file found for Basin {self.basin}")
        # elif len(files) >= 2:
        #     raise RuntimeError(f"Redundant discharge files found for Basin {self.basin}")
        # else:
        #     file_path = files[0]
        df_total = pd.read_csv(files[0], sep=r"\s+")
        dates = df_total.YR.map(str) + "/" + df_total.MNTH.map(str) + "/" + df_total.DY.map(str)
        df_total.index = pd.to_datetime(dates, format="%Y/%m/%d")
        # assert len(self.need) == 3

        for i in range(1, len(files)):
            try:
                df = pd.read_csv(files[i], sep=r"\s+")
                dates = df.YR.map(str) + "/" + df.MNTH.map(str) + "/" + df.DY.map(str)
                df.index = pd.to_datetime(dates, format="%Y/%m/%d")
                # assert len(self.need) == 3
                df_total = df_total + df
            except EmptyDataError:
                print(f"No columns to parse from file {files[i]}")

        df_t = df_total[self.need] / len(files)

        files = list(self.forcing_root.glob(f"**/{self.basin}_*.txt"))
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

        df_f = df[self.features]
        df_f = df_f[df_t.index[0]:df_t.index[-1]]
        # print(df_f.index[0],df_f.index[-1],df_t.index[0],df_t.index[-1])
        df = pd.concat([df_f, df_t], axis=1)

        return df

    def __init__(self, basin: str):
        self.basin = basin
        self.area = None
        df = self._load_data()
        df = self._process_invalid_data(df)
        # if runoff == True:
        #     self.df_x = df[self.features_runoff]
        # else:
        self.df_x = df[self.features]  # Datetime as index
        # print(self.df_x)
        # print(self.df_x.head()) #TEST
        self.df_y = df[self.target]  # Datetime as index
        # print(self.df_y)


# TODO: 需要蒸发量！！！
class  sExtHydroReader(DaymetHydroReader):
    camels_root = None  # needs to be set in class method "init_root"
    forcing_root = None
    discharge_root = None
    forcing_cols = ["Year", "Mnth", "Day", "Hr", "Dayl(s)", "PRCP(mm/day)", "SRAD(W/m2)", "SWE(mm)", "Tmax(C)",
                    "Tmin(C)", "Vp(Pa)"]
    features = ["PRCP(mm/day)", "SRAD(W/m2)", "Tmax(C)", "Tmin(C)", "Vp(Pa)"]

    @classmethod
    def init_root(cls, camels_root):
        cls.camels_root = Path(camels_root)
        cls.forcing_root = cls.camels_root / "basin_mean_forcing" / "nldas_extended"
        cls.discharge_root = cls.camels_root / "usgs_streamflow"

    def __init__(self, basin: str):
        super().__init__(basin)


class GBHydroReader:
    camels_root = None  # needs to be set in class method "init_root"
    features = ["prcp", "srad", "tmax", "tmin", "pet"]
    discharge_cols = ["runoff"]
    target = ["runoff", "pet"]

    @classmethod
    def init_root(cls, camels_root):  # often be rewritten
        cls.camels_root = Path(camels_root)
        cls.forcing_root = cls.camels_root / "forcing"
        cls.discharge_root = cls.camels_root / "runoff"

    def __init__(self, basin: str):
        self.basin = basin
        self.area = None
        df = self._load_data()
        df = self._process_invalid_data(df)
        # print(df)  # TEST
        self.df_x = df[self.features]  # Datetime as index #NOTE:原来的
        self.df_y = df[self.target]  # Datetime as index
        # print(df["discharge_spec"])  # TEST,runoff读取
        # print(df.head())  # TEST
        # print(self.df_y.shape)  # TEST

    def _load_data(self):
        df_forcing = self._load_forcing()
        df_discharge = self._load_discharge()
        df = pd.concat([df_forcing, df_discharge], axis=1)

        return df

    def _load_forcing(self):
        files = list(self.forcing_root.glob(f"{self.basin}.csv"))
        if len(files) == 0:
            raise RuntimeError(f"No forcing file found for Basin {self.basin}")
        elif len(files) >= 2:
            raise RuntimeError(f"Redundant forcing files found for Basin {self.basin}")
        else:
            file_path = files[0]

        df = pd.read_csv(file_path, header=0, index_col="date")
        df = df[self.features]
        # print(df)

        return df

    def _load_discharge(self):
        files = list(self.discharge_root.glob(f"{self.basin}.csv"))
        if len(files) == 0:
            raise RuntimeError(f"No discharge file found for Basin {self.basin}")
        elif len(files) >= 2:
            raise RuntimeError(f"Redundant discharge files found for Basin {self.basin}")
        else:
            file_path = files[0]

        df = pd.read_csv(file_path, header=0, index_col="date")

        return df[self.discharge_cols]

    # Processing invalid data
    def _process_invalid_data(self, df: pd.DataFrame):
        # Delete all row, where exits NaN (only discharge has NaN in this dataset)
        len_raw = len(df)
        df = df.dropna()
        len_drop_nan = len(df)
        if len_raw > len_drop_nan:
            print(f"Deleted {len_raw - len_drop_nan} records because of NaNs {self.basin}")

        # Deletes all records, where no discharge was measured (-999)
        df = df.drop((df[df["runoff"] < 0]).index)
        len_drop_neg = len(df)
        if len_drop_nan > len_drop_neg:
            print(f"Deleted {len_drop_nan - len_drop_neg} records because of negative discharge {self.basin}")

        return df

    def get_df_x(self):
        return self.df_x

    def get_df_y(self):
        return self.df_y


# TODO: 需要蒸发量！！！
class HydroReaderFactory:
    """
    Simple factory for producing HydroReader
    """

    @staticmethod
    def get_hydro_reader(camels_root, forcing_type, basin, runoff):
        if forcing_type == "daymet":
            DaymetHydroReader.init_root(camels_root)
            reader = DaymetHydroReader(basin)
        elif forcing_type == "maurer_extended":
            MaurerExtHydroReader.init_root(camels_root)
            reader = MaurerExtHydroReader(basin)
        elif forcing_type == "nldas_extended":
            NldasExtHydroReader.init_root(camels_root)
            reader = NldasExtHydroReader(basin)
        elif forcing_type == "gb":  # TODO:先省事这么用着，换了数据集显然不算不同的forcing_type
            GBHydroReader.init_root(camels_root)
            reader = GBHydroReader(basin)
        else:
            raise RuntimeError(f"No such hydro reader type: {forcing_type}")

        return reader


class CamelsDataset(Dataset):
    """CAMELS dataset working with subclasses of AbstractHydroReader.

    It works in a list way: the model trains, validates and tests with all of basins in attribute:basins_list.

    Attributes:
        camels_root: str
            The root of CAMELS dataset.
        basins_list: list of str
            A list contains all needed basins-ids (8-digit code).
        past_len: int
            Length of the past time steps for discharge data.
        pred_len: int
            Length of the predicting time steps for discharge data.
            And it is worth noting that the used length of meteorological data is (past_len + :pred_len).
        stage: str
            One of ['train', 'val', 'test'], decide whether calculating mean and std or not.
            Calculate mean and std in training stage.
        dates: List of pd.DateTimes
            Means the date range that is used, containing two elements, i.e, start date and end date.
        x_dict: dict as {basin: np.ndarray}
             Mapping a basin to its corresponding meteorological data.
        y_dict: dict as {basin: np.ndarray}
             Mapping a basin to its corresponding discharge data.
        length_ls: list of int
            Contains number of serialized sequences of each basin corresponding to basins_list.
        index_ls: list of int
            Created from length_ls, used in __getitem__ method.
        num_samples: int
            Number of serialized sequences of all basins.
        x_mean: numpy.ndarray
            Mean of input features derived from the training stage.
            Has to be provided for 'val' or 'test' stage.
            Can be retrieved if calling .get_means() on the data set.
        y_mean: numpy.ndarray
            Mean of output features derived from the training stage.
            Has to be provided for 'val' or 'test' stage.
            Can be retrieved if calling .get_means() on the data set.
        x_std: numpy.ndarray
            Std of input features derived from the training stage.
            Has to be provided for 'val' or 'test' stage.
            Can be retrieved if calling .get_stds() on the data set.
        y_std: numpy.ndarray
            Std of output features derived from the training stage.
            Has to be provided for 'val' or 'test' stage.
            Can be retrieved if calling .get_stds() on the data set.
    """

    def __init__(self, camels_root: str, forcing_type: str, basins_list: list, past_len: int, pred_len: int, stage: str,
                 dates: list, x_mean=None, y_mean=None, x_std=None, y_std=None, y_stds_dict=None):
        """Initialization

        x_mean, y_mean, x_std, y_std should be provided if stage != "train".
        """
        self.camels_root = camels_root
        self.basins_list = basins_list
        self.past_len = past_len
        self.pred_len = pred_len
        self.stage = stage
        self.dates = dates
        self.x_dict = dict()
        self.y_dict = dict()
        self.date_index_dict = dict()
        self.length_ls = list()

        if y_stds_dict is None:
            self.y_stds_dict = dict()
        else:
            self.y_stds_dict = y_stds_dict

        self._load_data(forcing_type)
        # Calculate mean and std
        if self.stage == 'train':
            self.x_mean, self.x_std = self.calc_mean_and_std(self.x_dict)
            self.y_mean, self.y_std = self.calc_mean_and_std(self.y_dict)
        else:
            self.x_mean = x_mean
            self.y_mean = y_mean
            self.x_std = x_std
            self.y_std = y_std
        self.normalize_data()

        self.num_samples = 0
        for item in self.length_ls:
            self.num_samples += item

        self.index_ls = [0]
        for i in range(len(self.length_ls)):
            v = self.index_ls[i] + self.length_ls[i]
            self.index_ls.append(v)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        basin_idx = bisect_right(self.index_ls, idx) - 1
        local_idx = idx - self.index_ls[basin_idx]
        basin = self.basins_list[basin_idx]
        x_seq = self.x_dict[basin][local_idx: local_idx + self.past_len + self.pred_len, :]
        y_seq_past = self.y_dict[basin][local_idx: local_idx + self.past_len, :]
        y_seq_future = self.y_dict[basin][local_idx + self.past_len: local_idx + self.past_len + self.pred_len, :]

        return x_seq, y_seq_past, y_seq_future, self.y_stds_dict[basin]

    def _load_data(self, forcing_type):
        # Loading vanilla data
        basin_number = len(self.basins_list)
        for idx, basin in enumerate(self.basins_list):
            print(self.stage, f"{basin}: loading data %.4f" % (idx / basin_number))
            reader = HydroReaderFactory.get_hydro_reader(self.camels_root, forcing_type, basin)
            df_x = reader.get_df_x()
            df_y = reader.get_df_y()

            # Select date
            df_x = df_x[self.dates[0]:self.dates[1]]
            df_y = df_y[self.dates[0]:self.dates[1]]
            assert len(df_x) == len(df_y)
            self.date_index_dict[basin] = df_x.index

            # Select used features and discharge
            x = df_x.values.astype("float32")
            y = df_y.values.astype("float32")
            self.x_dict[basin] = x
            self.y_dict[basin] = y

            self.length_ls.append(len(x) - self.past_len - self.pred_len + 1)
            # Calculate mean and std in training stage
            if self.stage == 'train':
                self.y_stds_dict[basin] = y.std(axis=0).item()

    @staticmethod
    def calc_mean_and_std(data_dict):
        data_all = np.concatenate(list(data_dict.values()), axis=0)  # CAN NOT serializable
        nan_mean = np.nanmean(data_all, axis=0)
        nan_std = np.nanstd(data_all, axis=0)
        return nan_mean, nan_std

    def _local_normalization(self, feature: np.ndarray, variable: str) -> np.ndarray:
        if variable == 'inputs':
            feature = (feature - self.x_mean) / self.x_std
        elif variable == 'output':
            feature = (feature - self.y_mean) / self.y_std
        else:
            raise RuntimeError(f"Unknown variable type {variable}")
        return feature

    def normalize_data(self):
        # Normalize data
        for idx, basin in enumerate(self.basins_list):
            print(self.stage, "Normalizing %.4f" % (idx / len(self.basins_list)))
            x = self.x_dict[basin]
            y = self.y_dict[basin]
            # Normalize data
            x_norm = self._local_normalization(x, variable='inputs')
            y_norm = self._local_normalization(y, variable='output')
            self.x_dict[basin] = x_norm
            self.y_dict[basin] = y_norm

    def local_rescale(self, feature: np.ndarray, variable: str) -> np.ndarray:
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

    @classmethod
    def get_instance(cls, past_len: int, pred_len: int, stage: str, specific_cfg: dict,
                     x_mean=None, y_mean=None, x_std=None, y_std=None, y_stds_dict=None):
        final_data_path = specific_cfg["final_data_path"]
        camels_root = specific_cfg["camels_root"]
        basins_list = specific_cfg["basins_list"]
        forcing_type = specific_cfg["forcing_type"]
        start_date = specific_cfg["start_date"]
        end_date = specific_cfg["end_date"]
        use_runoff = specific_cfg["use_runoff"]  # ADD
        if final_data_path is None:
            dates = [start_date, end_date]
            if use_runoff:  # ADD
                instance = cls(camels_root, forcing_type, basins_list, past_len, pred_len, stage,
                               dates, x_mean, y_mean, x_std, y_std, y_stds_dict, use_runoff)
            else:
                instance = cls(camels_root, forcing_type, basins_list, past_len, pred_len, stage,
                               dates, x_mean, y_mean, x_std, y_std, y_stds_dict)
            return instance
        else:
            if final_data_path.exists():
                instance = torch.load(final_data_path)
                return instance
            else:
                dates = [start_date, end_date]
                if use_runoff:  # ADD
                    instance = cls(camels_root, forcing_type, basins_list, past_len, pred_len, stage,
                                   dates, x_mean, y_mean, x_std, y_std, y_stds_dict, use_runoff)
                else:
                    instance = cls(camels_root, forcing_type, basins_list, past_len, pred_len, stage,
                                   dates, x_mean, y_mean, x_std, y_std, y_stds_dict)
                final_data_path.parent.mkdir(exist_ok=True, parents=True)
                torch.save(instance, final_data_path)
                return instance


class StaticReader(AbstractStaticReader):
    """Static hydrological data reader.

    Reads data from a selected norm static attributes file (.csv).
    Need to ensure conversion from static attributes to pandas.DataFrame.
    """

    def __init__(self, camels_root):
        self.camels_root = Path(camels_root)
        self.static_file_path = Path(
            "/data1/du/CAMELS/CAMELS-US") / "camels_attributes_v2.0" / "selected_norm_static_attributes.csv"
        self.df_static = pd.read_csv(self.static_file_path, header=0, dtype={"gauge_id": str}).set_index("gauge_id")
        self.df_static = self.df_static.astype("float32")

    def get_df_static(self, basin):
        return self.df_static.loc[[basin]].values


class OtherStaticReader(AbstractStaticReader):
    """
    非us的数据集静态数据
    """

    def __init__(self, camels_root):  # NOTE:gb:8个static
        self.camels_root = Path(camels_root)
        self.static_file_path = camels_root / "static" / "static_attributes.csv"
        self.df_static = pd.read_csv(self.static_file_path, header=0, dtype={"basin": str}).set_index("basin")
        self.df_static = self.df_static.astype("float32")

    def get_df_static(self, basin):
        return self.df_static.loc[[basin]].values


class CamelsDatasetWithStatic(CamelsDataset):
    """CAMELS dataset with static attributes injected into serialized sequences.

    Inherited from NullableCamelsDataset

    """

    def __init__(self, camels_root: str, forcing_type: str, basins_list: list, past_len: int, pred_len: int, stage: str,
                 dates: list, x_mean=None, y_mean=None, x_std=None, y_std=None, y_stds_dict=None, use_runoff=False):
        if forcing_type == "gb":
            self.static_reader = GBStaticReader(camels_root)
        else:
            self.static_reader = StaticReader(camels_root)
        self.norm_static_fea = dict()
        self.use_runoff = use_runoff
        super().__init__(camels_root, forcing_type, basins_list, past_len, pred_len, stage,
                         dates, x_mean, y_mean, x_std, y_std, y_stds_dict)

    def _load_data(self, forcing_type):
        # Loading vanilla data
        basin_number = len(self.basins_list)
        for idx, basin in enumerate(self.basins_list):
            print(self.stage, f"{basin}: loading data %.4f" % (idx / basin_number))
            reader = HydroReaderFactory.get_hydro_reader(self.camels_root, forcing_type, basin)
            df_x = reader.get_df_x()
            df_y = reader.get_df_y()

            print("runoff", self.use_runoff)
            # Select date
            if not self.use_runoff:
                df_x = df_x[self.dates[0]:self.dates[1]]
                df_y = df_y[self.dates[0]:self.dates[1]]
                assert len(df_x) == len(df_y)
                self.date_index_dict[basin] = df_x.index
            else:
                self.dates[0] = self.dates[0] - pd.DateOffset(days=14 - 1)  # NOTE:为了和ealstm匹配同样的长度
                df_y = df_y[self.dates[0] - pd.DateOffset(days=1): self.dates[1]]  # NOTE:预备工作，最大范围

                df_runoff = df_y[self.dates[0] - pd.DateOffset(days=1):df_y.index[-1] - pd.DateOffset(days=1)]

                df_x = df_x[df_runoff.index[0] + pd.DateOffset(days=1):self.dates[1]]
                df_y = df_y[df_runoff.index[0] + pd.DateOffset(days=1):self.dates[1]]

                df_x = df_x.reset_index(drop=True)
                df_runoff = df_runoff.reset_index(drop=True)

                df_x = pd.concat([df_runoff, df_x], axis=1)
                assert len(df_x) == len(df_y)
                self.date_index_dict[basin] = df_y.index  # df_x这时候已经没有datetime的index了

            # Select used features and discharge
            x = df_x.values.astype("float32")
            y = df_y.values.astype("float32")

            self.x_dict[basin] = x
            self.y_dict[basin] = y

            if (len(x) - self.past_len - self.pred_len + 1 < 0):
                print("len(x) - self.past_len - self.pred_len + 1 < 0")
                raise ValueError("len(x) - self.past_len - self.pred_len + 1 < 0")
            self.length_ls.append(len(x) - self.past_len - self.pred_len + 1)
            # adding static attributes
            self.norm_static_fea[basin] = self.static_reader.get_df_static(basin)

            # Calculate mean and std in training stage
            if self.stage == 'train':
                self.y_stds_dict[basin] = y.std(axis=0).item()

    def normalize_data(self):
        # Normalize data
        for idx, basin in enumerate(self.basins_list):
            print(self.stage, "Normalizing %.4f" % (idx / len(self.basins_list)))
            x = self.x_dict[basin]
            y = self.y_dict[basin]
            # Normalize data
            x_norm = self._local_normalization(x, variable='inputs')
            y_norm = self._local_normalization(y, variable='output')
            norm_static_fea = self.norm_static_fea[basin].repeat(x_norm.shape[0], axis=0)
            x_norm_static = np.concatenate([x_norm, norm_static_fea], axis=1)
            self.x_dict[basin] = x_norm_static
            self.y_dict[basin] = y_norm

    def __getitem__(self, idx: int):
        basin_idx = bisect_right(self.index_ls, idx) - 1
        local_idx = idx - self.index_ls[basin_idx]
        basin = self.basins_list[basin_idx]
        x_seq = self.x_dict[basin][local_idx: local_idx + self.past_len + self.pred_len, :]
        y_seq_past = self.y_dict[basin][local_idx: local_idx + self.past_len, :]
        y_seq_future = self.y_dict[basin][local_idx + self.past_len: local_idx + self.past_len + self.pred_len, :]

        if self.stage == "test":
            return x_seq, y_seq_past, y_seq_future, []
        else:
            return x_seq, y_seq_past, y_seq_future, self.y_stds_dict[basin]


class DatasetFactory:
    @staticmethod
    def get_dataset_type(use_future_fea, use_static):
        if (not use_future_fea) and use_static:
            raise RuntimeError("No implemented yet.")
        elif not use_future_fea:
            raise RuntimeError("No implemented yet.")
        elif use_static:
            ds = CamelsDatasetWithStatic
        else:
            ds = CamelsDataset
        return ds


# NOTE:Sac的dataset
class CamelsDataset2Sac(Dataset):
    def __init__(self, camels_root: str, forcing_type: str, runoff: bool, basins_list: list, past_len: int,
                 pred_len: int, stage: str, dates: list, x_mean=None, y_mean=None, x_std=None, y_std=None,
                 y_stds_dict=None, freq='h'):
        self.camels_root = camels_root
        self.use_runoff = runoff
        self.basins_list = basins_list
        self.past_len = past_len
        self.pred_len = pred_len
        self.stage = stage
        self.dates = dates
        self.freq = freq

        self.x_dict = dict()
        self.y_dict = dict()
        # self.date_index_dict = dict()
        self.date_index = []
        self.date_index_dict = dict()
        self.data_stamp_dict = dict()
        self.length_ls = list()

        self.norm_static_fea = dict()

        self.static_reader = StaticReader(camels_root)

        if y_stds_dict is None:
            self.y_stds_dict = dict()
        else:
            self.y_stds_dict = y_stds_dict

        self._load_data(forcing_type)

        self.y_origin = copy(self.y_dict)  # TEST:NOTE:修改似乎无用

        # Calculate mean and std
        if self.stage == 'train':
            self.x_mean, self.x_std = self.calc_mean_and_std(self.x_dict)
            self.y_mean, self.y_std = self.calc_mean_and_std(self.y_dict)
        else:
            self.x_mean = x_mean
            self.y_mean = y_mean
            self.x_std = x_std
            self.y_std = y_std

        # self.x_dict_noNorm = self.x_dict  # TEST,noNorm
        # self.y_dict_noNorm = self.y_dict  # TEST,noNorm
        self.normalize_data()  # TEST,Norm

        self.num_samples = 0
        for item in self.length_ls:
            self.num_samples += item

        self.index_ls = [0]
        for i in range(len(self.length_ls)):
            v = self.index_ls[i] + self.length_ls[i]
            self.index_ls.append(v)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        basin_idx = bisect_right(self.index_ls, idx) - 1
        local_idx = idx - self.index_ls[basin_idx]
        basin = self.basins_list[basin_idx]

        x_seq = self.x_dict[basin][local_idx: local_idx + self.past_len + self.pred_len, :]
        # x_seq_noNorm = self.x_dict_noNorm[basin][local_idx: local_idx + self.past_len + self.pred_len, :]  # TEST:noNorm
        x_seq_mark = self.data_stamp_dict[basin][local_idx: local_idx + self.past_len + self.pred_len, :]

        y_seq_past = self.y_dict[basin][local_idx: local_idx + self.past_len, :]
        y_seq_future = self.y_dict[basin][local_idx + self.past_len: local_idx + self.past_len + self.pred_len, :]

        static_values = self.norm_static_fea[basin].reshape(1, -1)

        y_stds_basin = np.array(self.y_stds_dict[basin])

        return x_seq, x_seq_mark, y_seq_past, y_seq_future, static_values, y_stds_basin

    def _load_data(self, forcing_type):
        # Loading vanilla data
        basin_number = len(self.basins_list)
        for idx, basin in enumerate(self.basins_list):
            print(self.stage, f"{basin}: loading data %.4f" % (idx / basin_number))
            reader = HydroReaderFactory.get_hydro_reader(self.camels_root, forcing_type, basin, self.use_runoff)
            df_x = reader.get_df_x()
            df_y = reader.get_df_y()
            # print(df_x.shape, df_y.shape)  # TEST

            # # Select date
            # df_x = df_x[self.dates[0]:self.dates[1]]
            # # if self.stage=="val":  # TEST
            # #     print("val-----",len(df_x))  # TEST
            # df_y = df_y[self.dates[0]:self.dates[1]]
            # # ADD:timestamp
            # self.date_index = df_x.index
            # self.date_index_dict[basin] = self.date_index

            # print("runoff", self.use_runoff)
            # Select date
            if not self.use_runoff:
                df_x = df_x[self.dates[0]:self.dates[1]]
                df_y = df_y[self.dates[0]:self.dates[1]]
                # assert len(df_x) == len(df_y)
                self.date_index = df_x.index
                self.date_index_dict[basin] = df_x.index
            else:
                self.dates[0] = self.dates[0] - pd.DateOffset(
                    days=self.past_len - self.pred_len)  # NOTE:为了和ealstm匹配同样的长度
                df_y = df_y[self.dates[0] - pd.DateOffset(days=self.pred_len): self.dates[1]]  # NOTE:预备工作，最大范围

                df_runoff = df_y[self.dates[0] - pd.DateOffset(days=self.pred_len):df_y.index[-1] - pd.DateOffset(
                    days=self.pred_len)
                            ]["OBS_RUN"]

                df_x = df_x[df_runoff.index[0] + pd.DateOffset(days=self.pred_len):self.dates[1]]
                df_y = df_y[df_runoff.index[0] + pd.DateOffset(days=self.pred_len):self.dates[1]]

                df_x = df_x.reset_index(drop=True)
                df_runoff = df_runoff.reset_index(drop=True)

                # df_x = pd.concat([df_runoff, df_x], axis=1) #TEST,原来是最前面
                df_x = pd.concat([df_x, df_runoff], axis=1)  # TEST,原来是最前面
                assert len(df_x) == len(df_y)
                self.date_index = df_y.index
                self.date_index_dict[basin] = df_y.index  # df_x这时候已经没有datetime的index了

            assert len(df_x) == len(df_y)

            data_stamp = time_features(pd.to_datetime(self.date_index.values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

            # Select used features and discharge
            x = df_x.values.astype("float32")
            y = df_y.values.astype("float32")
            self.x_dict[basin] = x
            self.y_dict[basin] = y
            # ADD:timestamp
            self.data_stamp_dict[basin] = data_stamp.astype("float32")

            self.length_ls.append(max(len(x) - self.past_len - self.pred_len + 1, 0))
            # adding static attributes
            self.norm_static_fea[basin] = self.static_reader.get_df_static(basin)

            # Calculate mean and std in training stage
            if self.stage == 'train':
                # print(y.shape)  # TEST
                # self.y_stds_dict[basin] = y.std(axis=0).item()
                self.y_stds_dict[basin] = y.std(axis=0).tolist()
                # print(self.y_stds_dict[basin]) # TEST

            self.norm_static_fea[basin] = self.static_reader.get_df_static(basin)

    @staticmethod
    def calc_mean_and_std(data_dict):
        data_all = np.concatenate(list(data_dict.values()), axis=0)  # CAN NOT serializable
        nan_mean = np.nanmean(data_all, axis=0)
        nan_std = np.nanstd(data_all, axis=0)
        return nan_mean, nan_std

    def _local_normalization(self, feature: np.ndarray, variable: str) -> np.ndarray:
        if variable == 'inputs':
            feature = (feature - self.x_mean) / self.x_std
            for i in range(feature.shape[1]):
                # print(i)
                if self.x_mean[i] == 0 and self.x_std[i] == 0:
                    feature[:, i] = 0
        elif variable == 'output':
            feature = (feature - self.y_mean) / self.y_std
        else:
            raise RuntimeError(f"Unknown variable type {variable}")
        return feature

    def normalize_data(self):
        # Normalize data
        for idx, basin in enumerate(self.basins_list):
            print(self.stage, "Normalizing %.4f" % (idx / len(self.basins_list)))
            x = self.x_dict[basin]
            y = self.y_dict[basin]
            # Normalize data
            x_norm = self._local_normalization(x, variable='inputs')
            norm_static_fea = self.norm_static_fea[basin].repeat(x_norm.shape[0], axis=0)
            x_norm_static = np.concatenate([x_norm, norm_static_fea], axis=1)

            y_norm = self._local_normalization(y, variable='output')
            # self.x_dict[basin] = x_norm
            self.x_dict[basin] = x_norm_static
            self.y_dict[basin] = y_norm

    def local_rescale(self, feature: np.ndarray, variable: str) -> np.ndarray:
        if variable == 'inputs':
            feature = feature * self.x_std + self.x_mean
        elif variable == 'output':
            # print(self.y_std, self.y_mean)  # TEST
            # print(feature)
            # print(feature * self.y_std )
            feature = feature * self.y_std + self.y_mean
        else:
            raise RuntimeError(f"Unknown variable type {variable}")
        return feature

    def get_means(self):
        return self.x_mean, self.y_mean

    def get_stds(self):
        return self.x_std, self.y_std


class CamelsDataset2Others(Dataset):
    def __init__(self, camels_root: str, forcing_type: str, runoff: bool, basins_list: list, past_len: int,
                 pred_len: int, stage: str, dates: list, x_mean=None, y_mean=None, x_std=None, y_std=None,
                 y_stds_dict=None, freq='h'):
        self.camels_root = camels_root
        self.use_runoff = runoff
        self.basins_list = basins_list
        self.past_len = past_len
        self.pred_len = pred_len
        self.stage = stage
        self.dates = dates
        self.freq = freq

        self.x_dict = dict()
        self.y_dict = dict()
        # self.date_index_dict = dict()
        self.date_index = []
        self.date_index_dict = dict()
        self.data_stamp_dict = dict()
        self.length_ls = list()

        self.norm_static_fea = dict()

        self.static_reader = OtherStaticReader(camels_root)

        if y_stds_dict is None:
            self.y_stds_dict = dict()
        else:
            self.y_stds_dict = y_stds_dict

        self._load_data(forcing_type)

        self.y_origin = copy(self.y_dict)  # TEST:NOTE:修改似乎无用

        # Calculate mean and std
        if self.stage == 'train':
            self.x_mean, self.x_std = self.calc_mean_and_std(self.x_dict)
            self.y_mean, self.y_std = self.calc_mean_and_std(self.y_dict)
        else:
            self.x_mean = x_mean
            self.y_mean = y_mean
            self.x_std = x_std
            self.y_std = y_std

        # self.x_dict_noNorm = self.x_dict  # TEST,noNorm
        # self.y_dict_noNorm = self.y_dict  # TEST,noNorm
        self.normalize_data()  # TEST,Norm

        self.num_samples = 0
        for item in self.length_ls:
            self.num_samples += item

        self.index_ls = [0]
        for i in range(len(self.length_ls)):
            v = self.index_ls[i] + self.length_ls[i]
            self.index_ls.append(v)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        basin_idx = bisect_right(self.index_ls, idx) - 1
        local_idx = idx - self.index_ls[basin_idx]
        basin = self.basins_list[basin_idx]

        x_seq = self.x_dict[basin][local_idx: local_idx + self.past_len + self.pred_len, :]
        x_seq_mark = self.data_stamp_dict[basin][local_idx: local_idx + self.past_len + self.pred_len, :]

        y_seq_past = self.y_dict[basin][local_idx: local_idx + self.past_len, :]
        y_seq_future = self.y_dict[basin][local_idx + self.past_len: local_idx + self.past_len + self.pred_len, :]

        static_values = self.norm_static_fea[basin].reshape(1, -1)  # origin:但其实没用到的

        y_stds_basin = np.array(self.y_stds_dict[basin])

        return x_seq, x_seq_mark, y_seq_past, y_seq_future, static_values, y_stds_basin  # origin:但其实没用到的
        # return x_seq, x_seq_mark, y_seq_past, y_seq_future, y_stds_basin

    def _load_data(self, forcing_type):
        # Loading vanilla data
        basin_number = len(self.basins_list)
        for idx, basin in enumerate(self.basins_list):
            print(self.stage, f"{basin}: loading data %.4f" % (idx / basin_number))
            reader = HydroReaderFactory.get_hydro_reader(self.camels_root, forcing_type, basin, self.use_runoff)
            df_x = reader.get_df_x()
            df_y = reader.get_df_y()
            # print(df_x.shape, df_y.shape)  # TEST

            # # Select date
            # df_x = df_x[self.dates[0]:self.dates[1]]
            # # if self.stage=="val":  # TEST
            # #     print("val-----",len(df_x))  # TEST
            # df_y = df_y[self.dates[0]:self.dates[1]]
            # # ADD:timestamp
            # self.date_index = df_x.index
            # self.date_index_dict[basin] = self.date_index

            # print("runoff", self.use_runoff)
            # Select date
            if not self.use_runoff:
                df_x = df_x[self.dates[0]:self.dates[1]]
                df_y = df_y[self.dates[0]:self.dates[1]]
                # assert len(df_x) == len(df_y)
                self.date_index = df_x.index
                self.date_index_dict[basin] = df_x.index
            else:
                self.dates[0] = self.dates[0] - pd.DateOffset(days=self.past_len - 1)  # NOTE:为了和ealstm匹配同样的长度
                df_y = df_y[self.dates[0] - pd.DateOffset(days=1): self.dates[1]]  # NOTE:预备工作，最大范围

                df_runoff = df_y[self.dates[0] - pd.DateOffset(days=1):df_y.index[-1] - pd.DateOffset(days=1)
                            ]["runoff"]

                df_x = df_x[df_runoff.index[0] + pd.DateOffset(days=1):self.dates[1]]
                df_y = df_y[df_runoff.index[0] + pd.DateOffset(days=1):self.dates[1]]

                df_x = df_x.reset_index(drop=True)
                df_runoff = df_runoff.reset_index(drop=True)

                # df_x = pd.concat([df_runoff, df_x], axis=1) #TEST,原来是最前面
                df_x = pd.concat([df_x, df_runoff], axis=1)  # TEST,原来是最前面
                assert len(df_x) == len(df_y)
                self.date_index = df_y.index
                self.date_index_dict[basin] = df_y.index  # df_x这时候已经没有datetime的index了

            assert len(df_x) == len(df_y)

            data_stamp = time_features(pd.to_datetime(self.date_index.values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

            # Select used features and discharge
            x = df_x.values.astype("float32")
            y = df_y.values.astype("float32")
            self.x_dict[basin] = x
            self.y_dict[basin] = y
            # ADD:timestamp
            self.data_stamp_dict[basin] = data_stamp.astype("float32")

            self.length_ls.append(max(len(x) - self.past_len - self.pred_len + 1, 0))
            # adding static attributes
            self.norm_static_fea[basin] = self.static_reader.get_df_static(basin)

            # Calculate mean and std in training stage
            if self.stage == 'train':
                # print(y.shape)  # TEST
                # self.y_stds_dict[basin] = y.std(axis=0).item()
                self.y_stds_dict[basin] = y.std(axis=0).tolist()
                # print(self.y_stds_dict[basin]) # TEST

            self.norm_static_fea[basin] = self.static_reader.get_df_static(basin)

    @staticmethod
    def calc_mean_and_std(data_dict):
        data_all = np.concatenate(list(data_dict.values()), axis=0)  # CAN NOT serializable
        nan_mean = np.nanmean(data_all, axis=0)
        nan_std = np.nanstd(data_all, axis=0)
        return nan_mean, nan_std

    def _local_normalization(self, feature: np.ndarray, variable: str) -> np.ndarray:
        if variable == 'inputs':
            feature = (feature - self.x_mean) / self.x_std
            for i in range(feature.shape[1]):
                # print(i)
                if self.x_mean[i] == 0 and self.x_std[i] == 0:
                    feature[:, i] = 0
        elif variable == 'output':
            feature = (feature - self.y_mean) / self.y_std
        else:
            raise RuntimeError(f"Unknown variable type {variable}")
        return feature

    def normalize_data(self):
        # Normalize data
        for idx, basin in enumerate(self.basins_list):
            print(self.stage, "Normalizing %.4f" % (idx / len(self.basins_list)))
            x = self.x_dict[basin]
            y = self.y_dict[basin]
            # Normalize data
            x_norm = self._local_normalization(x, variable='inputs')
            norm_static_fea = self.norm_static_fea[basin].repeat(x_norm.shape[0], axis=0)
            x_norm_static = np.concatenate([x_norm, norm_static_fea], axis=1)

            y_norm = self._local_normalization(y, variable='output')
            # self.x_dict[basin] = x_norm
            self.x_dict[basin] = x_norm_static
            self.y_dict[basin] = y_norm

    def local_rescale(self, feature: np.ndarray, variable: str) -> np.ndarray:
        if variable == 'inputs':
            feature = feature * self.x_std + self.x_mean
        elif variable == 'output':
            # print(self.y_std, self.y_mean)  # TEST
            # print(feature)
            # print(feature * self.y_std )
            feature = feature * self.y_std + self.y_mean
        else:
            raise RuntimeError(f"Unknown variable type {variable}")
        return feature

    def get_means(self):
        return self.x_mean, self.y_mean

    def get_stds(self):
        return self.x_std, self.y_std


data_dict = {
    'daymet': CamelsDataset2Sac,  # us-daymet
    'maurer': CamelsDataset2Sac,  # us-maurer_extended
    'united': UnitedSerializedDataset,
    'separated': SeparatedSerializedDatasetFactory.get_datasets
}


def data_provider(cfg, stage, single=False):
    # Data = data_dict[cfg['model']] #TEST：原来的dataset使用

    if stage != 'train' and cfg['stage'] != 'finetune' and 'yr_s' not in cfg['forcing_type'] and cfg['finetune_name'] == None:
        train_x_mean = np.loadtxt(cfg['data_dir'] / "x_means.csv", dtype="float32")
        train_x_std = np.loadtxt(cfg['data_dir'] / "x_stds.csv", dtype="float32")
        train_y_mean = np.loadtxt(cfg['data_dir'] / "y_means.csv", dtype="float32")
        train_y_std = np.loadtxt(cfg['data_dir'] / "y_stds.csv", dtype="float32")
        with open(cfg['data_dir'] / "y_stds_dict.json", "rt") as f:
            y_stds_dict = json.load(f)
    else:
        train_x_mean = None
        train_y_mean = None
        train_x_std = None
        train_y_std = None
        y_stds_dict = None

    # NOTE:可能需要修改
    if cfg['model'] in MODELLIST:
        if 'united' in cfg['forcing_type'] or 'separated' in cfg['forcing_type']:
            if 'separated' == cfg['forcing_type']:
                Data = data_dict['separated']  # TEST：
                dss_cfg = init_dss_cfg(cfg['dss_config_path'])

                if cfg['stage'] !='finetune' and cfg['finetune_name'] == None:
                    if stage == 'test':
                        _, _, datasets = Data(dss_cfg, cfg, norm_separated=False,
                                              x_mean=train_x_mean, y_mean=train_y_mean,
                                              x_std=train_x_std, y_std=train_y_std)
                        test_loaders = dict()
                        for full_index in datasets:
                            test_loaders[full_index] = DataLoader(datasets[full_index], batch_size=cfg['batch_size'],
                                                                  num_workers=cfg['num_workers'], shuffle=False)
                        # data_loader = DataLoader(dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                        #                          shuffle=True, drop_last=cfg['drop_last'])
                        return datasets, test_loaders
                    elif stage == 'train':
                        datasets, _, _ = Data(dss_cfg, cfg, norm_separated=True)
                        train_loaders = dict()
                        for full_index in datasets:
                            train_loaders[full_index] = DataLoader(datasets[full_index], batch_size=cfg['batch_size'],
                                                                  num_workers=cfg['num_workers'], shuffle=True)
                        # data_loader = DataLoader(dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                        #                          shuffle=True, drop_last=cfg['drop_last'])
                        return datasets, train_loaders
                    elif stage == 'val':
                        _, datasets, _ = Data(dss_cfg, cfg, norm_separated=False,
                                              x_mean=train_x_mean, y_mean=train_y_mean,
                                              x_std=train_x_std, y_std=train_y_std)
                        val_loaders = dict()
                        for full_index in datasets:
                            val_loaders[full_index] = DataLoader(datasets[full_index], batch_size=cfg['batch_size'],
                                                                  num_workers=cfg['num_workers'], shuffle=False)
                        # data_loader = DataLoader(dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                        #                          shuffle=True, drop_last=cfg['drop_last'])
                        return datasets, val_loaders
                else:
                    print('yr_finetune')
                    train_loaders = dict()
                    val_loaders = dict()
                    test_loaders = dict()
                    datasets_train, datasets_val, datasets_test = Data(dss_cfg, cfg, norm_separated=True,
                                              x_mean=train_x_mean, y_mean=train_y_mean,
                                              x_std=train_x_std, y_std=train_y_std)
                    test_loaders = dict()
                    for full_index in datasets_train:
                        train_loaders[full_index] = DataLoader(datasets_train[full_index], batch_size=cfg['batch_size'],
                                                               num_workers=cfg['num_workers'], shuffle=True)
                        val_loaders[full_index] = DataLoader(datasets_val[full_index], batch_size=cfg['batch_size'],
                                                             num_workers=cfg['num_workers'], shuffle=False)
                        test_loaders[full_index] = DataLoader(datasets_test[full_index], batch_size=cfg['batch_size'],
                                                              num_workers=cfg['num_workers'], shuffle=False)
                    if stage == 'test':
                        return datasets_test, test_loaders
                    elif stage == 'train':
                        return datasets_train, train_loaders
                    elif stage == 'val':
                        return datasets_val, val_loaders
            elif 'united' in cfg['forcing_type']:
                # 如果使用SAC对units进行预测dataset采用UnitedSerializedDataset
                Data = data_dict[cfg['forcing_type']]  # TEST：多数据集使用
                dss_cfg = init_dss_cfg(cfg['dss_config_path']) # 读取txt文件中的basinList
                dataset = Data(dss_cfg, cfg, stage, x_mean=train_x_mean, y_mean=train_y_mean,
                               x_std=train_x_std, y_std=train_y_std)  # NOTE:在dataset里面保存了mean和std
                # print(cfg['data_dir'])  # test
                # raise ValueError# test
                data_loader = DataLoader(dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                                         shuffle=True, drop_last=cfg['drop_last'])

        else:
            Data = data_dict[cfg['forcing_type']]  # TEST：多数据集使用
            if not single:
                # Dataset
                # 计算dataset需要的信息
                basins_list = pd.read_csv(cfg["basins_list_path"], header=None, dtype=str)[0].values.tolist()
                dates = [cfg[f'{stage}_start'], cfg[f'{stage}_end']]
                print(dates)  # TEST

                if stage == 'train':
                    print('train')
                    # Training data
                    dataset = Data(cfg["camels_root"], cfg["forcing_type"], cfg["use_runoff"], basins_list,
                                   cfg["past_len"],
                                   cfg["pred_len"], stage, dates,
                                   x_mean=train_x_mean, y_mean=train_y_mean,
                                   x_std=train_x_std, y_std=train_y_std, freq=cfg['freq'], y_stds_dict=y_stds_dict)
                    data_loader = DataLoader(dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                                             shuffle=True, drop_last=cfg['drop_last'])  # TEST：看哪里的破数据有问题

                    # We use the feature means/stds of the training data for normalization in val and test stage
                    train_x_mean, train_y_mean = dataset.get_means()
                    train_x_std, train_y_std = dataset.get_stds()
                    y_stds_dict = dataset.y_stds_dict

                    # Saving training mean and training std
                    np.savetxt(cfg['data_dir'] / "x_means.csv", train_x_mean)
                    np.savetxt(cfg['data_dir'] / "x_stds.csv", train_x_std)
                    np.savetxt(cfg['data_dir'] / "y_means.csv", train_y_mean)
                    np.savetxt(cfg['data_dir'] / "y_stds.csv", train_y_std)
                    with open(cfg['data_dir'] / "y_stds_dict.json", "wt") as f:
                        json.dump(y_stds_dict, f)
                else:
                    dataset = Data(cfg["camels_root"], cfg["forcing_type"], cfg["use_runoff"], basins_list,
                                   cfg["past_len"],
                                   cfg["pred_len"], stage, dates,
                                   x_mean=train_x_mean, y_mean=train_y_mean,
                                   x_std=train_x_std, y_std=train_y_std, freq=cfg['freq'], y_stds_dict=y_stds_dict)
                    data_loader = DataLoader(dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                                             shuffle=False, drop_last=False)  # NOTE:测试的时候还是不drop
            elif single:
                basins_list = pd.read_csv(cfg["basins_list_path"], header=None, dtype=str)[0].values.tolist()
                dates = [cfg[f'{stage}_start'], cfg[f'{stage}_end']]
                print(dates)  # TEST
                dataset_dict = dict()
                data_loader_dict = dict()
                for i, basin in enumerate(basins_list):
                    dataset = Data(cfg["camels_root"], cfg["forcing_type"], cfg["use_runoff"], [basin], cfg["past_len"],
                                   cfg["pred_len"], stage, dates,
                                   x_mean=train_x_mean, y_mean=train_y_mean,
                                   x_std=train_x_std, y_std=train_y_std, freq=cfg['freq'], y_stds_dict=y_stds_dict)
                    data_loader = DataLoader(dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                                             shuffle=False,  # NOTE:注意这里不要shuffle
                                             drop_last=False)  # NOTE:测试的时候还是不drop_last
                    dataset_dict[basin] = dataset
                    data_loader_dict[basin] = data_loader
                return dataset_dict, data_loader_dict
            else:
                raise NotImplementedError("未考虑")
    else:
        raise NotImplementedError("未考虑")

    return dataset, data_loader


def data_provider_finetune(cfg, stage, basin):
    Data = data_dict[cfg['model']]
    # print(stage)
    if stage != 'train':
        train_x_mean = np.loadtxt(cfg['data_dir'] / basin / "x_means.csv", dtype="float32")
        train_x_std = np.loadtxt(cfg['data_dir'] / basin / "x_stds.csv", dtype="float32")
        train_y_mean = np.loadtxt(cfg['data_dir'] / basin / "y_means.csv", dtype="float32")
        train_y_std = np.loadtxt(cfg['data_dir'] / basin / "y_stds.csv", dtype="float32")
        with open(cfg['data_dir'] / basin / "y_stds_dict.json", "rt") as f:
            y_stds_dict = json.load(f)
    else:
        train_x_mean = None
        train_y_mean = None
        train_x_std = None
        train_y_std = None
        y_stds_dict = None

    # NOTE:可能需要修改
    if cfg['model'] == 'Sac':
        # Dataset
        # 计算dataset需要的信息
        # basins_list = pd.read_csv(cfg["basins_list_path"], header=None, dtype=str)[0].values.tolist()
        dates = [cfg[f'{stage}_start'], cfg[f'{stage}_end']]
        print(dates)  # TEST

        if stage == 'train':
            print('train')
            # Training data
            dataset = Data(cfg["camels_root"], cfg["forcing_type"], cfg["use_runoff"], [basin], cfg["past_len"],
                           cfg["pred_len"], stage, dates,
                           x_mean=train_x_mean, y_mean=train_y_mean,
                           x_std=train_x_std, y_std=train_y_std, freq=cfg['freq'], y_stds_dict=y_stds_dict)
            data_loader = DataLoader(dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                                     shuffle=True, drop_last=cfg['drop_last'])  # TODO:随机化drop性能好，generator不drop好
        else:
            dataset = Data(cfg["camels_root"], cfg["forcing_type"], cfg["use_runoff"], [basin], cfg["past_len"],
                           cfg["pred_len"], stage, dates,
                           x_mean=train_x_mean, y_mean=train_y_mean,
                           x_std=train_x_std, y_std=train_y_std, freq=cfg['freq'], y_stds_dict=y_stds_dict)
            data_loader = DataLoader(dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                                     shuffle=False, drop_last=False)  # NOTE:测试的时候还是不drop

        if stage == 'train':
            # We use the feature means/stds of the training data for normalization in val and test stage
            train_x_mean, train_y_mean = dataset.get_means()
            train_x_std, train_y_std = dataset.get_stds()
            y_stds_dict = dataset.y_stds_dict

            # ADD
            t_dir = cfg['data_dir'] / basin
            if not t_dir.is_dir():
                t_dir.mkdir(parents=True)
                # ADD
                # Saving training mean and training std
                np.savetxt(cfg['data_dir'] / basin / "x_means.csv", train_x_mean)
                np.savetxt(cfg['data_dir'] / basin / "x_stds.csv", train_x_std)
                np.savetxt(cfg['data_dir'] / basin / "y_means.csv", train_y_mean)
                np.savetxt(cfg['data_dir'] / basin / "y_stds.csv", train_y_std)
                with open(cfg['data_dir'] / basin / "y_stds_dict.json", "wt") as f:
                    json.dump(y_stds_dict, f)
    else:
        raise NotImplementedError("未考虑")

    return dataset, data_loader
