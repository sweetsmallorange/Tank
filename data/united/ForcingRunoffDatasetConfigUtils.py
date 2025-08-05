import yaml
from pathlib import Path
import pandas as pd


def init_dss_cfg(find_root):
    dss_cfg = dict()
    #
    # selected_yml_paths = list()
    # for ds in used_ds:
    #     selected_yml_paths.append(list(Path(find_root).glob(f"[[]{ds}[]]Selected*.yml"))[0])
    # selected_yml_paths = sorted(selected_yml_paths, key=lambda x: x.name)
    # for selected_yml_path in selected_yml_paths:
    #     f_selected = open(selected_yml_path, "rb")
    #     left = str(selected_yml_path).find("[")
    #     right = str(selected_yml_path).find("]")
    #     dataset_name = str(selected_yml_path)[left + 1:right]
    #     f_split = open(f"{find_root}/[{dataset_name}]Split.yml", "rb")
    #     yaml_selected = yaml.load(f_selected, Loader=yaml.FullLoader)
    #     yaml_split = yaml.load(f_split, Loader=yaml.FullLoader)
    #     dss_cfg[dataset_name] = {
    #         "basins": yaml_selected["basins"],
    #         "start_date": yaml_selected["start_date"],
    #         "end_date": yaml_selected["end_date"],
    #         "train_start": yaml_split["train_start"],
    #         "train_end": yaml_split["train_end"],
    #         "val_start": yaml_split["val_start"],
    #         "val_end": yaml_split["val_end"],
    #         "test_start": yaml_split["test_start"],
    #         "test_end": yaml_split["test_end"]
    #     }
    #     f_split.close()
    #     f_selected.close()

    # 假设文件名为 'basins.txt'
    basin_list = []
    date_range = {
        "maurer": {
            "start_date" : pd.to_datetime("1980-01-01", format="%Y-%m-%d"),
            "end_date"   : pd.to_datetime("2008-12-31", format="%Y-%m-%d")
        }
    }
    # 读取文件
    with open(find_root, 'r', encoding='utf-8') as file:
        basin_list = [int(line.strip()) for line in file.readlines()]

    dss_cfg["basins"] = basin_list
    dss_cfg["start_date"] = date_range["maurer"]["start_date"]
    dss_cfg["end_date"] = date_range["maurer"]["end_date"]

    return dss_cfg
