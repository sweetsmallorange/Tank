import os
import sys
from pathlib import Path, PosixPath
from typing import List
import torch


def get_basin_list(basin_list_path) -> List:
    """Read list of basins from text file.

    Returns
    -------
    List
        List containing the 8-digit basin code of all basins
    """
    basin_file = basin_list_path
    with basin_file.open('r') as fp:
        basins = fp.readlines()
    basin_list = [basin.strip() for basin in basins]
    return basin_list


class BestModelLog:
    def __init__(self, init_model, saving_root, metric_name, high_better: bool,
                 log_all: bool = False):  # CHANGE:add log_all
        self.high_better = high_better
        self.saving_root = saving_root
        self.metric_name = metric_name
        worst = float("-inf") if high_better else float("inf")
        self.best_epoch = -1
        self.best_value = worst
        self.best_model_path = self.saving_root / f"({self.metric_name})_{self.best_epoch}_{self.best_value}.pkl"
        self.log_all = log_all  # ADD
        if not self.log_all:  # ADD
            torch.save(init_model.state_dict(), self.best_model_path)

    def update(self, model, new_value, epoch):
        if self.log_all:
            self.best_value = new_value
            self.best_epoch = epoch
            self.best_model_path = self.saving_root / f"({self.metric_name})_{self.best_epoch}_{self.best_value}.pkl"
            torch.save(model.state_dict(), self.best_model_path)
            return

        if ((self.high_better is True) and (new_value > self.best_value)) or \
                ((self.high_better is False) and (new_value < self.best_value)):
            os.remove(self.best_model_path)
            self.best_value = new_value
            self.best_epoch = epoch
            self.best_model_path = self.saving_root / f"({self.metric_name})_{self.best_epoch}_{self.best_value}.pkl"
            torch.save(model.state_dict(), self.best_model_path)


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'False', 'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'True', 'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')
