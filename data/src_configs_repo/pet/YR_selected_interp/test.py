import numpy as np
import pandas as pd
import os
import shutil
import glob
from pathlib import Path

# df = pd.read_csv("/data2/zw/dataset/9_Normal_Camels_YR/selected_norm_integrated_static_attributes.csv", dtype='str')
# basins = df['basin'].values
# print(basins)
#
# origin=Path("/data2/zw/dataset/9_Normal_Camels_YR/Camels_YR_basin_by_day")
# for file in os.listdir(origin):
#     if file[:4] in basins:
#         if file.endswith("_forcing.csv"):
#             src_path=origin/file
#             print(src_path)
#             dst_path=Path("/data2/zw/dataset/9_Normal_Camels_YR/my_YR")/file
#             print(dst_path)
#             shutil.copy(src_path,dst_path)
#
#
#
# targetDir = Path("/data2/zw/dataset/9_Normal_Camels_YR/my_YR")


#
# path=Path("/data2/zw/dataset/9_Normal_Camels_YR/my_YR/forcing")
# for file in os.listdir(path):
#     df = pd.read_csv(path/file,dtype='str')
#     df['pet']=df['evp']
#     print(df.iloc[0])



df = pd.read_csv("/data2/zw/dataset/9_Normal_Camels_YR/my_YR/static/selected_norm_integrated_static_attributes.csv")
print(df.head())
print(len(df.columns))
