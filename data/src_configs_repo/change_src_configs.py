import shutil
import os
import argparse
from pathlib import Path

configs_root_name = "src_configs"
repo_root = Path(os.path.dirname(__file__))
project_root = repo_root.parent
configs_root = project_root / configs_root_name
if configs_root.exists():
    shutil.rmtree(configs_root)

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir")
args = parser.parse_args()
d = args.dir
shutil.copytree(repo_root / d, project_root / configs_root_name)
print("src_configs set up completed.")
