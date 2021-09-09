import os
import pandas as pd
import torch


def get_extra_data(data_dir: str = "extra_data") -> pd.DataFrame:
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith(".csv")]
    datasets = [pd.read_csv(os.path.join(data_dir, f), encoding="utf-8") for f in files]
    return pd.concat(datasets)
