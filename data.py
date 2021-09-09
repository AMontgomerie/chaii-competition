import os
import pandas as pd
import torch


def get_extra_data(data_dir: str = "extra_data") -> pd.DataFrame:
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith(".csv")]
    datasets = [pd.read_csv(os.path.join(data_dir, f), encoding="utf-8") for f in files]
    return pd.concat(datasets)


class ChaiiDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if "token_type_ids" in self.data[item]:
            return {
                "ids": torch.tensor(self.data[item]["input_ids"], dtype=torch.long),
                "mask": torch.tensor(self.data[item]["attention_mask"], dtype=torch.long),
                "token_type_ids": torch.tensor(self.data[item]["token_type_ids"], dtype=torch.long),
                "start_positions": torch.tensor(self.data[item]["start_positions"], dtype=torch.long),
                "end_positions": torch.tensor(self.data[item]["end_positions"], dtype=torch.long),
            }
        return {
            "ids": torch.tensor(self.data[item]["input_ids"], dtype=torch.long),
            "mask": torch.tensor(self.data[item]["attention_mask"], dtype=torch.long),
            "start_positions": torch.tensor(self.data[item]["start_positions"], dtype=torch.long),
            "end_positions": torch.tensor(self.data[item]["end_positions"], dtype=torch.long),
        }
