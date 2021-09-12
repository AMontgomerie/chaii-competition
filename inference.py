import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import Dataset
from datasets.utils import disable_progress_bar
from transformers import AutoTokenizer
import gc

from model import ChaiiModel
from processing import prepare_validation_features
from utils import parse_args_inference

disable_progress_bar()


@torch.no_grad()
def predict(model: nn.Module, dataset: Dataset) -> np.ndarray:
    model.eval()
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False
    )
    start_logits = []
    end_logits = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        output = model(input_ids, attention_mask)
        start_logits.append(output.start_logits.cpu().numpy())
        end_logits.append(output.end_logits.cpu().numpy())
    return np.vstack(start_logits), np.vstack(end_logits)


if __name__ == "__main__":
    config = parse_args_inference()
    data = pd.read_csv(config.input_data)
    fold_start_logits = []
    fold_end_logits = []
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)

    for fold in range(config.num_folds):
        print(f"Generating predictions for fold {fold}")
        dataset = Dataset.from_pandas(data)
        tokenized_dataset = dataset.map(
            prepare_validation_features,
            batched=True,
            remove_columns=dataset.column_names,
            fn_kwargs={"tokenizer": tokenizer}
        )
        input_dataset = tokenized_dataset.map(
            lambda example: example, remove_columns=['example_id', 'offset_mapping']
        )
        input_dataset.set_format(type="torch")
        model = ChaiiModel(config.base_model)
        checkpoint = os.path.join(
            config.model_weights_dir,
            f"{config.base_model.replace('/', '-')}_fold_{fold}.bin"
        )
        model.load_state_dict(torch.load(checkpoint))
        model.to(config.device)
        start_logits, end_logits = predict(model, input_dataset)
        fold_start_logits.append(start_logits)
        fold_end_logits.append(end_logits)
        del model
        gc.collect()

    start_logits = np.array(fold_start_logits)
    end_logits = np.array(fold_end_logits)
    np.save(os.path.join(config.save_dir, f"start_logits.npy"), start_logits)
    np.save(os.path.join(config.save_dir, f"end_logits.npy"), end_logits)
