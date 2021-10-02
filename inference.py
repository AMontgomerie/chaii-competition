import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import Dataset
from datasets.utils import disable_progress_bar
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import gc

from model import ChaiiModel
from processing import prepare_validation_features, postprocess_qa_predictions
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
        if config.model_type == "hf":
            model = AutoModelForQuestionAnswering.from_pretrained(config.base_model)
        else:
            model = ChaiiModel(config.base_model)
        if config.model_name:
            filename = f"{config.model_name.replace('/', '-')}_fold_{fold}.bin"
        else:
            filename = f"{config.base_model.replace('/', '-')}_fold_{fold}.bin"
        checkpoint = os.path.join(config.model_weights_dir, filename)
        model.load_state_dict(torch.load(checkpoint))
        model.to(config.device)
        start_logits, end_logits = predict(model, input_dataset)
        pred_df = postprocess_qa_predictions(
            dataset,
            tokenized_dataset,
            (start_logits, end_logits),
            tokenizer
        )
        if config.base_model_name:
            filename = f"{config.model_name.replace('/', '-')}_fold_{fold}.csv"
        else:
            filename = f"{config.base_model.replace('/', '-')}_fold_{fold}.csv"
        pred_df.to_csv(filename, index=False)
        del model
        gc.collect()
