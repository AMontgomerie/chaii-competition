import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from datasets import Dataset
from datasets.utils import disable_progress_bar
from transformers import AutoTokenizer
import gc

from model import make_model
from processing import prepare_validation_features, postprocess_qa_predictions
from utils import parse_args_inference

os.environ["TOKENIZERS_PARALLELISM"] = "false"

disable_progress_bar()


@torch.no_grad()
def predict(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int = 64,
    workers: int = 4,
    device: str = "cuda"
) -> np.ndarray:
    model.eval()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    start_logits = []
    end_logits = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        output = model(input_ids, attention_mask)
        start_logits.append(output.start_logits.cpu().numpy())
        end_logits.append(output.end_logits.cpu().numpy())
    return np.vstack(start_logits), np.vstack(end_logits)


if __name__ == "__main__":
    config = parse_args_inference()
    data = pd.read_csv(config.input_data)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    dataset = Dataset.from_pandas(data)
    tokenized_dataset = dataset.map(
        prepare_validation_features,
        batched=True,
        remove_columns=dataset.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": config.max_length,
            "doc_stride": config.doc_stride
        }
    )
    input_dataset = tokenized_dataset.map(
        lambda example: example, remove_columns=['example_id', 'offset_mapping']
    )
    input_dataset.set_format(type="torch")

    if len(config.select_folds) > 0:
        folds = [int(fold) for fold in config.select_folds]
    else:
        folds = range(config.num_folds)

    for fold in folds:
        print(f"Generating predictions for fold {fold}")
        if config.model_name:
            filename = f"{config.model_name.replace('/', '-')}_fold_{fold}"
        else:
            filename = f"{config.base_model.replace('/', '-')}_fold_{fold}"
        checkpoint = os.path.join(config.model_weights_dir, f"{filename}.bin")
        model = make_model(
            config.base_model,
            config.model_type,
            checkpoint,
            config.device
        )
        if config.fp16:
            with autocast():
                start_logits, end_logits = predict(
                    model,
                    input_dataset,
                    config.batch_size,
                    config.dataloader_workers,
                    config.device
                )
        else:
            start_logits, end_logits = predict(
                model,
                input_dataset,
                config.batch_size,
                config.dataloader_workers,
                config.device
            )
        if config.output_csv:
            pred_df = postprocess_qa_predictions(
                dataset,
                tokenized_dataset,
                (start_logits, end_logits),
                tokenizer
            )
            pred_df.to_csv(f"{filename}.csv", index=False)
        if config.output_logits:
            np.save(f"{filename}_start_logits.npy", start_logits)
            np.save(f"{filename}_end_logits.npy", end_logits)
        del model
        gc.collect()
        torch.cuda.empty_cache()
