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

from model import AbhishekModel, TorchModel
from processing import prepare_validation_features, postprocess_qa_predictions
from utils import parse_args_inference

os.environ["TOKENIZERS_PARALLELISM"] = "false"

disable_progress_bar()


@torch.no_grad()
def predict(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int = 64,
    workers: int = 4
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
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        output = model(input_ids, attention_mask)
        start_logits.append(output.start_logits.cpu().numpy())
        end_logits.append(output.end_logits.cpu().numpy())
    return np.vstack(start_logits), np.vstack(end_logits)


def make_model(model_name: str, model_type: str = "hf", model_weights: str = None) -> nn.Module:
    if model_type == "hf":
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    elif model_type == "abhishek":
        model = AbhishekModel(model_name)
    elif model_type == "torch":
        model = TorchModel(model_name)
    else:
        raise ValueError(f"{model_type} is not a recognised model type.")
    if model_weights:
        print(f"Loading weights from {model_weights}")
        model.load_state_dict(torch.load(model_weights))
    return model


if __name__ == "__main__":
    config = parse_args_inference()
    data = pd.read_csv(config.input_data)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
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
        model = make_model(config.base_model, config.model_type, checkpoint)
        model.to(config.device)
        if config.fp16:
            model = model.half()
        start_logits, end_logits = predict(
            model,
            input_dataset,
            config.batch_size,
            config.dataloader_workers
        )
        pred_df = postprocess_qa_predictions(
            dataset,
            tokenized_dataset,
            (start_logits, end_logits),
            tokenizer
        )
        pred_df.to_csv(f"{filename}.csv", index=False)
        np.save(f"{filename}_start_logits.npy", start_logits)
        np.save(f"{filename}_end_logits.npy", end_logits)
        del model
        gc.collect()
        torch.cuda.empty_cache()
