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
from processing import (
    prepare_validation_features,
    postprocess_qa_predictions,
    filter_pred_strings
)
from utils import jaccard, parse_args_inference


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


def get_mean_oof(df: pd.DataFrame) -> float:
    jaccard_scores = df[["answer_text", "PredictionString"]].apply(jaccard, axis=1)
    return np.mean(jaccard_scores)


if __name__ == "__main__":
    config = parse_args_inference()
    data = pd.read_csv(config.input_data)
    fold_preds = []
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)

    for fold in range(config.num_folds):
        print(f"Generating predictions for fold {fold}")
        valid = data[data.kfold == fold]
        dataset = Dataset.from_pandas(valid)
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
        checkpoint = os.path.join(config.model_weights_dir, f"fold_{fold}", "model.bin")
        model.load_state_dict(torch.load(checkpoint))
        model.to(config.device)
        start_logits, end_logits = predict(model, input_dataset)
        processed_preds = postprocess_qa_predictions(
            dataset,
            tokenized_dataset,
            (start_logits, end_logits),
            tokenizer
        )
        preds_df = pd.DataFrame({
            "id": processed_preds.keys(),
            "PredictionString": processed_preds.values()}
        )
        fold_preds.append(preds_df)
        del model
        gc.collect()

    all_preds = pd.concat(fold_preds)
    oof = data.merge(all_preds, on="id")
    oof.to_csv(os.path.join(config.save_dir, "oof.csv"), index=False)
    oof["PredictionString"] = filter_pred_strings(oof.PredictionString)
    oof_hindi = get_mean_oof(oof[oof.language == "hindi"])
    oof_tamil = get_mean_oof(oof[oof.language == "tamil"])
    oof_all = get_mean_oof(oof)
    print(f"OOF (Hindi): {oof_hindi}")
    print(f"OOF (Tamil): {oof_tamil}")
    print(f"OOF (Overall): {oof_all}")
