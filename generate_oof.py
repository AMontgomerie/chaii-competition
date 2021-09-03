import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    AdamW,
    AutoTokenizer,
    AutoModelForQuestionAnswering
)
import collections
from tqdm.auto import tqdm
import gc

from processing import (
    prepare_validation_features,
    postprocess_qa_predictions,
    filter_pred_strings
)
from utils import jaccard


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", required=False)
    parser.add_argument("--doc_stride", type=int, default=128, required=False)
    parser.add_argument("--max_answer_length", type=int, default=30, required=False)
    parser.add_argument("--max_length", type=int, default=384, required=False)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--num_folds", type=int, default=5, required=False)
    parser.add_argument("--save_dir", type=str, default="../output", required=False)
    parser.add_argument("--batch_size", type=int, default=64, required=False)
    return parser.parse_args()


@torch.no_grad()
def predict(model, dataset):
    model.eval()
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False
    )
    print(f"Iterating over {len(dataloader)} batches.")
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
    config = parse_args()
    data = pd.read_csv("train_folds.csv")
    fold_preds = []

    for fold in range(config.num_folds):
        print(f"Generating predictions for fold {fold}")
        checkpoint = f"../input/chaii-deepset-xlm-roberta-large-squad2/fold_{fold}"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
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
        model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
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
    oof.to_csv(
        os.path.join(config.save_dir, f"{config.model_dir}", "oof.csv"),
        index=False
    )
    oof["PredictionString"] = filter_pred_strings(oof.PredictionString)
    jaccard_scores = oof[["answer_text", "PredictionString"]].apply(jaccard, axis=1)
    mean_oof = np.mean(jaccard_scores)
    print(f"Mean OOF: {mean_oof}")
