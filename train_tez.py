import os
from transformers import AutoTokenizer
import numpy as np
import torch
import pandas as pd
from functools import partial
from transformers import default_data_collator
from tez import enums
from tez.callbacks import Callback
from tqdm import tqdm
from datasets import Dataset
from datasets.utils import disable_progress_bar
from model import TezChaiiModel
from utils import seed_everything, jaccard, parse_args
from processing import (
    prepare_train_features,
    prepare_validation_features,
    postprocess_qa_predictions,
    convert_answers
)
from data import get_extra_data, ChaiiDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

disable_progress_bar()


class EarlyStopping(Callback):
    def __init__(
        self,
        monitor,
        model_path,
        valid_dataframe,
        valid_data_loader,
        tokenizer,
        pad_on_right,
        max_length,
        doc_stride,
        patience=3,
        mode="min",
        delta=0.001,
        save_weights_only=False,
    ):
        self.monitor = monitor
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.save_weights_only = save_weights_only
        self.model_path = model_path
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

        if self.monitor.startswith("train_"):
            self.model_state = "train"
            self.monitor_value = self.monitor[len("train_"):]
        elif self.monitor.startswith("valid_"):
            self.model_state = "valid"
            self.monitor_value = self.monitor[len("valid_"):]
        else:
            raise Exception("monitor must start with train_ or valid_")

        self.valid_targets = valid_dataframe.answer_text.values
        self.valid_data_loader = valid_data_loader
        self.tokenizer = tokenizer
        valid_dataframe = valid_dataframe.drop(["answer_text", "answer_start"], axis=1)
        self.valid_dataset = Dataset.from_pandas(valid_dataframe)
        self.valid_features = self.valid_dataset.map(
            partial(
                prepare_validation_features,
                tokenizer=self.tokenizer,
                pad_on_right=pad_on_right,
                max_length=max_length,
                doc_stride=doc_stride,
            ),
            batched=True,
            remove_columns=self.valid_dataset.column_names,
        )

    def on_epoch_end(self, model):
        model.eval()
        tk0 = tqdm(self.valid_data_loader, total=len(self.valid_data_loader))
        start_logits = []
        end_logits = []

        for _, data in enumerate(tk0):
            with torch.no_grad():
                for key, value in data.items():
                    data[key] = value.to("cuda")
                output, _, _ = model(**data)
                start = output[0].detach().cpu().numpy()
                end = output[1].detach().cpu().numpy()
                start_logits.append(start)
                end_logits.append(end)

        start_logits = np.vstack(start_logits)
        end_logits = np.vstack(end_logits)

        valid_preds = postprocess_qa_predictions(
            self.valid_dataset, self.tokenizer, self.valid_features, (start_logits, end_logits)
        )
        epoch_score = np.mean([jaccard(x, y)
                              for x, y in zip(self.valid_targets, valid_preds.values())])
        print(f"Jaccard Score = {epoch_score}")
        model.train()
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print("EarlyStopping counter: {} out of {}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                model.model_state = enums.ModelState.END
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print("Validation score improved ({} --> {}). Saving model!".format(self.val_score, epoch_score))
            model.save(self.model_path, weights_only=self.save_weights_only)
        self.val_score = epoch_score


if __name__ == "__main__":
    config = parse_args()
    # seed_everything(config.seed)
    output_path = f"{config.model.replace('/','-')}_fold_{config.fold}.bin"
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    pad_on_right = tokenizer.padding_side == "right"
    df = pd.read_csv(config.data_path)
    df_train = df[df.kfold != config.fold].reset_index(drop=True)
    external_data = get_extra_data()
    df_valid = df[df.kfold == config.fold].reset_index(drop=True)
    cols = ["context", "question", "answer_text", "answer_start"]
    external_data = external_data[cols].reset_index(drop=True)
    df_train = df_train[cols].reset_index(drop=True)
    df_train = pd.concat([df_train, external_data], axis=0).reset_index(drop=True)
    df_train["answers"] = df_train[["answer_start", "answer_text"]].apply(convert_answers, axis=1)
    df_valid["answers"] = df_valid[["answer_start", "answer_text"]].apply(convert_answers, axis=1)
    train_data = Dataset.from_pandas(df_train)
    train_features = train_data.map(
        partial(
            prepare_train_features,
            tokenizer=tokenizer,
            pad_on_right=pad_on_right,
            max_length=config.max_length,
            doc_stride=config.doc_stride,
        ),
        batched=True,
        remove_columns=train_data.column_names,
    )
    valid_data = Dataset.from_pandas(df_valid)
    valid_features = valid_data.map(
        partial(
            prepare_train_features,
            tokenizer=tokenizer,
            pad_on_right=pad_on_right,
            max_length=config.max_length,
            doc_stride=config.doc_stride,
        ),
        batched=True,
        remove_columns=valid_data.column_names,
    )
    train_dataset = ChaiiDataset(train_features)
    valid_dataset = ChaiiDataset(valid_features)
    n_train_steps = int(len(train_dataset) / config.train_batch_size * config.epochs)
    model = TezChaiiModel(
        model_name=config.model,
        num_train_steps=n_train_steps,
        learning_rate=config.learning_rate,
        steps_per_epoch=len(df_train) / config.train_batch_size,
        weight_decay=config.weight_decay
    )
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.valid_batch_size,
        num_workers=4,
        shuffle=False,
    )
    es = EarlyStopping(
        monitor="valid_jaccard",
        model_path=output_path,
        valid_dataframe=df_valid,
        valid_data_loader=valid_data_loader,
        tokenizer=tokenizer,
        pad_on_right=pad_on_right,
        max_length=config.max_length,
        doc_stride=config.doc_stride,
        save_weights_only=True,
        mode="max",
    )
    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        train_collate_fn=default_data_collator,
        valid_collate_fn=default_data_collator,
        train_bs=config.train_batch_size,
        valid_bs=config.valid_batch_size,
        device="cuda",
        epochs=config.epochs,
        callbacks=[es],
        fp16=config.fp16,
        accumulation_steps=config.accumulation_steps,
    )
