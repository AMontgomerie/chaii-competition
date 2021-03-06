import os
import math
import pandas as pd
import numpy as np
from datasets import Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, AutoTokenizer, get_scheduler
from transformers.data.data_collator import default_data_collator
from tqdm import tqdm
import collections
from typing import Tuple
from datasets.utils import disable_progress_bar

from model import make_model
from utils import AverageMeter, jaccard, seed_everything, parse_args_train
from processing import (
    prepare_train_features,
    prepare_validation_features,
    postprocess_qa_predictions,
    convert_answers,
    filter_pred_strings
)
from data import get_extra_data

os.environ["TOKENIZERS_PARALLELISM"] = "false"

disable_progress_bar()


class Trainer:
    def __init__(
        self,
        model_name: str,
        fold: int,
        train_set: Dataset,
        valid_set: Dataset,
        tokenizer: AutoTokenizer,
        model_type: str = "hf",
        learning_rate: float = 3e-5,
        weight_decay: float = 0.1,
        epochs: int = 1,
        train_batch_size: int = 4,
        valid_batch_size: int = 32,
        eval_step: int = 1500,
        max_length: int = 384,
        max_answer_length: int = 30,
        doc_stride: int = 128,
        save_path: str = "output",
        scheduler: str = "cosine",
        warmup: float = 0.05,
        adam_epsilon: float = 1e-8,
        early_stopping: int = 3,
        fp16: bool = False,
        accumulation_steps: int = 1,
        dataloader_workers: int = 4,
        pad_on_right: bool = True,
        model_weights: str = None,
        pretrain: bool = False
    ) -> None:
        self.model = make_model(model_name, model_type, model_weights)
        self.model.to("cuda")
        self.fold = fold
        self.train_set = train_set
        self.valid_set = valid_set
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.max_length = max_length
        self.max_answer_length = max_answer_length
        self.doc_stride = doc_stride
        if pretrain:
            file_name = f"{model_name.replace('/', '-')}_pretrain.bin"
        else:
            file_name = f"{model_name.replace('/', '-')}_fold_{fold}.bin"
        self.save_path = os.path.join(save_path, file_name)
        self.best_jaccard = 0
        self.current_jaccard = 0
        self.early_stopping_counter = 0
        self.early_stopping_limit = early_stopping
        self.optimizer = self._make_optimizer(learning_rate, adam_epsilon, weight_decay)
        total_steps = math.ceil(len(train_set) * epochs / train_batch_size / accumulation_steps)
        warmup_steps = total_steps * warmup
        self.scheduler = get_scheduler(scheduler, self.optimizer, warmup_steps, total_steps)
        self.eval_step = eval_step
        self.accumulation_steps = accumulation_steps
        self.dataloader_workers = dataloader_workers
        self.fp16 = fp16
        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        self.pad_on_right = pad_on_right
        self._prepare_validation_features()
        self.loss_score = AverageMeter()

    def train(self) -> None:
        self.model.train()
        dataloader = DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.dataloader_workers,
            pin_memory=True,
            collate_fn=default_data_collator
        )
        for epoch in range(1, self.epochs + 1):
            self.loss_score.reset()
            self.optimizer.zero_grad()
            end = False

            with tqdm(total=len(dataloader), unit="batches") as tepoch:
                tepoch.set_description(f"epoch {epoch}")

                for step, batch in enumerate(dataloader):
                    batch = self._to_device(batch)
                    if self.fp16:
                        with torch.cuda.amp.autocast():
                            output = self.model(**batch)
                        loss = output.loss
                        loss = loss / self.accumulation_steps
                        self.scaler.scale(loss).backward()
                    else:
                        output = self.model(**batch)
                        loss = output.loss
                        loss = loss / self.accumulation_steps
                        loss.backward()
                    if (step + 1) % self.accumulation_steps == 0:
                        if self.fp16:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
                        self.optimizer.zero_grad()
                        if self.scheduler:
                            self.scheduler.step()
                    self.loss_score.update(output.loss.item(), self.train_batch_size)
                    if step != 0 and step % self.eval_step == 0:
                        end = self.evaluate()
                        self.loss_score.reset()
                    if end:
                        break
                    lr = f"{self.scheduler.get_last_lr()[0]:0.3e}"
                    metrics = {"loss": self.loss_score.avg, "lr": lr}
                    tepoch.set_postfix(metrics)
                    tepoch.update(1)

            if not end:
                end = self.evaluate()
            if end:
                break
            print(f"End of epoch {epoch} | Best Validation Jaccard {self.best_jaccard}")

    def evaluate(self) -> bool:
        predictions = self.predict(self.valid_features_small)
        self.current_jaccard = self._calculate_validation_jaccard(
            self.valid_set,
            self.valid_features,
            predictions,
        )
        if self.current_jaccard > self.best_jaccard:
            print(f"Score improved from {self.best_jaccard} to {self.current_jaccard}.")
            self.best_jaccard = self.current_jaccard
            torch.save(self.model.state_dict(), self.save_path)
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
            print(
                f"{self.current_jaccard} is not an improvement."
                f" Early stopping {self.early_stopping_counter}/{self.early_stopping_limit}"
            )
        if self.early_stopping_counter >= self.early_stopping_limit:
            print("Early stopping limit reached. Terminating.")
            return True
        else:
            return False

    @torch.no_grad()
    def predict(self, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        dataloader = DataLoader(
            dataset,
            batch_size=self.valid_batch_size,
            shuffle=False,
            num_workers=self.dataloader_workers,
            pin_memory=True,
            collate_fn=default_data_collator
        )
        start_logits = []
        end_logits = []
        for batch in dataloader:
            batch = self._to_device(batch)
            output = self.model(**batch)
            start_logits.append(output.start_logits.cpu().numpy())
            end_logits.append(output.end_logits.cpu().numpy())
        return np.vstack(start_logits), np.vstack(end_logits)

    def _prepare_validation_features(self):
        self.valid_features = self.valid_set.map(
            prepare_validation_features,
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "pad_on_right": self.pad_on_right,
                "max_length": self.max_length,
                "doc_stride": self.doc_stride
            },
            batched=True,
            remove_columns=self.valid_set.column_names
        )
        self.valid_features_small = self.valid_features.map(
            lambda example: example, remove_columns=['example_id', 'offset_mapping']
        )
        self.valid_features_small.set_format(
            type='torch',
            columns=["input_ids", "attention_mask"]
        )
        self.example_id_to_index = {k: i for i, k in enumerate(self.valid_set["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(self.valid_features):
            features_per_example[self.example_id_to_index[feature["example_id"]]].append(i)

    def _calculate_validation_jaccard(
        self,
        dataset: Dataset,
        features: Dataset,
        raw_predictions: Tuple[np.ndarray, np.ndarray],
    ) -> float:
        final_predictions = postprocess_qa_predictions(
            dataset,
            features,
            raw_predictions,
            self.tokenizer,
            self.max_answer_length
        )
        references = [
            {"id": ex["id"], "answer": ex["answers"]['text'][0]}
            for ex in dataset
        ]
        res = pd.DataFrame(references)
        res['prediction'] = final_predictions.PredictionString
        res["prediction"] = filter_pred_strings(res.prediction)
        res['jaccard'] = res[['answer', 'prediction']].apply(jaccard, axis=1)
        return res.jaccard.mean()

    def _to_device(self, batch, device="cuda"):
        for k, v in batch.items():
            batch[k] = v.to(device)
        return batch

    def _make_optimizer(
        self,
        learning_rate: float,
        adam_epsilon: float,
        weight_decay: float
    ) -> AdamW:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            eps=adam_epsilon,
            correct_bias=True
        )


if __name__ == "__main__":
    config = parse_args_train()
    if config.fold is None:
        raise ValueError("No fold chosen. Use --fold.")
    seed_everything(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    pad_on_right = tokenizer.padding_side == "right"
    data = pd.read_csv(config.data_path, encoding="utf-8")
    train = data.loc[data.kfold != config.fold]
    valid = data.loc[data.kfold == config.fold]
    if config.use_extra_data:
        extra_data = get_extra_data(config.extra_data_dir)
        train = pd.concat([train, extra_data])
    train['answers'] = train.loc[:, ['answer_start', 'answer_text']].apply(
        convert_answers,
        axis=1
    )
    valid['answers'] = valid.loc[:, ['answer_start', 'answer_text']].apply(
        convert_answers,
        axis=1
    )
    train_dataset = Dataset.from_pandas(train)
    valid_dataset = Dataset.from_pandas(valid)
    tokenized_train_ds = train_dataset.map(
        prepare_train_features,
        batched=True,
        remove_columns=train_dataset.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": config.max_length,
            "doc_stride": config.doc_stride,
            "pad_on_right": pad_on_right
        }
    )
    tokenized_train_ds.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions']
    )
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    trainer = Trainer(
        config.model,
        config.fold,
        tokenized_train_ds,
        valid_dataset,
        tokenizer,
        model_weights=config.model_weights,
        model_type=config.model_type,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        epochs=config.epochs,
        train_batch_size=config.train_batch_size,
        valid_batch_size=config.valid_batch_size,
        eval_step=config.eval_step,
        max_length=config.max_length,
        max_answer_length=config.max_answer_length,
        doc_stride=config.doc_stride,
        save_path=config.save_path,
        scheduler=config.scheduler,
        warmup=config.warmup,
        adam_epsilon=config.adam_epsilon,
        early_stopping=config.early_stopping,
        fp16=config.fp16,
        accumulation_steps=config.accumulation_steps,
        dataloader_workers=config.dataloader_workers,
        pad_on_right=pad_on_right
    )
    trainer.train()
