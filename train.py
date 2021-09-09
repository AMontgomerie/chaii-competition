import os
import argparse
import pandas as pd
import numpy as np
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import collections
from typing import Tuple

from model import ChaiiModel
from utils import AverageMeter, jaccard, seed_everything
from processing import (
    prepare_train_features,
    prepare_validation_features,
    postprocess_qa_predictions,
    convert_answers,
    filter_pred_strings
)
from datasets.utils import disable_progress_bar

disable_progress_bar()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accumulation_steps", type=int, default=1, required=False)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, required=False)
    parser.add_argument("--dataloader_workers", type=int, default=8, required=False)
    parser.add_argument("--data_path", type=str, default="train_folds.csv", required=False)
    parser.add_argument("--doc_stride", type=int, default=128, required=False)
    parser.add_argument("--early_stopping", type=int, default=3, required=False)
    parser.add_argument("--epochs", type=int, default=1, required=False)
    parser.add_argument("--evals_per_epoch", type=int, default=0, required=False)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=3e-5, required=False)
    parser.add_argument("--max_answer_length", type=int, default=30, required=False)
    parser.add_argument("--max_length", type=int, default=384, required=False)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="../output", required=False)
    parser.add_argument("--scheduler", type=str, default="cosine", required=False)
    parser.add_argument("--seed", type=int, default=0, required=False)
    parser.add_argument("--valid_batch_size", type=int, default=32, required=False)
    parser.add_argument("--train_batch_size", type=int, default=4, required=False)
    parser.add_argument("--use_extra_data", dest="use_extra_data", action="store_true")
    parser.add_argument("--warmup", type=float, default=0.05, required=False)
    parser.add_argument("--weight_decay", type=float, default=0.0, required=False)
    return parser.parse_args()


def get_extra_data(data_dir: str = "extra_data") -> pd.DataFrame:
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith(".csv")]
    datasets = [pd.read_csv(os.path.join(data_dir, f), encoding="utf-8") for f in files]
    return pd.concat(datasets)


class Trainer:
    def __init__(
        self,
        model_name: str,
        fold: int,
        train_set: Dataset,
        valid_set: Dataset,
        tokenizer: AutoTokenizer,
        learning_rate: float = 3e-5,
        weight_decay: float = 0.1,
        epochs: int = 1,
        train_batch_size: int = 4,
        valid_batch_size: int = 32,
        evals_per_epoch: int = 0,
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
        dataloader_workers: int = 1,
    ) -> None:
        self.model = ChaiiModel(model_name)
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
        self.evals_per_epoch = evals_per_epoch
        self.max_length = max_length
        self.max_answer_length = max_answer_length
        self.doc_stride = doc_stride
        self.save_path = save_path
        self.best_jaccard = 0
        self.current_jaccard = 0
        self.early_stopping_counter = 0
        self.early_stopping_limit = early_stopping
        self.optimizer = self._make_optimizer(learning_rate, adam_epsilon, weight_decay)
        total_steps = len(train_set)//train_batch_size
        warmup_steps = total_steps * warmup
        self.scheduler = self._make_scheduler(scheduler, warmup_steps, total_steps)
        self.accumulation_steps = accumulation_steps
        self.dataloader_workers = dataloader_workers
        self.fp16 = fp16
        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()

    def train(self) -> None:
        self.model.train()
        dataloader = DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.dataloader_workers,
            pin_memory=True
        )
        for epoch in range(1, self.epochs + 1):
            loss_score = AverageMeter()
            self.optimizer.zero_grad()

            with tqdm(total=len(dataloader), unit="batches") as tepoch:
                tepoch.set_description(f"epoch {epoch}")

                for step, batch in enumerate(dataloader):
                    batch = self._to_device(batch)
                    if self.fp16:
                        with torch.cuda.amp.autocast():
                            output = self.model(**batch)
                        loss = output.loss
                        self.scaler.scale(loss).backward()
                    else:
                        output = self.model(**batch)
                        loss = output.loss
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
                    loss_score.update(loss.item(), self.train_batch_size)
                    if (
                        self.evals_per_epoch > 0
                        and step != 0
                        and step % (len(dataloader) // self.evals_per_epoch) == 0
                    ):
                        self.evaluate()
                    metrics = {"loss": loss_score.avg}
                    if self.evals_per_epoch > 0:
                        metrics["jccd"] = self.current_jaccard
                    tepoch.set_postfix(metrics)
                    tepoch.update(1)

            self.evaluate()
            print(f"End of epoch {epoch} | Best Validation Jaccard {self.best_jaccard}")

            if self.early_stopping_counter >= self.early_stopping_limit:
                print("Early stopping limit reached. Terminating.")
                break

    def evaluate(self) -> float:
        valid_features = self.valid_set.map(
            prepare_validation_features,
            fn_kwargs={
                "tokenizer": tokenizer,
                "pad_on_right": pad_on_right,
                "max_length": self.max_length,
                "doc_stride": self.doc_stride
            },
            batched=True,
            remove_columns=self.valid_set.column_names
        )
        valid_features_small = valid_features.map(
            lambda example: example, remove_columns=['example_id', 'offset_mapping']
        )
        valid_features_small.set_format(
            type='torch',
            columns=["input_ids", "attention_mask"]
        )
        predictions = self.predict(valid_features_small)
        self.current_jaccard = self._calculate_validation_jaccard(
            self.valid_set,
            valid_features,
            predictions,
        )
        if self.current_jaccard > self.best_jaccard:
            self.best_jaccard = self.current_jaccard
            torch.save(self.model.state_dict(), os.path.join(self.save_path, "model.bin"))
        else:
            self.early_stopping_counter += 1
            print(f"Early stopping {self.early_stopping_counter}/{self.early_stopping_limit}")

    @torch.no_grad()
    def predict(self, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        dataloader = DataLoader(
            dataset,
            batch_size=self.valid_batch_size,
            shuffle=False,
            num_workers=self.dataloader_workers,
            pin_memory=True
        )
        start_logits = []
        end_logits = []
        for batch in dataloader:
            batch = self._to_device(batch)
            output = self.model(**batch)
            start_logits.append(output.start_logits.cpu().numpy())
            end_logits.append(output.end_logits.cpu().numpy())
        return np.vstack(start_logits), np.vstack(end_logits)

    def _calculate_validation_jaccard(
        self,
        dataset: Dataset,
        features: Dataset,
        raw_predictions: Tuple[np.ndarray, np.ndarray],
    ) -> float:
        example_id_to_index = {k: i for i, k in enumerate(dataset["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        final_predictions = postprocess_qa_predictions(
            dataset,
            features,
            raw_predictions,
            self.tokenizer,
            self.max_answer_length
        )
        references = [
            {"id": ex["id"], "answer": ex["answers"]['text'][0]}
            for ex in valid_dataset
        ]
        res = pd.DataFrame(references)
        res['prediction'] = res['id'].apply(lambda r: final_predictions[r])
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
    ):
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

    def _make_scheduler(
        self,
        scheduler_type: str,
        num_warmup_steps: int,
        num_training_steps: int
    ):
        if scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        return scheduler


if __name__ == "__main__":
    config = parse_args()
    seed_everything(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    pad_on_right = tokenizer.padding_side == "right"
    data = pd.read_csv(config.data_path, encoding="utf-8")
    train = data.loc[data.kfold != config.fold]
    valid = data.loc[data.kfold == config.fold]
    if config.use_extra_data:
        extra_data = get_extra_data()
        train = pd.concat([train, extra_data])
    train['answers'] = train[['answer_start', 'answer_text']].apply(
        convert_answers,
        axis=1
    )
    valid['answers'] = valid[['answer_start', 'answer_text']].apply(
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
    full_save_path = os.path.join(
        config.save_path,
        f"{config.model.replace('/', '-')}",
        f"fold_{config.fold}"
    )
    if not os.path.exists(full_save_path):
        os.makedirs(full_save_path)
    trainer = Trainer(
        config.model,
        config.fold,
        tokenized_train_ds,
        valid_dataset,
        tokenizer,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        epochs=config.epochs,
        train_batch_size=config.train_batch_size,
        valid_batch_size=config.valid_batch_size,
        evals_per_epoch=config.evals_per_epoch,
        max_length=config.max_length,
        max_answer_length=config.max_answer_length,
        doc_stride=config.doc_stride,
        save_path=full_save_path,
        scheduler=config.scheduler,
        warmup=config.warmup,
        adam_epsilon=config.adam_epsilon,
        early_stopping=config.early_stopping,
        fp16=config.fp16,
        accumulation_steps=config.accumulation_steps,
        dataloader_workers=config.dataloader_workers
    )
    trainer.train()
