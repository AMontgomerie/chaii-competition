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
    get_cosine_schedule_with_warmup
)
from tqdm import tqdm
import collections
from typing import Tuple

from utils import AverageMeter, jaccard, seed_everything
from preprocessing import (
    convert_answers,
    prepare_train_features,
    prepare_validation_features,
    postprocess_qa_predictions,
    convert_answers
)
from datasets.utils import disable_progress_bar

disable_progress_bar()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="train_folds.csv", required=False)
    parser.add_argument("--doc_stride", type=int, default=128, required=False)
    parser.add_argument("--epochs", type=int, default=1, required=False)
    parser.add_argument("--evals_per_epoch", type=int, default=0, required=False)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, default=3e-5, required=False)
    parser.add_argument("--max_answer_length", type=int, default=30, required=False)
    parser.add_argument("--max_length", type=int, default=384, required=False)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="output", required=False)
    parser.add_argument("--scheduler", type=str, default="cosine", required=False)
    parser.add_argument("--seed", type=int, default=0, required=False)
    parser.add_argument("--valid_batch_size", type=int, default=32, required=False)
    parser.add_argument("--train_batch_size", type=int, default=4, required=False)
    parser.add_argument("--warmup", type=float, default=0.05, required=False)
    parser.add_argument("--weight_decay", type=float, default=0.0, required=False)
    return parser.parse_args()


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
        doc_stride: int = 128,
        save_path: str = "output",
        scheduler: str = "cosine",
        warmup: float = 0.05
    ) -> None:
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
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
        self.doc_stride = doc_stride
        self.save_path = save_path
        self.best_jaccard = 0
        self.current_jaccard = 0
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        total_steps = len(train_set)//train_batch_size
        if scheduler == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=total_steps*warmup,
                num_training_steps=total_steps
            )
        else:
            self.scheduler = None

    def train(self) -> None:
        self.model.train()
        dataloader = DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            shuffle=True
        )
        best_jaccard = 0
        for epoch in range(1, self.epochs + 1):
            loss_score = AverageMeter()

            with tqdm(total=len(dataloader), unit="batches") as tepoch:
                tepoch.set_description(f"epoch {epoch}")

                for step, batch in enumerate(dataloader):
                    batch = self._to_device(batch)
                    output = self.model(**batch)
                    loss = output.loss
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
                    loss_score.update(loss.detach().item(), self.train_batch_size)
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
            predictions
        )
        if self.current_jaccard > self.best_jaccard:
            self.best_jaccard = self.current_jaccard
            self.model.save_pretrained(self.save_path)

    @torch.no_grad()
    def predict(self, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        dataloader = DataLoader(
            dataset,
            batch_size=self.valid_batch_size,
            shuffle=False
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
        raw_predictions: Tuple[np.ndarray, np.ndarray]
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
            config.max_answer_length
        )
        references = [
            {"id": ex["id"], "answer": ex["answers"]['text'][0]}
            for ex in valid_dataset
        ]
        res = pd.DataFrame(references)
        res['prediction'] = res['id'].apply(lambda r: final_predictions[r])
        res['jaccard'] = res[['answer', 'prediction']].apply(jaccard, axis=1)
        return res.jaccard.mean()

    def _to_device(self, batch, device="cuda"):
        for k, v in batch.items():
            batch[k] = v.to(device)
        return batch


if __name__ == "__main__":
    config = parse_args()
    seed_everything(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    pad_on_right = tokenizer.padding_side == "right"
    data = pd.read_csv(config.data_path, encoding="utf-8")
    train = data[data.kfold != config.fold].reset_index()
    valid = data[data.kfold == config.fold].reset_index()
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
        doc_stride=config.doc_stride,
        save_path=full_save_path,
        scheduler=config.scheduler,
        warmup=config.warmup
    )
    trainer.train()
    tokenizer.save_pretrained(full_save_path)
    
