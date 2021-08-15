import argparse
import pandas as pd
import numpy as np
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoTokenizer,
    AutoModelForQuestionAnswering
)
from tqdm import tqdm
import collections
from typing import Tuple
from datasets.utils.logging import set_verbosity_error

from utils import AverageMeter, jaccard, seed_everything
from preprocessing import (
    convert_answers,
    prepare_train_features,
    prepare_validation_features,
    postprocess_qa_predictions,
    convert_answers
)

set_verbosity_error()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accumulation_steps", type=int, default=8, required=False)
    parser.add_argument("--data_path", type=str, default="train_folds.csv", required=False)
    parser.add_argument("--doc_stride", type=int, default=128, required=False)
    parser.add_argument("--epochs", type=int, default=1, required=False)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, default=3e-5, required=False)
    parser.add_argument("--max_answer_length", type=int, default=30, required=False)
    parser.add_argument("--max_length", type=int, default=384, required=False)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="output", required=False)
    parser.add_argument("--seed", type=int, default=0, required=False)
    parser.add_argument("--test_batch_size", type=int, default=32, required=False)
    parser.add_argument("--train_batch_size", type=int, default=4, required=False)
    parser.add_argument("--warmup", type=float, default=0.1, required=False)
    parser.add_argument("--weight_decay", type=float, default=0.1, required=False)
    return parser.parse_args()


class Trainer:
    def __init__(
        self,
        model_name: str,
        train_set: Dataset,
        valid_set: Dataset,
        tokenizer: AutoTokenizer,
        learning_rate: float = 3e-5,
        weight_decay: float = 0.1,
        epochs: int = 1,
        train_batch_size: int = 4,
        valid_batch_size: int = 32,
        logs_per_epoch: int = 10
    ) -> None:
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.to("cuda")
        self.train_set = train_set
        self.valid_set = valid_set
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.logs_per_epoch = logs_per_epoch
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

    def train(self) -> None:
        self.model.train()
        dataloader = DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            shuffle=True
        )
        for epoch in range(1, self.epochs + 1):
            loss_score = AverageMeter()

            with tqdm(total=len(dataloader), unit="batches") as tepoch:
                tepoch.set_description(f"epoch {epoch}")

                for batch in dataloader:
                    batch = self._to_device(batch)
                    output = self.model(**batch)
                    loss = output.loss
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # scheduler.step()
                    loss_score.update(loss.detach().item(), self.train_batch_size)
                    tepoch.set_postfix(loss=loss_score.avg)
                    tepoch.update(1)

            valid_jaccard = self.evaluate()
            print(f"End of epoch {epoch} | Validation Jaccard {valid_jaccard}")

    def evaluate(self) -> float:
        predictions = self.predict(self.valid_set)
        return self._calculate_validation_jaccard(predictions)

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
        raw_predictions: Tuple[np.ndarray, np.ndarray]
    ) -> float:
        validation_features = valid_dataset.map(
            prepare_validation_features,
            batched=True,
            remove_columns=valid_dataset.column_names
        )
        example_id_to_index = {k: i for i, k in enumerate(valid_dataset["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(validation_features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        final_predictions = postprocess_qa_predictions(
            valid_dataset,
            validation_features,
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
    data = pd.read_csv(config.data_path)
    train = data[data.kfold != config.fold]
    valid = data[data.kfold == config.fold]
    train = train.sample(frac=1, random_state=config.seed)
    train['answers'] = train[['answer_start', 'answer_text']].apply(
        convert_answers,
        axis=1
    )
    train_dataset = Dataset.from_pandas(train)
    valid_dataset = Dataset.from_pandas(valid)
    tokenized_train_ds = train_dataset.map(
        prepare_train_features,
        batched=True,
        remove_columns=train_dataset.column_names,
        tokenizer=tokenizer,
        config=config,
        pad_on_right=pad_on_right,
    )
    tokenized_valid_ds = valid_dataset.map(
        prepare_validation_features,
        batched=True,
        tokenizer=tokenizer,
        config=config,
        remove_columns=train_dataset.column_names,
        pad_on_right=pad_on_right
    )
    tokenized_train_ds.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions']
    )
    tokenized_valid_ds.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions']
    )
    trainer = Trainer(
        config.checkpoint,
        tokenized_train_ds,
        tokenized_valid_ds,
        tokenizer
    )
    trainer.train()
