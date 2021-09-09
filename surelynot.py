from data import get_extra_data
from datasets import Dataset
from tez.callbacks import Callback
from tez import enums
from tqdm import tqdm
import collections
from transformers import default_data_collator
from functools import partial
import pandas as pd
import os
import random
from transformers import AutoTokenizer
import argparse
import tez
import numpy as np
import torch.nn as nn
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
import transformers
from sklearn import metrics
import sys
sys.path.append("../input/tez-lib/")


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


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


def convert_answers(r):
    start = r[0]
    text = r[1]
    return {"answer_start": [start], "text": [text]}


def prepare_train_features(examples, tokenizer, pad_on_right, max_length, doc_stride):
    # ref: https://github.com/huggingface/notebooks/blob/master/examples/question_answering.ipynb
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not(
                    offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >=
                    end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def prepare_validation_features(examples, tokenizer, pad_on_right, max_length, doc_stride):
    # ref: https://github.com/huggingface/notebooks/blob/master/examples/question_answering.ipynb
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def postprocess_qa_predictions(
        examples, tokenizer, features, raw_predictions, n_best_size=20, max_answer_length=30,
        squad_v2=False):
    # ref: https://github.com/huggingface/notebooks/blob/master/examples/question_answering.ipynb
    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    print(
        f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None  # Only used if squad_v2 is True.
        valid_answers = []

        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char:end_char],
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}

        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        if not squad_v2:
            predictions[example["id"]] = best_answer["text"]
        else:
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            predictions[example["id"]] = answer

    return predictions


class ChaiiModel(tez.Model):
    def __init__(self, model_name, num_train_steps, steps_per_epoch, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.model_name = model_name
        self.num_train_steps = num_train_steps
        self.step_scheduler_after = "batch"

        hidden_dropout_prob: float = 0.0
        layer_norm_eps: float = 1e-7

        config = transformers.AutoConfig.from_pretrained(model_name)
        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
            }
        )
        self.transformer = transformers.AutoModel.from_pretrained(model_name, config=config)
        self.output = nn.Linear(config.hidden_size, config.num_labels)

    def fetch_optimizer(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(optimizer_parameters, lr=self.learning_rate)
        return opt

    def fetch_scheduler(self):
        sch = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_train_steps,
        )
        return sch

    def loss(self, start_logits, end_logits, start_positions, end_positions):
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)

        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)
        loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        return total_loss

    def monitor_metrics(self, outputs, targets):
        return {"jaccard": None}

    def forward(self, ids, mask, token_type_ids=None, start_positions=None, end_positions=None):
        if token_type_ids:
            transformer_out = self.transformer(ids, mask, token_type_ids)
        else:
            transformer_out = self.transformer(ids, mask)
        sequence_output = transformer_out[0]
        logits = self.output(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self.loss(start_logits, end_logits, start_positions, end_positions)

        return (start_logits, end_logits), loss, {}


class ChaiiDataset:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if "token_type_ids" in self.data[item]:
            return {
                "ids": torch.tensor(self.data[item]["input_ids"], dtype=torch.long),
                "mask": torch.tensor(self.data[item]["attention_mask"], dtype=torch.long),
                "token_type_ids": torch.tensor(self.data[item]["token_type_ids"], dtype=torch.long),
                "start_positions": torch.tensor(self.data[item]["start_positions"], dtype=torch.long),
                "end_positions": torch.tensor(self.data[item]["end_positions"], dtype=torch.long),
            }
        return {
            "ids": torch.tensor(self.data[item]["input_ids"], dtype=torch.long),
            "mask": torch.tensor(self.data[item]["attention_mask"], dtype=torch.long),
            "start_positions": torch.tensor(self.data[item]["start_positions"], dtype=torch.long),
            "end_positions": torch.tensor(self.data[item]["end_positions"], dtype=torch.long),
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class args:
    # NOTE: you need to train for all folds from 0 to 9
    fold = 0
    model = "deepset/xlm-roberta-large-squad2"
    batch_size = 2
    max_len = 384
    doc_stride = 128
    learning_rate = 1e-5
    epochs = 20
    accumulation_steps = 8
    max_answer_length = 30
    seed = 0


seed_everything(args.seed)

output_path = f"{args.model.replace('/',':')}__fold_{args.fold}.bin"

tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
pad_on_right = tokenizer.padding_side == "right"

df = pd.read_csv("train_folds.csv")
df_train = df[df.kfold != args.fold].reset_index(drop=True)

external_data = get_extra_data()
external_data = external_data.drop_duplicates(keep="last")
external_data = external_data.reset_index(drop=True)

df_valid = df[df.kfold == args.fold].reset_index(drop=True)

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
        max_length=args.max_len,
        doc_stride=args.doc_stride,
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
        max_length=args.max_len,
        doc_stride=args.doc_stride,
    ),
    batched=True,
    remove_columns=valid_data.column_names,
)

train_dataset = ChaiiDataset(train_features)
valid_dataset = ChaiiDataset(valid_features)

n_train_steps = int(len(train_dataset) / args.batch_size * args.epochs)
model = ChaiiModel(
    model_name=args.model,
    num_train_steps=n_train_steps,
    learning_rate=args.learning_rate,
    steps_per_epoch=len(df_train) / args.batch_size,
)

valid_data_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=64,
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
    max_length=args.max_len,
    doc_stride=args.doc_stride,
    save_weights_only=True,
    mode="max",
)
model.fit(
    train_dataset,
    valid_dataset=valid_dataset,
    train_collate_fn=default_data_collator,
    valid_collate_fn=default_data_collator,
    train_bs=args.batch_size,
    valid_bs=64,
    device="cuda",
    epochs=args.epochs,
    callbacks=[es],
    fp16=False,
    accumulation_steps=args.accumulation_steps,
)
