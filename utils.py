import os
import numpy as np
import random
import torch
import argparse


class AverageMeter(object):
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def jaccard(row):
    str1 = row[0]
    str2 = row[1]
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_args_train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accumulation_steps", type=int, default=1, required=False)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, required=False)
    parser.add_argument("--dataloader_workers", type=int, default=8, required=False)
    parser.add_argument("--data_path", type=str, default="train_folds.csv", required=False)
    parser.add_argument("--doc_stride", type=int, default=128, required=False)
    parser.add_argument("--early_stopping", type=int, default=3, required=False)
    parser.add_argument("--epochs", type=int, default=1, required=False)
    parser.add_argument("--eval_on_first_step", dest="eval_on_first_step", action="store_true")
    parser.add_argument("--evals_per_epoch", type=int, default=0, required=False)
    parser.add_argument("--extra_data_dir", type=str, default="extra_data", required=False)
    parser.add_argument("--fold", type=int, required=False)
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=3e-5, required=False)
    parser.add_argument("--max_answer_length", type=int, default=30, required=False)
    parser.add_argument("--max_length", type=int, default=384, required=False)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_weights", type=str, default=None, required=False)
    parser.add_argument("--model_type", type=str, default="default", required=False)
    parser.add_argument("--save_path", type=str, default="../output", required=False)
    parser.add_argument("--scheduler", type=str, default="cosine", required=False)
    parser.add_argument("--seed", type=int, default=0, required=False)
    parser.add_argument("--valid_batch_size", type=int, default=32, required=False)
    parser.add_argument("--train_batch_size", type=int, default=4, required=False)
    parser.add_argument("--use_extra_data", dest="use_extra_data", action="store_true")
    parser.add_argument("--valid_data_path", type=str, default=None, required=False)
    parser.add_argument("--warmup", type=float, default=0.05, required=False)
    parser.add_argument("--weight_decay", type=float, default=0.0, required=False)
    return parser.parse_args()


def parse_args_inference() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default="deepset/xlm-roberta-large-squad2",
        required=False
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default=None,
        required=False
    )
    parser.add_argument("--device", type=str, default="cuda", required=False)
    parser.add_argument("--doc_stride", type=int, default=128, required=False)
    parser.add_argument("--input_data", type=str, default="train_folds.csv", required=False)
    parser.add_argument("--max_answer_length", type=int, default=30, required=False)
    parser.add_argument("--max_length", type=int, default=384, required=False)
    parser.add_argument("--model_type", type=str, default="default", required=False)
    parser.add_argument("--model_weights_dir", type=str, required=True)
    parser.add_argument("--num_folds", type=int, default=5, required=False)
    parser.add_argument("--save_dir", type=str, default="", required=False)
    parser.add_argument("--batch_size", type=int, default=64, required=False)
    return parser.parse_args()
