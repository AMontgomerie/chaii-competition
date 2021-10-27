import argparse
from transformers import AutoTokenizer
import torch
import pandas as pd
from typing import Tuple
import os

from model import make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="train_folds_10.csv", required=False)
    parser.add_argument("--device", type=str, default="cuda", required=False)
    parser.add_argument("--export_type", type=str, default="trace", required=False)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model_type", type=str, default="hf", required=False)
    parser.add_argument("--save_dir", type=str, default=".", required=False)
    parser.add_argument("--weights_dir", type=str, required=True)
    return parser.parse_args()


def export_to_torchscript(
    base_model: str,
    model_type: str,
    model_weights: str,
    save_path: str,
    dummy_input: str,
    export_type: str,
    device: str = "cuda"
) -> None:
    model = make_model(
        base_model,
        model_type,
        model_weights,
        device,
        torchscript=True
    )
    model.eval()
    if export_type == "trace":
        ts_model = torch.jit.trace(model, dummy_input)
    elif export_type == "script":
        ts_model = torch.jit.script(model)
    else:
        raise ValueError(f"Unrecognised export_type: {export_type}.")
    torch.jit.save(ts_model, save_path)
    print(f"Saved {save_path}")


def get_dummy_input(
    base_model: str,
    example_text: str,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    dummy_input = tokenizer(
        example_text,
        return_tensors="pt",
        truncation=True,
        max_length=384,
        padding=True
    )
    return (
        dummy_input["input_ids"].to(device),
        dummy_input["attention_mask"].to(device)
    )


if __name__ == "__main__":
    config = parse_args()
    train_data = pd.read_csv(config.data_dir)
    example_text = train_data.loc[0].context
    dummy_input = get_dummy_input(config.base_model, example_text, config.device)
    model_weights = os.path.join(
        config.weights_dir,
        f"{config.base_model.replace('/', '-')}_fold_{config.fold}.bin"
    )
    save_path = os.path.join(
        config.save_dir,
        f"torchscript_{config.base_model.replace('/', '-')}_fold_{config.fold}.pt"
    )
    export_to_torchscript(
        config.base_model,
        config.model_type,
        model_weights,
        save_path,
        dummy_input,
        config.export_type,
        config.device
    )
