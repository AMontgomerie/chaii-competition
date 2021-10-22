import argparse
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import pandas as pd
from typing import Tuple
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="train_folds_10.csv", required=False)
    parser.add_argument("--num_folds", type=int, default=10, required=False)
    parser.add_argument("--save_dir", type=str, default=".", required=False)
    parser.add_argument("--weights_dir", type=str, required=True)
    return parser.parse_args()


def export_to_torchscript(
    base_model: str,
    model_weights: str,
    save_path: str,
    dummy_input: str
) -> None:
    model = AutoModelForQuestionAnswering.from_pretrained(base_model, torchscript=True)
    model.load_state_dict(torch.load(model_weights, map_location=torch.device('cpu')))
    model.eval()
    traced_model = torch.jit.trace(model, dummy_input)
    torch.jit.save(traced_model, save_path)
    print(f"Saved {save_path}")


def get_dummy_input(base_model: str, example_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    dummy_input = tokenizer(
        example_text,
        return_tensors="pt",
        truncation=True,
        max_length=384,
        padding=True
    )
    return (
        dummy_input["input_ids"],
        dummy_input["attention_mask"]
    )


if __name__ == "__main__":
    config = parse_args()
    train_data = pd.read_csv(config.data_dir)
    example_text = train_data.loc[0].context
    dummy_input = get_dummy_input(config.base_model, example_text)
    for fold in range(10):
        model_weights = os.path.join(config.weights_dir, f"{config.base_model}_fold_{fold}.bin")
        save_path = os.path.join(config.save_dir, f"torchscript_{config.base_model}_fold_{fold}.pt")
        export_to_torchscript(config.base_model, model_weights, save_path, dummy_input)
