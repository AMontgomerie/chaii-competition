import gc
from transformers import AutoTokenizer
from datasets.utils import disable_progress_bar
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
import pandas as pd
import numpy as np
import os
import onnxruntime

from utils import parse_args_inference
from processing import prepare_validation_features, postprocess_qa_predictions

os.environ["TOKENIZERS_PARALLELISM"] = "false"

disable_progress_bar()


@torch.no_grad()
def predict(
    ort_session,
    dataset: Dataset,
    batch_size: int = 64,
    workers: int = 4
) -> np.ndarray:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    start_logits = []
    end_logits = []
    for batch in dataloader:
        output = get_ort_model_prediction(ort_session, batch)
        start_logits.append(output["start_logits"])
        end_logits.append(output["end_logits"])
    return np.vstack(start_logits), np.vstack(end_logits)


def get_ort_model_prediction(session, inputs):
    inputs = (
        inputs["input_ids"].numpy(),
        inputs["attention_mask"].numpy(),
    )
    ort_inputs = {
        session.get_inputs()[i].name: inputs[i]
        for i in range(len(session.get_inputs()))
    }
    ort_outs = session.run(None, ort_inputs)
    return {
        "start_logits": ort_outs[0],
        "end_logits": ort_outs[1]
    }


def get_ort_model(onnx_model_path):
    ort_session = onnxruntime.InferenceSession(
        onnx_model_path,
        providers=["CUDAExecutionProvider"]
    )
    return ort_session


if __name__ == "__main__":
    config = parse_args_inference()
    data = pd.read_csv(config.input_data)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    dataset = Dataset.from_pandas(data)
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

    if len(config.select_folds) > 0:
        folds = [int(fold) for fold in config.select_folds]
    else:
        folds = range(config.num_folds)

    for fold in folds:
        print(f"Generating predictions for fold {fold}")
        if config.model_name:
            filename = f"{config.model_name.replace('/', '-')}_fold_{fold}"
        else:
            filename = f"{config.base_model.replace('/', '-')}_fold_{fold}"
        onnx_model_path = f"{config.model_weights_dir}/fold_{fold}/model.onnx"
        ort_model = get_ort_model(onnx_model_path)
        start_logits, end_logits = predict(
            ort_model,
            input_dataset,
            config.batch_size,
            config.dataloader_workers
        )
        pred_df = postprocess_qa_predictions(
            dataset,
            tokenized_dataset,
            (start_logits, end_logits),
            tokenizer
        )
        pred_df.to_csv(f"{filename}.csv", index=False)
        np.save(f"{filename}_start_logits.npy", start_logits)
        np.save(f"{filename}_end_logits.npy", end_logits)
        del ort_model
        gc.collect()
        torch.cuda.empty_cache()
