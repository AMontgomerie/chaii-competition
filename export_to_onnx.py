import argparse
import gc
import torch.onnx
import torch
import onnxruntime
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--token_type_ids", dest="token_type_ids", action="store_true")
    parser.add_argument("--weights_file", type=str, required=True)
    config = parser.parse_args()
    return config


def export(
    torch_model: torch.nn.Module,
    inputs: torch.Tensor,
    fold: int,
    has_token_type_ids: bool = False
) -> None:
    onnx_model_path = f"onnx-muril/fold_{fold}/model.onnx"
    axes = {
        'input_ids': {0: 'batch', 1: 'sequence'},
        'attention_mask': {0: 'batch', 1: 'sequence'},
        'output_0': {0: 'batch', 1: 'sequence'},
        'output_1': {0: 'batch'}
    }
    input_names = ['input_ids', 'attention_mask']
    if has_token_type_ids:
        axes["token_type_ids"] = {0: 'batch', 1: 'sequence'}
        input_names.append('token_type_ids')
    torch.onnx.export(
        torch_model,
        inputs,
        onnx_model_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=input_names,
        output_names=['output_0', 'output_1'],
        dynamic_axes=axes,
        use_external_data_format=True
    )
    print(f"Exported fold {fold}!")


def validate(fold: int, inputs: torch.Tensor, torch_out: torch.Tensor) -> None:
    onnx_model_path = f"onnx-muril/fold_{fold}/model.onnx"
    ort_session = onnxruntime.InferenceSession(
        onnx_model_path,
        providers=["CUDAExecutionProvider"]
    )
    ort_inputs = {
        ort_session.get_inputs()[i].name: to_numpy(inputs[i])
        for i in range(len(ort_session.get_inputs()))
    }
    ort_outs = ort_session.run(None, ort_inputs)
    np.testing.assert_allclose(
        to_numpy(torch_out["start_logits"]),
        ort_outs[0],
        rtol=1e-03,
        atol=1e-05
    )
    np.testing.assert_allclose(
        to_numpy(torch_out["end_logits"]),
        ort_outs[1],
        rtol=1e-03,
        atol=1e-05
    )
    print(f"fold {fold} passed validation.")
    del ort_session, ort_inputs, ort_outs
    torch.cuda.empty_cache()
    gc.collect()


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


if __name__ == "__main__":
    config = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    inputs = tokenizer("Here's a test phrase.", return_tensors="pt")
    inputs = (inputs["input_ids"], inputs["attention_mask"])
    torch_model = AutoModelForQuestionAnswering.from_pretrained(config.base_model)
    torch_model.load_state_dict(torch.load(config.weights_file))
    torch_model.eval()
    torch_out = torch_model(*inputs)
    export(torch_model, inputs, config.fold, has_token_type_ids=config.token_type_ids)
    validate(config.fold, inputs, torch_out)
    del torch_model
    torch.cuda.empty_cache()
    gc.collect()
