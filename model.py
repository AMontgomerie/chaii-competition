import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForQuestionAnswering
from dataclasses import dataclass


@dataclass
class ModelOutput:
    start_logits: torch.Tensor
    end_logits: torch.Tensor
    loss: torch.Tensor


class AbhishekModel(nn.Module):
    def __init__(self, model_name: str) -> None:
        super(AbhishekModel, self).__init__()
        hidden_dropout_prob: float = 0.0
        layer_norm_eps: float = 1e-7
        config = AutoConfig.from_pretrained(model_name)
        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
            }
        )
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        self.output = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        start_positions: torch.Tensor = None,
        end_positions: torch.Tensor = None
    ) -> ModelOutput:
        transformer_out = self.transformer(input_ids, attention_mask)
        sequence_output = transformer_out[0]
        logits = self.output(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self._loss(start_logits, end_logits, start_positions, end_positions)

        return ModelOutput(start_logits=start_logits, end_logits=end_logits, loss=loss)

    def _loss(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        start_positions: torch.Tensor,
        end_positions: torch.Tensor
    ):
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


class TorchModel(nn.Module):
    def __init__(self, model_name: str):
        super(TorchModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.xlm_roberta = AutoModel.from_pretrained(model_name, config=self.config)
        self.qa_outputs = nn.Linear(self.config.hidden_size, 2)
        self._init_weights(self.qa_outputs)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        start_positions: torch.Tensor = None,
        end_positions: torch.Tensor = None
    ):
        outputs = self.xlm_roberta(input_ids, attention_mask)
        sequence_output = outputs[0]
        qa_logits = self.qa_outputs(sequence_output)

        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self._loss_fn(start_logits, end_logits, start_positions, end_positions)

        return ModelOutput(start_logits=start_logits, end_logits=end_logits, loss=loss)

    def _loss_fn(
        self,
        start_preds: torch.Tensor,
        end_preds: torch.Tensor,
        start_labels: torch.Tensor,
        end_labels: torch.Tensor
    ):
        start_loss = nn.CrossEntropyLoss(ignore_index=-1)(start_preds, start_labels)
        end_loss = nn.CrossEntropyLoss(ignore_index=-1)(end_preds, end_labels)
        total_loss = (start_loss + end_loss) / 2
        return total_loss


class TTSModel(TorchModel):
    def __init__(self, model_name: str):
        super(TTSModel, self).__init__(model_name)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        outputs = self.xlm_roberta(input_ids, attention_mask)
        sequence_output = outputs[0]
        qa_logits = self.qa_outputs(sequence_output)

        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits


def make_model(
    model_name: str,
    model_type: str = "hf",
    model_weights: str = None,
    device: str = "cuda"
) -> nn.Module:
    if model_type == "hf":
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    elif model_type == "abhishek":
        model = AbhishekModel(model_name)
    elif model_type == "torch":
        model = TorchModel(model_name)
    else:
        raise ValueError(f"{model_type} is not a recognised model type.")
    if model_weights:
        print(f"Loading weights from {model_weights}")
        model.load_state_dict(torch.load(model_weights, map_location=torch.device(device)))
    model.to(device)
    return model
