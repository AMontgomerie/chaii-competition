import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AdamW, get_linear_schedule_with_warmup
from dataclasses import dataclass
import tez


@dataclass
class ModelOutput:
    start_logits: torch.Tensor
    end_logits: torch.Tensor
    loss: torch.Tensor


class ChaiiModel(nn.Module):
    def __init__(self, model_name: str) -> None:
        super(ChaiiModel, self).__init__()
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


class TezChaiiModel(tez.Model):
    def __init__(self, model_name, num_train_steps, steps_per_epoch, learning_rate, weight_decay):
        super().__init__()
        self.learning_rate = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.model_name = model_name
        self.num_train_steps = num_train_steps
        self.step_scheduler_after = "batch"
        self.weight_decay

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

    def fetch_optimizer(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
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
