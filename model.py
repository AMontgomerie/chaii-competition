import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from utils import AverageMeter, jaccard, seed_everything
from dataclasses import dataclass

@dataclass
class ModelOutput:
    start_logits: torch.Tensor
    end_logits: torch.Tensor
    loss: torch.Tensor


class ChaiiModel(nn.Module):
    def __init__(self, model_name):
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
        
    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None):
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
    
    def _loss(self, start_logits, end_logits, start_positions, end_positions):
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