from transformers.models.switch_transformers.modeling_switch_transformers import (
    SwitchTransformersSparseMLP,
)
from transformers.models.bert.modeling_bert import (
    BertModel,
    BertLayer,
    BertEncoder,
    BertAttention,
    BertIntermediate,
    BertOutput,
    BertConfig
)

from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEBlock(nn.Module):
    def __init__(self, config, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        hidden = config.hidden_size
        inter = config.hidden_size

        self.gate = nn.Linear(hidden, num_experts)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, inter),
                nn.GELU(),
                nn.Linear(inter, hidden),  # ключ! выход всегда hidden_size
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        B, S, H = x.size()
        logits = self.gate(x)  # [B, S, num_experts]
        gates = F.softmax(logits, dim=-1)
        indices = torch.argmax(gates, dim=-1)  # [B, S]

        output = torch.zeros_like(x)

        for expert_id in range(self.num_experts):
            # создаём маску и координаты для scatter
            mask = (indices == expert_id)  # [B, S]
            if mask.sum() == 0:
                continue
            # flatten B, S
            flat_mask = mask.view(-1)
            tokens = x.view(-1, H)[flat_mask]  # [N, hidden]
            processed = self.experts[expert_id](tokens)  # [N, hidden]

            # scatter обратно в правильную форму
            output.view(-1, H)[flat_mask] = processed

        return output  # [B, S, hidden_size] корректно


class BertMoELayer(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = BertAttention(config)
        self.intermediate = MoEBlock(
            config,
            num_experts=getattr(config, "num_experts", 4),
            hidden_size=config.hidden_size,       # output MoE = hidden_size
            intermediate_size=config.hidden_size  # чтобы BertOutput не ломался
        )
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False
    ):
        # Attention
        attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions
        )
        attention_output = attention_outputs[0]

        # MoE
        intermediate_output = self.intermediate(attention_output)

        # Output
        layer_output = self.output(intermediate_output, attention_output)

        outputs = (layer_output,)
        if output_attentions:
            outputs += (attention_outputs[1],)
        return outputs


class BertMoEEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([BertMoELayer(config) for _ in range(config.num_hidden_layers)])

class BertMoEModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = BertMoEEncoder(config)