from transformers.models.switch_transformers.modeling_switch_transformers import (
    SwitchTransformersSparseMLP,
)
from transformers.models.switch_transformers import SwitchTransformersConfig
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

class BertMoELayer(BertLayer):
    def __init__(self, bert_config, moe_config=None):
        super().__init__(bert_config)
        self.attention = BertAttention(bert_config)
        if moe_config is None:
            moe_config = SwitchTransformersConfig(
                hidden_size=bert_config.hidden_size,
                intermediate_size=bert_config.intermediate_size,
                num_experts=getattr(bert_config, "num_experts", 1),
                expert_capacity=getattr(bert_config, "expert_capacity", 32),
                top_k=getattr(bert_config, "top_k", 1),
            )
        self.intermediate = SwitchTransformersSparseMLP(moe_config)
        self.output = BertOutput(bert_config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, **kwargs):
        # Attention
        self_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            **kwargs
        )
        attention_output = self_outputs[0]

        # Intermediate (MoE)
        intermediate_output = self.intermediate(attention_output)
        if isinstance(intermediate_output, tuple):
            intermediate_output = intermediate_output[0]  # берём только Tensor

        # Output
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + self_outputs[1:]  # preserve attentions if any
        return outputs



class BertMoEEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([BertMoELayer(config) for _ in range(config.num_hidden_layers)])

class BertMoEModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = BertMoEEncoder(config)
