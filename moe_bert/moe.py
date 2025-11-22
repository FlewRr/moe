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

    def forward(self, hidden_states, *args, **kwargs):
        attention_mask = kwargs.pop("attention_mask", None)
        head_mask = kwargs.pop("head_mask", None)
        output_attentions = kwargs.pop("output_attentions", False)
        kwargs.pop("encoder_attention_mask", None)
        kwargs.pop("output_hidden_states", None)

        self_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            **kwargs  # теперь kwargs не содержит output_attentions
        )

        attention_output = self_outputs[0]

        intermediate_output = self.intermediate(attention_output)
        if isinstance(intermediate_output, tuple):
            intermediate_output = intermediate_output[0]

        batch_size = hidden_states.size(0)
        seq_len = hidden_states.size(1)
        hidden_size = hidden_states.size(2)
        top_k = getattr(self.intermediate.config, "top_k", 1)

        if intermediate_output.dim() == 2:  # [batch*top_k*seq_len, hidden_size]?
            intermediate_output = intermediate_output.view(batch_size, seq_len, hidden_size)
        elif intermediate_output.dim() == 3 and intermediate_output.size(0) == batch_size * top_k:
            intermediate_output = intermediate_output.view(batch_size, seq_len, hidden_size)
    
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + self_outputs[1:]
        return outputs


class BertMoEEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([BertMoELayer(config) for _ in range(config.num_hidden_layers)])

class BertMoEModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = BertMoEEncoder(config)
