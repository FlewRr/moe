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

class BertMoELayer(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = BertAttention(config)
        self.intermediate = self.intermediate = SwitchTransformersSparseMLP(
            config.hidden_size,                          # hidden_size
            config.intermediate_size,                    # intermediate_size
            getattr(config, "num_experts", 4),          # num_experts
            top_k=1,                                     # optional
            dropout=0.0                                  # optional
        )
        self.output = BertOutput(config)


class BertMoEEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([BertMoELayer(config) for _ in range(config.num_hidden_layers)])

class BertMoEModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = BertMoEEncoder(config)
