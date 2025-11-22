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

class BertMoELayer(BertLayer):
    def __init__(self, config):
        super().__init__(config)

        self.attention = BertAttention(config)
        self.intermediateм = SwitchTransformersSparseMLP(config)
        self.output = BertOutput(config)


class BertMoEEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)

        self.layer = nn.ModuleList(
            [BertMoELayer(config) for _ in range(config.num_hidden_layers)]
        )


class BertMoEModel(BertModel):
    def __init__(self, config):
        super().__init__(config)

        self.encoder = BertMoEEncoder(config)