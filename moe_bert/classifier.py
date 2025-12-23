from datasets import load_dataset
import os
import torch
import torch.nn as nn
from transformers import BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import BertConfig, BertPreTrainedModel
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)
from sklearn.metrics import precision_recall_fscore_support
from transformers import EvalPrediction
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List

class BertMoEForMultiLabelClassification(BertMoEForMaskedLM):
    def __init__(self, config: BertConfig):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.config.problem_type = "multi_label_classification"

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        **kwargs,
    ):
        bert_kwargs = {
            k: v for k, v in {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "position_ids": position_ids,
                "head_mask": head_mask,
                "inputs_embeds": inputs_embeds,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }.items() if v is not None
        }

        outputs = self.bert(**bert_kwargs)
        sequence_output = outputs.last_hidden_state

        cls_output = sequence_output[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())


        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ImprovedBertMoEForMultiLabelClassification(BertMoEForMultiLabelClassification):
    def __init__(self, config):
        super().__init__(config)

        self.classifier_dropout = nn.Dropout(config.classifier_dropout if hasattr(config, 'classifier_dropout') else 0.4)

        self.additional_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.additional_layer_norm = nn.LayerNorm(config.hidden_size)

        self.classifier = nn.Sequential(
            self.additional_layer_norm,
            self.classifier_dropout,
            self.additional_dense,
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
