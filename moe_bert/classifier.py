from moe import BertMoEModel
import torch.nn as nn

class BertForSequenceClassificationMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels

        self.bert = BertMoEModel(config)

        # классификационная голова
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.num_labels)
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled_output = outputs.last_hidden_state[:, 0]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits