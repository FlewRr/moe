import torch
import torch.nn as nn
from transformers import BertConfig, BertPreTrainedModel
from moe import BertMoEForMaskedLM  # чтобы использовать ту же архитектуру слоёв


class BertMoEForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # Используем ТОТ ЖЕ encoder, что и в MLM-версии
        from moe import BertMoEForMaskedLM
        base_model = BertMoEForMaskedLM(config)
        self.bert = base_model.bert  # encoder без MLM head

        # Новая классификационная голова
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Инициализируем веса головы, остальное может быть загружено из .pt
        self.post_init()

    @classmethod
    def from_pretrained_backbone(cls, backbone_path: str, num_labels: int, config=None):
        """
        Загружает backbone из .pt файла (state_dict без classifier) и создаёт новую модель.
        """
        if config is None:
            # Создаём минимальный конфиг — вы можете передать свой
            config = BertConfig(
                hidden_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=1024,
                vocab_size=30522,
                num_labels=num_labels,
            )
        config.num_labels = num_labels

        model = cls(config)

        # Загружаем state_dict
        state_dict = torch.load(backbone_path, map_location="cpu", weights_only=True)

        # Фильтруем только параметры backbone'а (всё, что начинается с 'bert.')
        backbone_state = {k: v for k, v in state_dict.items() if k.startswith("bert.")}

        # Загружаем в модель
        model.load_state_dict(backbone_state, strict=False)
        print(f"Loaded backbone from {backbone_path} (keys matched: {len(backbone_state)})")

        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Используем pooled_output (как в оригинальном BERT)
        # pooled_output = outputs.pooler_output  # [batch, hidden]
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch, num_labels]

        loss = None
        if labels is not None:
            # Multi-label: BCEWithLogitsLoss
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }