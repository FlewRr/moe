# moe.py
import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertModel, BertLayer, BertConfig, BertEncoder, BertEmbeddings

# ----------------------------
# Простая MoE с Top-K
# ----------------------------
class TopKMoE(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts=4, top_k=1, dropout=0.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dropout = nn.Dropout(dropout)

        # Эксперты — обычный feed-forward
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, intermediate_size),
                nn.GELU(),
                nn.Linear(intermediate_size, hidden_size),
            ) for _ in range(num_experts)
        ])

        # Гейтинг: learnable веса для маршрутизации
        self.gate = nn.Linear(hidden_size, num_experts, bias=True)

    def forward(self, x):
        # x: [batch, seq_len, hidden_size]
        batch, seq_len, hidden = x.size()
        # Гейтинг: logits экспертов
        gate_logits = self.gate(x)  # [batch, seq_len, num_experts]
        topk_vals, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)  # top_k экспертов
        topk_scores = torch.softmax(topk_vals, dim=-1)

        # Подготавливаем выход
        out = torch.zeros_like(x)

        # Простейшая реализация Top-K: суммируем вклад выбранных экспертов
        for i in range(self.top_k):
            idx = topk_idx[..., i]  # [batch, seq_len]
            score = topk_scores[..., i].unsqueeze(-1)  # [batch, seq_len, 1]
            expert_out = torch.stack([self.experts[e](x[b]) for b, e in enumerate(idx[:, 0])], dim=0)
            # Применяем вес gating
            out += self.dropout(expert_out * score)

        return out

# ----------------------------
# Подменяем FFN в BertLayer
# ----------------------------
class BertLayerMoE(BertLayer):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.moe = TopKMoE(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_experts=getattr(config, "num_experts", 1),
            top_k=getattr(config, "top_k", 1),
            dropout=getattr(config, "dropout", 0.0)
        )
        # Отключаем стандартный FFN
        self.intermediate = None
        self.output = None

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # стандартное внимание
        attention_output = self.attention(hidden_states, attention_mask, **kwargs)[0]
        # заменяем FFN на MoE
        layer_output = self.moe(attention_output)
        return layer_output, None

# ----------------------------
# Bert с MoE
# ----------------------------
class BertMoEModel(BertModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        # embeddings оставляем стандартные
        self.embeddings = BertEmbeddings(config)
        # подменяем все слои encoder на MoE
        self.encoder = BertEncoder(config)
        for i in range(config.num_hidden_layers):
            self.encoder.layer[i] = BertLayerMoE(config)
