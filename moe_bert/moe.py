import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertLayer, BertModel, BertEncoder, BertConfig, BertEmbeddings

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
                nn.Linear(intermediate_size, hidden_size)
            ) for _ in range(num_experts)
        ])
        # Гейтинг
        self.gate = nn.Linear(hidden_size, num_experts)

    def forward(self, x):
        # x: [B, L, H]
        B, L, H = x.size()
        gate_logits = self.gate(x)  # [B, L, E]
        topk_vals, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)
        topk_scores = torch.softmax(topk_vals, dim=-1)  # [B,L,top_k]

        # Вычисляем все эксперты
        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=2)  # [B,L,H,E]

        # Собираем top-k
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            idx = topk_idx[..., k]  # [B,L]
            score = topk_scores[..., k].unsqueeze(-1)  # [B,L,1]
            mask = nn.functional.one_hot(idx, num_classes=self.num_experts).unsqueeze(2).float()  # [B,L,1,E]
            out += (expert_outs * mask).sum(-1) * score

        return self.dropout(out)


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
        self.intermediate = None
        self.output = None

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        attention_output = self.attention(hidden_states, attention_mask, **kwargs)[0]
        layer_output = self.moe(attention_output)
        return layer_output, None


class BertMoEModel(BertModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        for i in range(config.num_hidden_layers):
            self.encoder.layer[i] = BertLayerMoE(config)
