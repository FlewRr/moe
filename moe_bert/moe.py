import torch
import torch.nn as nn
from typing import Optional

class MoEFFN(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 4,
        expert_size: Optional[int] = None,
        k: int = 2,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.k = k
        self.expert_size = expert_size or hidden_size * 4

        # Experts: Linear -> GELU -> Linear
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, self.expert_size),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(self.expert_size, hidden_size),
            )
            for _ in range(num_experts)
        ])

        # Gating network
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states):
        # hidden_states: [batch_size, seq_len, hidden_size]
        batch_size, seq_len, hidden_dim = hidden_states.shape
        assert hidden_dim == self.hidden_size

        # Flatten
        x = hidden_states.view(-1, hidden_dim)  # [N, H], N = B*L
        N = x.size(0)

        # Compute gate logits
        gate_logits = self.gate(x)  # [N, E]
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.k, dim=1)  # [N, k]
        top_k_weights = torch.softmax(top_k_logits, dim=1)  # [N, k]

        # Output accumulator
        final_output = torch.zeros_like(x)

        # Dispatch to experts
        for i in range(self.num_experts):
            # Which tokens go to expert i?
            expert_mask = (top_k_indices == i)  # [N, k], bool
            if expert_mask.any():
                # Indices of tokens routed to this expert
                token_indices = expert_mask.nonzero(as_tuple=True)[0]  # [M]
                # Which position in top-k (0 or 1 if k=2)
                pos_in_topk = expert_mask.nonzero(as_tuple=True)[1]    # [M]

                # Gather input tokens
                expert_inputs = x[token_indices]  # [M, H]
                # Get corresponding weights
                expert_weights = top_k_weights[token_indices, pos_in_topk]  # [M]

                # Forward through expert
                expert_out = self.experts[i](expert_inputs)  # [M, H]

                # Weighted contribution
                weighted_out = expert_out * expert_weights.unsqueeze(-1)  # [M, H]

                # Accumulate
                final_output.index_add_(0, token_indices, weighted_out)

        return final_output.view(batch_size, seq_len, hidden_dim)

from transformers.models.bert.modeling_bert import BertLayer, BertOutput

class BertLayerWithMoE(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        # Удаляем стандартный FFN
        del self.intermediate
        del self.output

        # Вставляем MoE
        self.moe_ffn = MoEFFN(
            hidden_size=config.hidden_size,
            num_experts=getattr(config, "num_experts", 4),
            expert_size=config.intermediate_size,
            k=getattr(config, "moe_k", 2),
            dropout_prob=config.hidden_dropout_prob,
        )

        # Сохраняем LayerNorm после FFN (как в оригинальном BERT)
        self.output = BertOutput(config)  # этот модуль содержит только LayerNorm + dropout

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        **kwargs
    ):
        # Self-attention
        self_attn_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        attn_output = self_attn_output[0]

        # MoE FFN
        moe_output = self.moe_ffn(attn_output)

        # Apply LayerNorm: residual + Norm
        # В оригинале: output = LayerNorm(attention_output + ffn_output)
        layer_output = self.output(moe_output, attn_output)

        outputs = (layer_output,) + self_attn_output[1:]
        return outputs


from transformers import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import BertLMPredictionHead

class BertMoEForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # Создаём BERT, но заменяем слои на MoE-версии
        self.bert = BertModel(config, add_pooling_layer=False)
        for layer in self.bert.encoder.layer:
            layer.__class__ = BertLayerWithMoE
            layer.__init__(config)

        # MLM голова
        self.cls = BertLMPredictionHead(config)

        self.post_init()

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
            **kwargs
        )

        sequence_output = outputs.last_hidden_state  # [B, L, H]
        prediction_scores = self.cls(sequence_output)  # [B, L, vocab_size]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1)
            )

        return {
            "loss": loss,
            "logits": prediction_scores,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }