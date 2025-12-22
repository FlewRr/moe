# moe.py
import torch
import torch.nn as nn
from typing import Optional
from transformers import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import (
    BertLayer,
    BertOutput,
    BertLMPredictionHead,
)


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

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, self.expert_size),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(self.expert_size, hidden_size),
            )
            for _ in range(num_experts)
        ])

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        assert hidden_dim == self.hidden_size

        x = hidden_states.view(-1, hidden_dim)
        gate_logits = self.gate(x)
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.k, dim=1)
        top_k_weights = torch.softmax(top_k_logits, dim=1

        final_output = torch.zeros_like(x)

        for i in range(self.num_experts):
            expert_mask = (top_k_indices == i)
            if expert_mask.any():
                token_indices = expert_mask.nonzero(as_tuple=True)[0]
                pos_in_topk = expert_mask.nonzero(as_tuple=True)[1]

                expert_inputs = x[token_indices]
                expert_weights = top_k_weights[token_indices, pos_in_topk]
                expert_out = self.experts[i](expert_inputs)
                weighted_out = expert_out * expert_weights.unsqueeze(-1)

                final_output.index_add_(0, token_indices, weighted_out)

        return final_output.view(batch_size, seq_len, hidden_dim)


from transformers.models.bert.modeling_bert import BertLayer
import torch.nn as nn

class BertLayerWithMoE(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        del self.intermediate
        del self.output

        self.moe_ffn = MoEFFN(
            hidden_size=config.hidden_size,
            num_experts=getattr(config, "num_experts", 4),
            expert_size=config.intermediate_size,
            k=getattr(config, "moe_k", 2),
            dropout_prob=config.hidden_dropout_prob,
        )

        self.moe_output_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.moe_output_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        **kwargs,
    ):
        self_attn_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        attn_output = self_attn_output[0]

        moe_output = self.moe_ffn(attn_output)

        moe_output = self.moe_output_dropout(moe_output)
        layer_output = self.moe_output_layer_norm(attn_output + moe_output)

        outputs = (layer_output,) + self_attn_output[1:]
        return outputs


class BertMoEForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config, add_pooling_layer=False)
        for layer in self.bert.encoder.layer:
            layer.__class__ = BertLayerWithMoE
            layer.__init__(config)

        self.cls = BertLMPredictionHead(config)
        self.post_init()

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
        prediction_scores = self.cls(sequence_output)

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
