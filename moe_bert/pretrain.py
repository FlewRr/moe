from datasets import load_dataset
from config import PretrainConfig
from transformers import (
    BertTokenizerFast, BertForMaskedLM, BertConfig,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)
from moe import BertMoEModel
import torch
from torch.utils.data import IterableDataset

cfg = PretrainConfig()


# ----------------------------
# Кастомный IterableDataset для токенизации на лету
# ----------------------------
class WikiStreamDataset(IterableDataset):
    def __init__(self, tokenizer, seq_len, dataset_name, dataset_config):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.dataset = load_dataset(dataset_name, dataset_config, split="train", streaming=True)

    def __iter__(self):
        for example in self.dataset:
            tokenized = self.tokenizer(
                example[cfg.text_column],
                truncation=True,
                max_length=self.seq_len,
                padding="max_length",
                return_tensors="pt"
            )
            yield {
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0)
            }

class BertMoEConfig(BertConfig):
    def __init__(
        self,
        num_experts=1,
        expert_capacity=32,
        top_k=1,
        router_bias=True,
        dropout=0.0,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.top_k = top_k
        self.router_bias = router_bias
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps


# ----------------------------
# Модель
# ----------------------------
def get_model(cfg, pretrained_path=None):
    config = BertMoEConfig(
        hidden_size=cfg.bert_hidden_size,
        intermediate_size=cfg.bert_intermediate_size,
        num_hidden_layers=cfg.bert_num_hidden_layers,
        num_attention_heads=cfg.bert_num_attention_heads,
        num_experts=getattr(cfg, "num_experts", 1),
        expert_capacity=getattr(cfg, "expert_capacity", 32),
        top_k=getattr(cfg, "top_k", 1)
    )
    config.num_experts = getattr(cfg, "num_experts", 1)

    backbone = BertMoEModel(config)
    model = BertForMaskedLM(config)
    model.bert = backbone

    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location="cpu")
        model.bert.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {pretrained_path}")

    return model


# ----------------------------
# Pretrain
# ----------------------------
def pretrain(cfg: PretrainConfig, pretrained_path=None):
    tokenizer = BertTokenizerFast.from_pretrained(cfg.tokenizer)

    train_dataset = WikiStreamDataset(
        tokenizer=tokenizer,
        seq_len=cfg.seq_len,
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=cfg.masking_prob,
    )

    model = get_model(cfg, pretrained_path=pretrained_path)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        max_steps=cfg.max_steps,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # iterable dataset
        data_collator=collator,
        tokenizer=tokenizer
    )

    print("Started training!")
    trainer.train()
    trainer.save_model(cfg.output_dir)
    print("Pretraining finished!")


if __name__ == "__main__":
    pretrain(cfg, pretrained_path=None)
