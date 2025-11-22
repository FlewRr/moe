import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    BertTokenizerFast, BertConfig,
    Trainer, TrainingArguments
)
from moe import BertMoEModel

# ===============================
# Конфигурация
# ===============================
class Config:
    dataset_name = "shivanandmn/multilabel-classification-dataset"
    dataset_config = None
    text_column = "text"
    num_labels = 6
    seq_len = 256
    tokenizer = "bert-base-uncased"
    batch_size = 16
    lr = 3e-5
    max_steps = 500
    bert_hidden_size = 768
    bert_intermediate_size = 3072
    bert_num_hidden_layers = 12
    bert_num_attention_heads = 12
    num_experts = 4
    masking_prob = 0.15
    output_dir = "./moe_classifier"
    pretrained_path = None  # путь к pt файлу с pretrained MoE-BERT


cfg = Config()


# ===============================
# Модель для multilabel classification
# ===============================
class BertForMultilabelClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.bert = BertMoEModel(
            BertConfig(
                hidden_size=config.bert_hidden_size,
                intermediate_size=config.bert_intermediate_size,
                num_hidden_layers=config.bert_num_hidden_layers,
                num_attention_heads=config.bert_num_attention_heads,
                num_experts=config.num_experts,
            )
        )

        # классификационная голова
        self.classifier = nn.Sequential(
            nn.Linear(config.bert_hidden_size, config.bert_hidden_size),
            nn.ReLU(),
            nn.Linear(config.bert_hidden_size, config.num_labels)
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS токен
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# ===============================
# Подготовка датасета
# ===============================
def prepare_dataset(cfg, quick_test=False):
    # streaming=True для быстрого старта
    ds = load_dataset(cfg.dataset_name, split="train", streaming=True)
    if quick_test:
        ds = ds.take(20000)  # быстрый тест

    tokenizer = BertTokenizerFast.from_pretrained(cfg.tokenizer)

    def tokenize_fn(batch):
        enc = tokenizer(batch[cfg.text_column], padding="max_length", truncation=True, max_length=cfg.seq_len)
        # формируем multi-label тензор
        labels = torch.tensor([[batch[f"label_{i}"][j] for i in range(1, cfg.num_labels+1)] for j in range(len(batch[cfg.text_column]))], dtype=torch.float)
        enc["labels"] = labels
        return enc

    ds = ds.map(tokenize_fn, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


# ===============================
# Кастомный Trainer для BCEWithLogits
# ===============================
class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        logits = model(**inputs)
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        return (loss, logits) if return_outputs else loss


# ===============================
# Основная функция
# ===============================
def train(cfg, quick_test=False):
    ds = prepare_dataset(cfg, quick_test=quick_test)

    model = BertForMultilabelClassification(cfg)

    # загружаем pretrained веса для MoE-BERT если указаны
    if cfg.pretrained_path:
        state_dict = torch.load(cfg.pretrained_path, map_location="cpu")
        moe_state_dict = {k.replace("bert.", ""): v for k,v in state_dict.items() if k.startswith("bert.")}
        model.bert.load_state_dict(moe_state_dict, strict=False)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        max_steps=cfg.max_steps,
        logging_steps=50,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=100,
        report_to="none"
    )

    trainer = MultilabelTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        eval_dataset=None,
    )

    print("Started training!")
    trainer.train()
    trainer.save_model(cfg.output_dir)
    print("Training finished!")


if __name__ == "__main__":
    # быстрый тест на 20k примеров
    train(cfg, quick_test=True)
