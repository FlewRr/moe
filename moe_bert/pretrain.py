from datasets import load_dataset
from config import PretrainConfig
from transformers import (
    BertTokenizerFast, BertForMaskedLM, BertConfig,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)
from moe import BertMoEModel

cfg = PretrainConfig

def load_wiki_dataset(cfg):
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split="train")
    return ds

def prepare_tokenizer(cfg):
    tokenizer = BertTokenizerFast.from_pretrained(cfg.tokenizer)
    return tokenizer

def tokenize(batch, tokenizer, cfg):
    return tokenizer(
        batch[cfg.text_column],
        truncation=True,
        max_length=cfg.seq_len,
        padding=False,
    )

def build_collator(tokenizer, cfg):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=cfg.masking_prob,
    )

def get_model(cfg):
    config = BertConfig(
        hidden_size=cfg.bert_hidden_size,
        intermediate_size=cfg.bert_intermediate_size,
        num_hidden_layers=cfg.bert_num_hidden_layers,
        num_attention_heads=cfg.bert_num_attention_heads,
    )

    config.num_experts = cfg.num_experts

    backbone = BertMoEModel(config)

    model = BertForMaskedLM(config)
    model.bert = backbone
    return model

def pretrain(cfg: PretrainConfig):
    # 1. Загружаем датасет
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split="train")

    # 2. Загружаем токенизатор
    tokenizer = BertTokenizerFast.from_pretrained(cfg.tokenizer)

    # 3. Токенизация
    def tokenize_batch(batch):
        return tokenizer(
            batch[cfg.text_column],
            truncation=True,
            max_length=cfg.seq_len,
            padding=False,
        )

    ds = ds.map(
        tokenize_batch,
        batched=True,
        num_proc=16,
        batch_size=1000
    )

    # Оставляем только нужные колонки
    ds = ds.remove_columns([c for c in ds.column_names if c not in ["input_ids", "attention_mask"]])
    ds = ds.with_format("torch")

    # 4. Коллатор для MLM
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=cfg.masking_prob,
    )

    # 5. Модель
    model = get_model(cfg)

    # 6. TrainingArguments
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

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # 8. Запуск обучения
    print("Started training!")

    trainer.train()
    trainer.save_model(cfg.output_dir)

    print("Pretraining finished!")



if __name__ == "__main__":
    cfg = PretrainConfig()

    pretrain(cfg)