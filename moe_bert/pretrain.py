from datasets import load_dataset
from config import PretrainConfig
from transformers import (
    BertTokenizerFast, BertForMaskedLM, BertConfig,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)
from moe import BertMoEModel

cfg = PretrainConfig()

def get_model(cfg):
    config = BertConfig(
        hidden_size=cfg.bert_hidden_size,
        intermediate_size=cfg.bert_intermediate_size,
        num_hidden_layers=cfg.bert_num_hidden_layers,
        num_attention_heads=cfg.bert_num_attention_heads,
    )
    config.num_experts = getattr(cfg, "num_experts", 1)
    backbone = BertMoEModel(config)
    model = BertForMaskedLM(config)
    model.bert = backbone
    return model

def pretrain(cfg: PretrainConfig, quick_test=False):
    # ========================
    # 1. Загружаем датасет
    # ========================
    if quick_test:
        # маленький срез для теста
        ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split="train", streaming=True)
        ds = ds.take(20000)
    else:
        # полный датасет с потоковой загрузкой
        ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split="train", streaming=True)

    # ========================
    # 2. Токенизатор
    # ========================
    tokenizer = BertTokenizerFast.from_pretrained(cfg.tokenizer)

    def tokenize_batch(batch):
        return tokenizer(batch[cfg.text_column], truncation=True, max_length=cfg.seq_len, padding=False)

    ds = ds.map(tokenize_batch, batched=True)

    # ========================
    # 3. Коллатор для MLM
    # ========================
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=cfg.masking_prob,
    )

    # ========================
    # 4. Модель
    # ========================
    model = get_model(cfg)

    # ========================
    # 5. TrainingArguments
    # ========================
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

    # ========================
    # 6. Trainer
    # ========================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # ========================
    # 7. Обучение
    # ========================
    print("Started training!")
    trainer.train()
    trainer.save_model(cfg.output_dir)
    print("Pretraining finished!")


if __name__ == "__main__":
    # быстрый тест
    pretrain(cfg, quick_test=True)
