# train.py
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from config import PretrainConfig
from moe import BertMoEForMaskedLM
from transformers import BertConfig


def main():
    set_seed(42)

    # --- 1. Загрузка конфигурации ---
    cfg = PretrainConfig()

    # --- 2. Создание конфигурации модели BERT + MoE ---
    model_config = BertConfig(
        vocab_size=30522,  # как у bert-base-uncased
        hidden_size=cfg.bert_hidden_size,
        num_hidden_layers=cfg.bert_num_hidden_layers,
        num_attention_heads=cfg.bert_num_attention_heads,
        intermediate_size=cfg.bert_intermediate_size,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        # Передаём MoE параметры как атрибуты конфига
        num_experts=cfg.num_experts,
        moe_k=2,  # фиксировано, но можно вынести в PretrainConfig
    )

    # --- 3. Инициализация модели и токенизатора ---
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
    model = BertMoEForMaskedLM(model_config)

    print(f"Model initialized with {cfg.num_experts} experts per layer.")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- 4. Загрузка и подготовка датасета ---
    print("Loading dataset...")
    dataset = load_dataset(
        cfg.dataset_name,
        cfg.dataset_config,
        split="train[:1%]"  # ⚠️ Ограничение для теста! Уберите [:1%] для полного обучения
    )

    def tokenize_function(examples):
        return tokenizer(
            examples[cfg.text_column],
            truncation=True,
            padding=False,
            max_length=cfg.seq_len,
            return_special_tokens_mask=True,
        )

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    # Фильтруем слишком короткие последовательности (опционально)
    tokenized_dataset = tokenized_dataset.filter(
        lambda x: len(x["input_ids"]) >= cfg.seq_len // 2
    )

    # --- 5. Data collator для MLM ---
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=cfg.masking_prob,
    )

    # --- 6. Настройка обучения ---
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=True,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        evaluation_strategy="steps" if cfg.eval_steps else "no",
        save_strategy="steps",
        load_best_model_at_end=False,
        fp16=True,  # включить, если GPU поддерживает
        dataloader_num_workers=4,
        report_to="none",  # или "tensorboard", "wandb"
        remove_unused_columns=False,  # важно при кастомных collator'ах
    )

    # --- 7. Создание Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=None,  # можно добавить отдельный split при необходимости
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # --- 8. Запуск предобучения ---
    print("Starting pretraining...")
    trainer.train()

    # --- 9. Сохранение финальной модели ---
    final_dir = os.path.join(cfg.output_dir, "final_model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved to {final_dir}")


if __name__ == "__main__":
    main()