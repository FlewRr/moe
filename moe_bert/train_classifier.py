import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    DataCollatorWithPadding,
)
from config import ClassificationConfig
from classifier import BertMoEForMultiLabelClassification
from transformers import BertConfig


def main():
    set_seed(42)
    cfg = ClassificationConfig()

    # --- 1. Токенизатор ---
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)

    # --- 2. Датасет ---
    dataset = load_dataset("csv", data_files=cfg.dataset_path, split="train")
    # Должен содержать: "text", и 6 колонок с бинарными метками (0/1)
    # надо считать из kaggle локально и дать путь

    def tokenize_and_align_labels(examples):
        tokenized = tokenizer(
            examples[cfg.text_column],
            truncation=True,
            padding=False,
            max_length=cfg.seq_len,
        )
        # Собираем метки в список списков
        labels = [
            [examples[col][i] for col in cfg.label_columns]
            for i in range(len(examples[cfg.text_column]))
        ]
        tokenized["labels"] = labels
        return tokenized

    tokenized_ds = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset.column_names,
    )

    # --- 3. Конфиг модели ---
    model_config = BertConfig(
        vocab_size=30522,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        intermediate_size=cfg.intermediate_size,
        num_labels=cfg.num_labels,
        num_experts=cfg.num_experts,
        moe_k=2,
    )

    # --- 4. Загрузка модели с backbone'ом ---
    model = BertMoEForMultiLabelClassification.from_pretrained_backbone(
        backbone_path=cfg.backbone_path,
        num_labels=cfg.num_labels,
        config=model_config,
    )

    # --- 5. Collator ---
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- 6. Training args ---
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        report_to="none",
    )

    # --- 7. Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        eval_dataset=tokenized_ds,  # ← замените на отдельный val-сплит в реальности
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # --- 8. Обучение ---
    trainer.train()
    trainer.save_model(cfg.output_dir + "/final")
    tokenizer.save_pretrained(cfg.output_dir + "/final")


if __name__ == "__main__":
    main()