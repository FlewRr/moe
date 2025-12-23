import os
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)
from sklearn.metrics import precision_recall_fscore_support
from transformers import EvalPrediction
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List
from config import ImprovedMultiLabelConfig

def cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr=0):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def build_compute_metrics_fn(threshold: float = 0.5):
    def compute_metrics(eval_pred: EvalPrediction):
        logits, labels = eval_pred

        probs = 1 / (1 + np.exp(-logits))

        y_pred = (probs >= threshold).astype(int)
        y_true = labels.astype(int)

        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        return {
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
            "f1_micro": f1_micro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_per_class": precision_per_class.tolist(),
            "recall_per_class": recall_per_class.tolist(),
            "f1_per_class": f1_per_class.tolist(),
        }

    return compute_metrics

def apply_text_augmentation(texts, augmentation_prob=0.1):
    augmented_texts = []
    for text in texts:
        if np.random.random() < augmentation_prob:
            words = text.split()
            if len(words) > 3:
                mid_start = len(words) // 4
                mid_end = 3 * len(words) // 4
                mid_words = words[mid_start:mid_end]
                np.random.shuffle(mid_words)
                words[mid_start:mid_end] = mid_words
                augmented_texts.append(' '.join(words))
            else:
                augmented_texts.append(text)
        else:
            augmented_texts.append(text)
    return augmented_texts

class ImprovedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_metric = -float('inf')
        self.patience_counter = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()

            if hasattr(self.args, 'label_smoothing_factor') and self.args.label_smoothing_factor > 0:
                smooth_factor = self.args.label_smoothing_factor
                labels = labels * (1 - smooth_factor) + 0.5 * smooth_factor

            loss = loss_fct(logits, labels.float())
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

def main():
    set_seed(42)
    cfg = ImprovedMultiLabelConfig()

    print(f"Configuration:")
    print(f"  - Learning rate: {cfg.learning_rate}")
    print(f"  - Weight decay: {cfg.weight_decay}")
    print(f"  - Dropout: {cfg.hidden_dropout_prob}")
    print(f"  - Early stopping patience: {cfg.early_stopping_patience}")
    print(f"  - Max epochs: {cfg.num_train_epochs}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True)

    raw_dataset = load_dataset("csv", data_files={"train": cfg.train_csv})["train"]

    dataset_splits = raw_dataset.train_test_split(test_size=0.15, seed=42)
    train_dataset = dataset_splits["train"]
    eval_dataset = dataset_splits["test"]

    LABEL_COLUMNS = cfg.label_columns
    print("📊 Dataset info:")
    print(f"  - Total samples: {len(raw_dataset)}")
    print(f"  - Train samples: {len(train_dataset)}")
    print(f"  - Eval samples: {len(eval_dataset)}")
    print(f"  - Label columns: {LABEL_COLUMNS}")

    base_config = AutoConfig.from_pretrained(cfg.pretrained_mlm_path)
    base_config.num_labels = len(LABEL_COLUMNS)
    base_config.problem_type = "multi_label_classification"

    base_config.hidden_dropout_prob = cfg.hidden_dropout_prob
    base_config.attention_probs_dropout_prob = cfg.attention_probs_dropout_prob
    base_config.classifier_dropout = cfg.classifier_dropout

    model = ImprovedBertMoEForMultiLabelClassification.from_pretrained(
        cfg.pretrained_mlm_path,
        config=base_config,
        ignore_mismatched_sizes=True,
    )

    print(f"🧠 Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

    def preprocess_function(examples):
        if cfg.use_data_augmentation:
            titles = apply_text_augmentation(examples[cfg.title_column], cfg.augmentation_prob)
            abstracts = apply_text_augmentation(examples[cfg.abstract_column], cfg.augmentation_prob)
        else:
            titles = examples[cfg.title_column]
            abstracts = examples[cfg.abstract_column]

        tokenized = tokenizer(
            titles,
            abstracts,
            truncation=True,
            max_length=cfg.max_length,
            padding=False,
        )

        labels = []
        for i in range(len(titles)):
            labels.append([examples[col][i] for col in LABEL_COLUMNS])
        tokenized["labels"] = labels

        return tokenized

    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_dataset.column_names,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=cfg.max_grad_norm,

        lr_scheduler_type=cfg.lr_scheduler_type,

        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,

        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy="steps",

        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        gradient_accumulation_steps=2,

        report_to="none",
        save_safetensors=False,

        label_smoothing_factor=0.01,
    )

    trainer = ImprovedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics_fn(threshold=0.5),
    )

    print("🏃‍♂️ Starting training...")
    trainer.train()

    print("📊 Final evaluation...")
    eval_results = trainer.evaluate()
    print("Final results:", eval_results)

    final_dir = os.path.join(cfg.output_dir, "final_model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    import json
    with open(os.path.join(final_dir, "training_metrics.json"), "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"✅ Model saved to {final_dir}")
    print(f"🎯 Best F1 Micro: {eval_results.get('eval_f1_micro', 'N/A'):.4f}")
    print(f"🎯 Best F1 Macro: {eval_results.get('eval_f1_macro', 'N/A'):.4f}")

if __name__ == "__main__":
    main()
