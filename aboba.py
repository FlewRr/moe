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

# Import the original model classes
from test import BertMoEForMultiLabelClassification

@dataclass
class ImprovedMultiLabelConfig:
    # Путь к твоей предобученной MLM-MoE модели
    pretrained_mlm_path: str = "final_model"
    
    # Путь к Kaggle csv
    train_csv: str = "train.csv"
    test_csv: str = "test.csv"
    
    # Колонки
    title_column: str = "TITLE"
    abstract_column: str = "ABSTRACT"
    label_columns: List[str] = field(default_factory=lambda: [
        "Computer Science",
        "Physics", 
        "Mathematics",
        "Statistics",
        "Quantitative Biology",
        "Quantitative Finance",
    ])
    
    # Токенизатор
    tokenizer_name: str = "bert-base-uncased"
    
    # Улучшенные гиперпараметры для борьбы с оверфиттингом
    max_length: int = 256
    train_batch_size: int = 16  # Уменьшили для стабильности
    eval_batch_size: int = 32
    num_train_epochs: int = 20  # Уменьшили максимальное количество эпох
    learning_rate: float = 3e-5  # Чуть меньше learning rate
    weight_decay: float = 0.1  # Увеличили weight decay для регуляризации
    warmup_ratio: float = 0.1  # Используем ratio вместо абсолютных шагов
    max_grad_norm: float = 1.0  # Gradient clipping
    
    # Early stopping параметры
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    metric_for_best_model: str = "eval_f1_micro"
    greater_is_better: bool = True
    load_best_model_at_end: bool = True
    
    # Logging и сохранение
    logging_steps: int = 50
    save_steps: int = 50
    eval_steps: int = 50
    save_total_limit: int = 3
    output_dir: str = "./improved_moe_multilabel"
    
    # Регуляризация
    hidden_dropout_prob: float = 0.3  # Увеличили dropout
    attention_probs_dropout_prob: float = 0.2
    classifier_dropout: float = 0.4
    
    # Learning rate scheduler
    lr_scheduler_type: str = "cosine"  # Cosine annealing
    
    # Data augmentation
    use_data_augmentation: bool = True
    augmentation_prob: float = 0.1

class ImprovedBertMoEForMultiLabelClassification(BertMoEForMultiLabelClassification):
    """
    Улучшенная версия модели с дополнительной регуляризацией
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Увеличиваем dropout в классификаторе
        self.classifier_dropout = nn.Dropout(config.classifier_dropout if hasattr(config, 'classifier_dropout') else 0.4)
        
        # Добавляем дополнительный слой для регуляризации
        self.additional_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.additional_layer_norm = nn.LayerNorm(config.hidden_size)
        
        # Re-initialize classifier with additional regularization
        self.classifier = nn.Sequential(
            self.additional_layer_norm,
            self.classifier_dropout,
            self.additional_dense,
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels)
        )
        
        # Weight initialization для лучшей сходимости
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Инициализация весов для лучшей сходимости"""
        if isinstance(module, nn.Linear):
            # Xavier/Glorot initialization
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

def cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr=0):
    """
    Cosine learning rate scheduler with warmup
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def build_compute_metrics_fn(threshold: float = 0.5):
    """
    Улучшенная функция для вычисления метрик
    """
    def compute_metrics(eval_pred: EvalPrediction):
        logits, labels = eval_pred
        
        # Применяем sigmoid
        probs = 1 / (1 + np.exp(-logits))
        
        # Бинаризация
        y_pred = (probs >= threshold).astype(int)
        y_true = labels.astype(int)
        
        # Считаем метрики
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        
        # Добавляем метрики для каждого класса
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
    """
    Простая аугментация текста для борьбы с оверфиттингом
    """
    augmented_texts = []
    for text in texts:
        if np.random.random() < augmentation_prob:
            # Простая аугментация: перемешивание слов (очень базовая)
            words = text.split()
            if len(words) > 3:
                # Перемешиваем 20% слов в середине текста
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
    """
    Улучшенный Trainer с дополнительной логикой
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_metric = -float('inf')
        self.patience_counter = 0
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Улучшенная функция потерь с label smoothing
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if labels is not None:
            # BCEWithLogitsLoss с label smoothing
            loss_fct = nn.BCEWithLogitsLoss()
            
            # Label smoothing (очень легкое)
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
    
    print("🚀 Starting improved training with overfitting prevention...")
    print(f"Configuration:")
    print(f"  - Learning rate: {cfg.learning_rate}")
    print(f"  - Weight decay: {cfg.weight_decay}")
    print(f"  - Dropout: {cfg.hidden_dropout_prob}")
    print(f"  - Early stopping patience: {cfg.early_stopping_patience}")
    print(f"  - Max epochs: {cfg.num_train_epochs}")
    
    # Загрузка токенизатора
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True)
    
    # Загрузка и подготовка данных
    raw_dataset = load_dataset("csv", data_files={"train": cfg.train_csv})["train"]
    
    # Разделение на train/val с stratification
    dataset_splits = raw_dataset.train_test_split(test_size=0.15, seed=42)
    train_dataset = dataset_splits["train"]
    eval_dataset = dataset_splits["test"]
    
    LABEL_COLUMNS = cfg.label_columns
    print("📊 Dataset info:")
    print(f"  - Total samples: {len(raw_dataset)}")
    print(f"  - Train samples: {len(train_dataset)}")
    print(f"  - Eval samples: {len(eval_dataset)}")
    print(f"  - Label columns: {LABEL_COLUMNS}")
    
    # Загрузка конфигурации модели
    base_config = AutoConfig.from_pretrained(cfg.pretrained_mlm_path)
    base_config.num_labels = len(LABEL_COLUMNS)
    base_config.problem_type = "multi_label_classification"
    
    # Установка улучшенных параметров регуляризации
    base_config.hidden_dropout_prob = cfg.hidden_dropout_prob
    base_config.attention_probs_dropout_prob = cfg.attention_probs_dropout_prob
    base_config.classifier_dropout = cfg.classifier_dropout
    
    # Создание улучшенной модели
    model = ImprovedBertMoEForMultiLabelClassification.from_pretrained(
        cfg.pretrained_mlm_path,
        config=base_config,
        ignore_mismatched_sizes=True,
    )
    
    print(f"🧠 Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    def preprocess_function(examples):
        """Улучшенная предобработка с аугментацией"""
        # Аугментация для тренировочных данных
        if cfg.use_data_augmentation:
            titles = apply_text_augmentation(examples[cfg.title_column], cfg.augmentation_prob)
            abstracts = apply_text_augmentation(examples[cfg.abstract_column], cfg.augmentation_prob)
        else:
            titles = examples[cfg.title_column]
            abstracts = examples[cfg.abstract_column]
        
        # Токенизация
        tokenized = tokenizer(
            titles,
            abstracts,
            truncation=True,
            max_length=cfg.max_length,
            padding=False,
        )
        
        # Подготовка labels
        labels = []
        for i in range(len(titles)):
            labels.append([examples[col][i] for col in LABEL_COLUMNS])
        tokenized["labels"] = labels
        
        return tokenized
    
    # Применение предобработки
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
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Улучшенные training arguments
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
        
        # Scheduler
        lr_scheduler_type=cfg.lr_scheduler_type,
        
        # Early stopping
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        
        # Logging и сохранение
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy="steps",
        
        # Оптимизации
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        gradient_accumulation_steps=2,  # Эффективный batch size = 32
        
        # Отключение отчетов
        report_to="none",
        save_safetensors=False,
        
        # Дополнительные параметры
        label_smoothing_factor=0.01,  # Легкое label smoothing
    )
    
    # Создание improved trainer
    trainer = ImprovedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics_fn(threshold=0.5),
        # callbacks=[
        #     EarlyStoppingCallback(
        #         early_stopping_patience=cfg.early_stopping_patience,
        #         early_stopping_threshold=cfg.early_stopping_threshold
        #     )
        # ]
    )
    
    print("🏃‍♂️ Starting training...")
    trainer.train()
    
    # Финальная оценка
    print("📊 Final evaluation...")
    eval_results = trainer.evaluate()
    print("Final results:", eval_results)
    
    # Сохранение модели
    final_dir = os.path.join(cfg.output_dir, "final_model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    # Сохранение метрик
    import json
    with open(os.path.join(final_dir, "training_metrics.json"), "w") as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"✅ Model saved to {final_dir}")
    print(f"🎯 Best F1 Micro: {eval_results.get('eval_f1_micro', 'N/A'):.4f}")
    print(f"🎯 Best F1 Macro: {eval_results.get('eval_f1_macro', 'N/A'):.4f}")

if __name__ == "__main__":
    main()
