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
    cfg = PretrainConfig()

    model_config = BertConfig(
        vocab_size=30522,
        hidden_size=cfg.bert_hidden_size,
        num_hidden_layers=cfg.bert_num_hidden_layers,
        num_attention_heads=cfg.bert_num_attention_heads,
        intermediate_size=cfg.bert_intermediate_size,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        pad_token_id=0,
        num_experts=cfg.num_experts,
        moe_k=2,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
    model = BertMoEForMaskedLM(model_config)

    print(f"Model initialized with {cfg.num_experts} experts.")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("Loading dataset in streaming mode...")
    dataset = load_dataset(
        cfg.dataset_name,
        cfg.dataset_config,
        split="train",
        streaming=True
    )

    def tokenize_function(examples):
        return tokenizer(
            examples[cfg.text_column],
            truncation=True,
            padding=False,
            max_length=cfg.seq_len,
            return_special_tokens_mask=True,
        )

    original_columns = dataset.column_names
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=original_columns,
    )

    def filter_short(example):
        return len(example["input_ids"]) >= cfg.seq_len // 2

    tokenized_dataset = tokenized_dataset.filter(filter_short)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=cfg.masking_prob,
    )

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=True,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_strategy="steps",
        load_best_model_at_end=False,
        fp16=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",
        dataloader_drop_last=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Starting pretraining (streaming)...")
    trainer.train()

    final_dir = os.path.join(cfg.output_dir, "final_model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved to {final_dir}")


if __name__ == "__main__":
    main()
