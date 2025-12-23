from dataclasses import dataclass, field
from typing import List

class PretrainConfig:
    model_name = "your-moe-bert"
    dataset_name = "wikimedia/wikipedia"
    dataset_config = "20231101.en"
    text_column = "text"
    tokenizer = "bert-base-uncased"
    output_dir = "."
    seq_len = 128
    batch_size = 32

    masking_prob = 0.15

    lr = 5e-5
    weight_decay = 0.01
    warmup_steps = 1000
    max_steps = 50_000

    save_steps = 5_000
    logging_steps = 100
    eval_steps = 2000

    bert_hidden_size = 256
    bert_intermediate_size = 1024
    bert_num_hidden_layers = 4
    bert_num_attention_heads = 4
    num_experts = 4


class ClassificationConfig:
    backbone_path = "./final_model/pytorch_model.bin"
    dataset_path = "your_dataset"
    dataset_split = "train"
    text_column = "text"
    label_columns = ["label1", "label2", "label3", "label4", "label5", "label6"]

    num_labels = 6
    tokenizer = "bert-base-uncased"
    hidden_size = 256
    num_hidden_layers = 4
    num_attention_heads = 4
    intermediate_size = 1024
    num_experts = 4

    output_dir = "./cls_output"
    seq_len = 128
    batch_size = 32
    lr = 2e-5
    weight_decay = 0.01
    num_train_epochs = 3
    logging_steps = 100
    save_steps = 500
    eval_steps = 500

@dataclass
class ImprovedMultiLabelConfig:
    pretrained_mlm_path: str = "final_model"
    
    train_csv: str = "train.csv"
    test_csv: str = "test.csv"
    
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
    
    tokenizer_name: str = "bert-base-uncased"
    
    max_length: int = 256
    train_batch_size: int = 16
    eval_batch_size: int = 32
    num_train_epochs: int = 20
    learning_rate: float = 3e-5
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    metric_for_best_model: str = "eval_f1_micro"
    greater_is_better: bool = True
    load_best_model_at_end: bool = True
    
    logging_steps: int = 50
    save_steps: int = 50
    eval_steps: int = 50
    save_total_limit: int = 3
    output_dir: str = "./improved_moe_multilabel"
    
    hidden_dropout_prob: float = 0.3
    attention_probs_dropout_prob: float = 0.2
    classifier_dropout: float = 0.4
    
    lr_scheduler_type: str = "cosine"
    
    use_data_augmentation: bool = True
    augmentation_prob: float = 0.1
