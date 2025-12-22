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
