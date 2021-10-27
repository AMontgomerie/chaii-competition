import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from utils import seed_everything, parse_args_train
from processing import prepare_train_features, convert_answers
from train import Trainer

if __name__ == "__main__":
    config = parse_args_train()
    seed_everything(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    pad_on_right = tokenizer.padding_side == "right"
    train = pd.read_csv(config.data_path, encoding="utf-8")
    if config.valid_data_path is None:
        raise ValueError(
            "Pretraining requires a validation dataset to be specified with --valid_data_path")
    valid = pd.read_csv(config.valid_data_path, encoding="utf-8")
    train['answers'] = train[['answer_start', 'answer_text']].apply(
        convert_answers,
        axis=1
    )
    valid['answers'] = valid[['answer_start', 'answer_text']].apply(
        convert_answers,
        axis=1
    )
    train_dataset = Dataset.from_pandas(train)
    valid_dataset = Dataset.from_pandas(valid)
    tokenized_train_ds = train_dataset.map(
        prepare_train_features,
        batched=True,
        remove_columns=train_dataset.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": config.max_length,
            "doc_stride": config.doc_stride,
            "pad_on_right": pad_on_right
        }
    )
    tokenized_train_ds.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions']
    )
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    trainer = Trainer(
        config.model,
        config.fold,
        tokenized_train_ds,
        valid_dataset,
        tokenizer,
        model_type=config.model_type,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        epochs=config.epochs,
        train_batch_size=config.train_batch_size,
        valid_batch_size=config.valid_batch_size,
        eval_step=config.eval_step,
        max_length=config.max_length,
        max_answer_length=config.max_answer_length,
        doc_stride=config.doc_stride,
        save_path=config.save_path,
        scheduler=config.scheduler,
        warmup=config.warmup,
        adam_epsilon=config.adam_epsilon,
        early_stopping=config.early_stopping,
        fp16=config.fp16,
        accumulation_steps=config.accumulation_steps,
        dataloader_workers=config.dataloader_workers,
        pretrain=True
    )
    trainer.train()
