import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import random
import psutil
import torch

import datasets
from datasets import load_dataset, DatasetDict
import numpy as np
import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    BartConfig, 
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Trainer,
)
from transformers.trainer_utils import get_last_checkpoint


sys.path.append(".")
from apetokenizer.ape_tokenizer import APETokenizer

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    min_lr: Optional[float] = field(default=1e-5, metadata={"help": "Min learning rate"})
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    pretrain_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the pretraining data."}
    )
    val_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the validation data."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(default=1, metadata={"help": "Number of workers for preprocessing."})
    max_seq_length: Optional[int] = field(default=256, metadata={"help": "Max input length."})
    debug_flag: bool = field(default=False, metadata={"help": "Debug mode."})

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.max_grad_norm = 1.0  # clip gradients to norm 1

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    set_seed(training_args.seed)

    delimiter = "\t" if "tsv" in data_args.pretrain_path else ","
    with training_args.main_process_first(desc="load data"):
        # Load TSV dataset with only relevant columns
        full_dataset = load_dataset(
            "csv",
            data_files=data_args.pretrain_path,
            delimiter="\t",
            cache_dir="cache",
            column_names=["canonical_smiles", "fps", "split"],  # force only these columns
        )["train"]

        # Split dataset into train/val based on 'split' column
        train_dataset = full_dataset.filter(lambda x: x["split"] == "train")
        val_dataset = full_dataset.filter(lambda x: x["split"] == "val")

        pretrain_datasets = DatasetDict({
            "train": train_dataset,
            "val": val_dataset
        })

        pretrain_datasets = pretrain_datasets.filter(
            lambda x: x["fps"] and x["fps"].strip() not in ["", "NA", "None", "null"] and
                    x["canonical_smiles"] and x["canonical_smiles"].strip() not in ["", "NA", "None", "null"],
            num_proc=data_args.preprocessing_num_workers
        )

        logger.info(f"RAM:{psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

        # Optional debug mode
        if data_args.debug_flag:
            pretrain_datasets = DatasetDict({
                "train": pretrain_datasets["train"].select(range(1000)),
                "val": pretrain_datasets["val"].select(range(1000))
            })
            training_args.report_to = []
            training_args.overwrite_output_dir = True

    # Initialize tokenizer and model
    tokenizer_path = model_args.tokenizer_name
    tokenizer = APETokenizer()
    tokenizer.load_vocabulary(f"{tokenizer_path}/vocab.json")
    
    config = BartConfig.from_pretrained("facebook/bart-base")
    config.vocab_size = len(tokenizer)
    config.bos_token_id=tokenizer.bos_token_id
    config.eos_token_id=tokenizer.eos_token_id
    config.pad_token_id=tokenizer.pad_token_id
    config.unk_token_id=tokenizer.unk_token_id
    
    dtype = {
        None: None,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }.get(model_args.torch_dtype, None)

    model = BartForConditionalGeneration(config)

    model = model.to(dtype) if dtype is not None else model

    if training_args.should_log:
        print("=="*24, "Model Config", "=="*24)
        logger.info(f"{model.config}")
        logger.info(f"Model vocab size matches tokenizer: {model.config.vocab_size == len(tokenizer)}")
        logger.info(f"Model dtype: {next(model.parameters()).dtype}")

    def tokenize_function(examples):
        # Data: <fps> → <canonical_smiles> (Translation Task)
        data = {"input_ids": [], "labels": []}
        examples_num = len(examples["fps"])
        for i in range(examples_num):
            fps = examples["fps"][i]
            fps_ids = tokenizer.encode(fps, max_length=data_args.max_seq_length, add_special_tokens=False)

            smiles = examples["canonical_smiles"][i]
                
            smiles_ids = tokenizer.encode(smiles, max_length=data_args.max_seq_length-1, add_special_tokens=False)
            smiles_ids = smiles_ids + [tokenizer.eos_token_id]

            data["input_ids"].append(fps_ids)
            data["labels"].append(smiles_ids)

        return data


    with training_args.main_process_first(desc="dataset map processing"):
        tokenized_dataset = pretrain_datasets.map(
            function=tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            remove_columns= pretrain_datasets["train"].column_names
        )
    
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)
            
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        labels_flat = labels.reshape(-1)
        preds_flat = preds.reshape(-1)
        mask = labels_flat != -100
        labels_flat = labels_flat[mask]
        preds_flat = preds_flat[mask]
        correct_predictions = np.sum(labels_flat == preds_flat)
        total_predictions = len(labels_flat)
        token_level_accuracy = correct_predictions / total_predictions

        cnt = 0
        for pred, label in zip(preds, labels):
            # Truncate label after EOS token
            if tokenizer.eos_token_id in label:
                eos_index = list(label).index(tokenizer.eos_token_id) + 1
                label = label[:eos_index]
            # Truncate pred to same length as label
            pred = pred[:len(label)]
            # Compare sequences
            if list(pred) == list(label):
                cnt += 1
        accuracy = cnt / len(preds)
        
        return {"token_level_accuracy": token_level_accuracy, "sequence_level_accuracy": accuracy}

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )
    training_args.remove_unused_columns = False
    training_args.lr_scheduler_kwargs={"min_lr": model_args.min_lr} if training_args.lr_scheduler_type == 'cosine_with_min_lr' else None
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
    )

    # Training
    if training_args.do_train:
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
        elif training_args.resume_from_checkpoint is not None:
            last_checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        
        metrics = train_result.metrics
        metrics["train_samples"] = len(tokenized_dataset["train"])
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(tokenized_dataset["val"])

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"tasks": "bart pretrain"}
    trainer.create_model_card(**kwargs)

if __name__ == "__main__":
    main()