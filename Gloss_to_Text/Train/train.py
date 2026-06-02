from __future__ import annotations
import csv
import inspect
import json
import shutil
from pathlib import Path
from typing import Any
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
TRAIN_CONFIG: dict[str, Any] = {
    "model_name_or_path": "cointegrated/rut5-small",
    "train_file": "Gloss_to_Text/Dataset/data.jsonl",
    "validation_file": None,
    "validation_ratio": 0.15,
    "output_dir": "Gloss_to_Text/Model/rut5-small-gloss",
    "source_prefix": "gloss to russian: ",
    "source_column": "source",
    "target_column": "target",
    "max_source_length": 128,
    "max_target_length": 128,
    "num_train_epochs": 8,
    "learning_rate": 5e-5,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 20,
    "weight_decay": 0.01,
    "eval_strategy": "epoch",
    "save_strategy": "no",
    "logging_steps": 25,
    "predict_with_generate": True,
    "generation_max_length": 128,
    "generation_num_beams": 4,
    "seed": 42,
    "fp16": False,
}

def load_config() -> dict[str, Any]:
    return dict(TRAIN_CONFIG)

def load_records(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if any(char in str(file_path) for char in "*?[]"):
        matched_paths = sorted(file_path.parent.glob(file_path.name))
        if not matched_paths:
            raise FileNotFoundError(f"No dataset files matched: {file_path}")
        records: list[dict[str, Any]] = []
        for matched_path in matched_paths:
            records.extend(load_records(matched_path))
        return records
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"JSONL row must be an object at {file_path}:{line_number}")
                records.append(item)
        return records
    if suffix == ".json":
        data = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return [item for item in data["data"] if isinstance(item, dict)]
        raise ValueError(f"JSON dataset must be a list or an object with a data list: {file_path}")
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))
    raise ValueError(f"Unsupported dataset format: {file_path.suffix}")

def dataset_path_exists(path: str | Path) -> bool:
    file_path = Path(path)
    if any(char in str(file_path) for char in "*?[]"):
        return any(file_path.parent.glob(file_path.name))
    return file_path.exists()

def validate_records(
    records: list[dict[str, Any]],
    source_column: str,
    target_column: str,
) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for row in records:
        source = str(row.get(source_column, "")).strip()
        target = str(row.get(target_column, "")).strip()
        if source and target:
            cleaned.append({"source": source, "target": target})
    if not cleaned:
        raise ValueError(
            f"No valid records found. Expected columns: {source_column}, {target_column}"
        )
    return cleaned

def build_dataset(config: dict[str, Any]) -> tuple[Dataset, Dataset]:
    source_column = str(config.get("source_column", "source"))
    target_column = str(config.get("target_column", "target"))
    train_records = validate_records(
        load_records(config["train_file"]),
        source_column=source_column,
        target_column=target_column,
    )
    train_dataset = Dataset.from_list(train_records)
    validation_file = config.get("validation_file")
    if validation_file and dataset_path_exists(validation_file):
        valid_records = validate_records(
            load_records(validation_file),
            source_column=source_column,
            target_column=target_column,
        )
        valid_dataset = Dataset.from_list(valid_records)
    else:
        split = train_dataset.train_test_split(
            test_size=float(config.get("validation_ratio", 0.1)),
            seed=int(config.get("seed", 42)),
        )
        train_dataset = split["train"]
        valid_dataset = split["test"]
    return train_dataset, valid_dataset

def make_preprocess_fn(config: dict[str, Any], tokenizer):
    prefix = str(config.get("source_prefix", ""))
    max_source_length = int(config.get("max_source_length", 128))
    max_target_length = int(config.get("max_target_length", 128))
    def preprocess(batch):
        sources = [prefix + text for text in batch["source"]]
        model_inputs = tokenizer(
            sources,
            max_length=max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return preprocess

def compute_text_metrics(tokenizer):
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    def clean_token_ids(values):
        values = np.asarray(values)
        return np.where(values >= 0, values, pad_token_id)
    def compute(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = clean_token_ids(predictions)
        labels = clean_token_ids(labels)
        decoded_predictions = tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True,
        )
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        normalized_predictions = [text.strip() for text in decoded_predictions]
        normalized_labels = [text.strip() for text in decoded_labels]
        exact = [
            pred == label
            for pred, label in zip(normalized_predictions, normalized_labels, strict=False)
        ]
        return {"exact_match": float(np.mean(exact)) if exact else 0.0}
    return compute

def build_trainer_kwargs(tokenizer, model, training_args, tokenized_train, tokenized_valid, collator):
    kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_train,
        "eval_dataset": tokenized_valid,
        "data_collator": collator,
        "compute_metrics": compute_text_metrics(tokenizer),
    }
    trainer_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    return kwargs

def remove_training_artifacts(output_dir: str | Path) -> None:
    model_dir = Path(output_dir)
    if not model_dir.exists():
        return
    for checkpoint_dir in model_dir.glob("checkpoint-*"):
        if checkpoint_dir.is_dir():
            shutil.rmtree(checkpoint_dir)
    training_args = model_dir / "training_args.bin"
    if training_args.exists():
        training_args.unlink()

def main() -> None:
    config = load_config()
    set_seed(int(config.get("seed", 42)))
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name_or_path"])
    train_dataset, valid_dataset = build_dataset(config)
    preprocess = make_preprocess_fn(config, tokenizer)
    tokenized_train = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
    tokenized_valid = valid_dataset.map(preprocess, batched=True, remove_columns=valid_dataset.column_names)
    training_kwargs: dict[str, Any] = {
        "output_dir": config["output_dir"],
        "learning_rate": float(config.get("learning_rate", 5e-5)),
        "per_device_train_batch_size": int(config.get("per_device_train_batch_size", 4)),
        "per_device_eval_batch_size": int(config.get("per_device_eval_batch_size", 4)),
        "gradient_accumulation_steps": int(config.get("gradient_accumulation_steps", 1)),
        "num_train_epochs": float(config.get("num_train_epochs", 8)),
        "warmup_steps": int(config.get("warmup_steps", 0)),
        "weight_decay": float(config.get("weight_decay", 0.0)),
        "save_strategy": str(config.get("save_strategy", "epoch")),
        "logging_steps": int(config.get("logging_steps", 25)),
        "predict_with_generate": bool(config.get("predict_with_generate", True)),
        "generation_max_length": int(config.get("generation_max_length", 128)),
        "generation_num_beams": int(config.get("generation_num_beams", 4)),
        "fp16": bool(config.get("fp16", False)),
        "report_to": [],
    }
    eval_strategy_name = (
        "eval_strategy"
        if "eval_strategy" in Seq2SeqTrainingArguments.__dataclass_fields__
        else "evaluation_strategy"
    )
    training_kwargs[eval_strategy_name] = str(config.get("eval_strategy", "epoch"))
    training_args = Seq2SeqTrainingArguments(**training_kwargs)
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )
    trainer = Seq2SeqTrainer(
        **build_trainer_kwargs(
            tokenizer=tokenizer,
            model=model,
            training_args=training_args,
            tokenized_train=tokenized_train,
            tokenized_valid=tokenized_valid,
            collator=collator,
        )
    )
    trainer.train()
    output_dir = str(config["output_dir"])
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(config["output_dir"])
    remove_training_artifacts(output_dir)

if __name__ == "__main__":
    main()
