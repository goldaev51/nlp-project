from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .data import prepare_goemotions_mood_dataset
from .labels import ID_TO_MOOD, MOOD_TO_ID


def _build_hf_datasets_from_pandas() -> DatasetDict:
    train_df, val_df, test_df = prepare_goemotions_mood_dataset()

    train_df = train_df[["text", "mood_id"]].copy()
    val_df = val_df[["text", "mood_id"]].copy()
    test_df = test_df[["text", "mood_id"]].copy()

    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)
    test_ds = Dataset.from_pandas(test_df, preserve_index=False)

    train_ds = train_ds.rename_column("mood_id", "labels")
    val_ds = val_ds.rename_column("mood_id", "labels")
    test_ds = test_ds.rename_column("mood_id", "labels")

    ds_dict = DatasetDict(
        {
            "train": train_ds,
            "validation": val_ds,
            "test": test_ds,
        }
    )
    return ds_dict


def _tokenize_datasets(
    ds_dict: DatasetDict,
    tokenizer: AutoTokenizer,
    max_length: int = 128,
) -> DatasetDict:
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
        )

    tokenized = ds_dict.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    return tokenized


def _compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
    }


def train_emotion_transformer(
    model_name: str = "distilbert-base-uncased",
    output_dir: str = "models/emotion_transformer",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 32,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_length: int = 128,
) -> None:
    ds_dict = _build_hf_datasets_from_pandas()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_ds = _tokenize_datasets(ds_dict, tokenizer, max_length=max_length)

    num_labels = len(MOOD_TO_ID)
    id2label = {i: mood for mood, i in MOOD_TO_ID.items()}
    label2id = {v: k for k, v in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=100,
        report_to=[],
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
    )

    trainer.train()

    eval_metrics = trainer.evaluate()
    print("Validation metrics:", eval_metrics)

    test_preds = trainer.predict(tokenized_ds["test"])
    test_logits = test_preds.predictions
    test_labels = test_preds.label_ids
    test_pred_ids = np.argmax(test_logits, axis=-1)

    test_acc = accuracy_score(test_labels, test_pred_ids)
    test_f1_macro = f1_score(test_labels, test_pred_ids, average="macro")

    print("\n=== Test metrics (transformer) ===")
    print("Test accuracy:", test_acc)
    print("Test macro F1:", test_f1_macro)
    print("\nTest classification report:")
    print(
        classification_report(
            test_labels,
            test_pred_ids,
            target_names=[ID_TO_MOOD[i] for i in sorted(ID_TO_MOOD.keys())],
        )
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    trainer.save_model(out_path)
    tokenizer.save_pretrained(out_path)

    print(f"\nTransformer emotion model saved to: {out_path}")


def load_emotion_transformer(
    model_dir: str = "models/emotion_transformer",
) -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model


@torch.no_grad()
def predict_mood_transformer(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: str | None = None,
) -> Dict[str, Any]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    outputs = model(**encoded)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    pred_id = int(np.argmax(probs))
    mood = ID_TO_MOOD[pred_id]

    probs_dict = {ID_TO_MOOD[i]: float(probs[i]) for i in range(len(probs))}

    return {
        "mood_id": pred_id,
        "mood": mood,
        "probs": probs_dict,
    }


if __name__ == "__main__":
    train_emotion_transformer()
