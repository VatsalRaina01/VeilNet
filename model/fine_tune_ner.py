"""
VeilNet NER Fine-Tuning Script
==============================
Fine-tunes dslim/bert-base-NER on ai4privacy/pii-masking-300k
to improve PII detection and reduce false positives.

Usage:
    # On Google Colab / Kaggle (free GPU):
    !pip install transformers datasets seqeval accelerate
    !python fine_tune_ner.py

    # Locally with GPU:
    python fine_tune_ner.py

    # With custom args:
    python fine_tune_ner.py --epochs 5 --batch_size 8 --max_samples 50000
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

# =============================================================================
# Label Mapping: ai4privacy labels → VeilNet simplified labels
# =============================================================================
# The ai4privacy dataset has 19+ fine-grained PII labels.
# We consolidate them into labels that match VeilNet's detection categories.

AI4PRIVACY_TO_VEILNET = {
    # Person names
    "FIRSTNAME": "PER",
    "LASTNAME": "PER",
    "MIDDLENAME": "PER",
    "PREFIX": "PER",
    "TITLE": "PER",
    # Organizations
    "COMPANY": "ORG",
    "ORGANIZATION": "ORG",
    # Locations
    "CITY": "LOC",
    "STATE": "LOC",
    "COUNTRY": "LOC",
    "STREET": "LOC",
    "ZIPCODE": "LOC",
    "COUNTY": "LOC",
    "BUILDINGNUMBER": "LOC",
    "SECONDARYADDRESS": "LOC",
    # Contact
    "EMAIL": "EMAIL",
    "TEL": "PHONE",
    "PHONE": "PHONE",
    "PHONENUMBER": "PHONE",
    # Financial / Identity
    "SOCIALNUMBER": "SSN",
    "SSN": "SSN",
    "CREDITCARDNUMBER": "CREDITCARD",
    "CREDITCARDCVV": "CREDITCARD",
    "IBAN": "FINANCIAL",
    "ACCOUNTNUMBER": "FINANCIAL",
    "BITCOINADDRESS": "FINANCIAL",
    "ETHEREUMADDRESS": "FINANCIAL",
    # Dates
    "BOD": "DATE",
    "DATE": "DATE",
    "DOB": "DATE",
    "DATEOFBIRTH": "DATE",
    "TIME": "DATE",
    # Digital / Other
    "IP": "DIGITAL",
    "IPV4": "DIGITAL",
    "IPV6": "DIGITAL",
    "URL": "DIGITAL",
    "USERNAME": "DIGITAL",
    "USERAGENT": "DIGITAL",
    "PASSWORD": "DIGITAL",
    "MAC": "DIGITAL",
    "IMEI": "DIGITAL",
    # Misc
    "GENDER": "MISC",
    "AGE": "MISC",
    "JOBTYPE": "MISC",
    "JOBTITLE": "MISC",
    "JOBAREA": "MISC",
    "VEHICLEVIN": "MISC",
    "VEHICLEVRM": "MISC",
    "CURRENCYCODE": "MISC",
    "CURRENCYSYMBOL": "MISC",
    "CURRENCYNAME": "MISC",
    "AMOUNT": "MISC",
    "NUMBER": "MISC",
    "NEARBYGPSCOORDINATE": "MISC",
    "LITECOINADDRESS": "MISC",
    "MASKEDNUMBER": "MISC",
    "DISPLAYNAME": "PER",
}

# Build the BIO label list for our simplified set
VEILNET_ENTITY_TYPES = [
    "PER", "ORG", "LOC", "EMAIL", "PHONE", "SSN",
    "CREDITCARD", "FINANCIAL", "DATE", "DIGITAL", "MISC",
]

# Create label list: O, B-PER, I-PER, B-ORG, I-ORG, ...
LABEL_LIST = ["O"]
for ent in VEILNET_ENTITY_TYPES:
    LABEL_LIST.append(f"B-{ent}")
    LABEL_LIST.append(f"I-{ent}")

LABEL_TO_ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

NUM_LABELS = len(LABEL_LIST)


def remap_bio_label(bio_label: str) -> str:
    """
    Remap an ai4privacy BIO label to VeilNet's simplified label set.
    e.g. 'B-FIRSTNAME' → 'B-PER', 'I-CREDITCARDNUMBER' → 'I-CREDITCARD'
    """
    if bio_label == "O":
        return "O"

    prefix, entity = bio_label.split("-", 1)
    entity_upper = entity.upper()

    veilnet_type = AI4PRIVACY_TO_VEILNET.get(entity_upper)
    if veilnet_type is None:
        # Fallback: try case-insensitive match
        for key, val in AI4PRIVACY_TO_VEILNET.items():
            if key.upper() == entity_upper:
                veilnet_type = val
                break

    if veilnet_type is None:
        veilnet_type = "MISC"

    return f"{prefix}-{veilnet_type}"


def prepare_dataset(max_samples: int = 10000, seed: int = 42):
    """
    Load ai4privacy dataset, filter to English, and prepare for training.
    Converts span-level annotations to token-level BIO labels.
    """
    print("📦 Loading ai4privacy/pii-masking-300k dataset...")
    dataset = load_dataset("ai4privacy/pii-masking-300k", trust_remote_code=True)
    train_data = dataset["train"]

    # Filter to English only
    print("🌐 Filtering to English samples...")
    train_data = train_data.filter(lambda x: x.get("language", "").lower() == "english")
    print(f"   Found {len(train_data)} English samples")

    # Subsample for faster training
    if max_samples and len(train_data) > max_samples:
        train_data = train_data.shuffle(seed=seed).select(range(max_samples))
        print(f"   Using {max_samples} samples for training")

    # Split into train/validation (90/10)
    split = train_data.train_test_split(test_size=0.1, seed=seed)
    print(f"   Train: {len(split['train'])}, Validation: {len(split['test'])}")

    return split


def tokenize_and_align_labels(examples, tokenizer, max_length=128):
    """
    Tokenize text and align span-level PII annotations to BIO token labels.

    The ai4privacy dataset provides:
    - source_text: the raw text
    - privacy_mask: list of {value, start, end, label} dicts

    We tokenize source_text with the BERT tokenizer and build BIO labels
    by mapping character-level spans to token positions.
    """
    texts = examples["source_text"]
    all_privacy_masks = examples["privacy_mask"]

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_offsets_mapping=True,
    )

    all_labels = []

    for i, offsets in enumerate(tokenized["offset_mapping"]):
        # Parse privacy_mask — it may be a string or a list
        privacy_mask = all_privacy_masks[i]
        if isinstance(privacy_mask, str):
            try:
                privacy_mask = json.loads(privacy_mask)
            except (json.JSONDecodeError, TypeError):
                privacy_mask = []
        if privacy_mask is None:
            privacy_mask = []

        # Build character-to-label mapping from spans
        text = texts[i] if texts[i] else ""
        char_labels = ["O"] * len(text)

        for span in privacy_mask:
            if not isinstance(span, dict):
                continue
            label = span.get("label", "")
            start = span.get("start", 0)
            end = span.get("end", 0)

            if not label or start >= end:
                continue

            veilnet_type = AI4PRIVACY_TO_VEILNET.get(label.upper(), "MISC")

            # Assign BIO tags at character level
            for char_idx in range(start, min(end, len(text))):
                if char_idx == start:
                    char_labels[char_idx] = f"B-{veilnet_type}"
                else:
                    char_labels[char_idx] = f"I-{veilnet_type}"

        # Map character-level labels to token-level labels
        labels = []
        for offset in offsets:
            start, end = offset
            if start == 0 and end == 0:
                # Special tokens ([CLS], [SEP], [PAD])
                labels.append(-100)
            else:
                # Use the label of the first character in this token's span
                token_label = char_labels[start] if start < len(char_labels) else "O"
                labels.append(LABEL_TO_ID.get(token_label, 0))

        all_labels.append(labels)

    tokenized["labels"] = all_labels
    # Remove offset_mapping — Trainer doesn't expect it
    del tokenized["offset_mapping"]

    return tokenized


def compute_metrics(pred):
    """Compute precision, recall, F1 using seqeval."""
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)

    # Convert IDs back to label strings, ignoring -100
    true_labels = []
    pred_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        true_seq = []
        pred_seq_clean = []
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                true_seq.append(ID_TO_LABEL.get(l, "O"))
                pred_seq_clean.append(ID_TO_LABEL.get(p, "O"))
        true_labels.append(true_seq)
        pred_labels.append(pred_seq_clean)

    return {
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels),
    }


def train(
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_samples: int = 10000,
    max_length: int = 128,
    output_dir: str = None,
):
    """Run the full fine-tuning pipeline."""

    if output_dir is None:
        output_dir = str(Path(__file__).parent / "veilnet-ner-model")

    checkpoint_dir = str(Path(__file__).parent / "checkpoints")

    base_model = "dslim/bert-base-NER"

    # 1. Load tokenizer & model
    print(f"\n🤖 Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=NUM_LABELS,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,  # The classifier head size changes
    )
    print(f"   Model has {model.num_parameters():,} parameters")
    print(f"   Labels: {NUM_LABELS} ({len(VEILNET_ENTITY_TYPES)} entity types + O)")

    # 2. Prepare dataset
    dataset = prepare_dataset(max_samples=max_samples)

    # 3. Tokenize
    print("\n✂️  Tokenizing and aligning labels...")
    tokenized_train = dataset["train"].map(
        lambda x: tokenize_and_align_labels(x, tokenizer, max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing train",
    )
    tokenized_val = dataset["test"].map(
        lambda x: tokenize_and_align_labels(x, tokenizer, max_length),
        batched=True,
        remove_columns=dataset["test"].column_names,
        desc="Tokenizing validation",
    )

    print(f"   Train tokens: {len(tokenized_train)}, Val tokens: {len(tokenized_val)}")

    # 4. Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # 5. Training arguments
    print("\n🏋️  Setting up training...")
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=True,  # Mixed precision for GPU efficiency
        report_to="none",  # Disable wandb/tensorboard
        dataloader_num_workers=0,  # Colab-safe
        save_total_limit=2,  # Keep only 2 best checkpoints
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 7. Train!
    print("\n🚀 Starting training...\n")
    train_result = trainer.train()

    # 8. Evaluate
    print("\n📊 Evaluating on validation set...")
    eval_results = trainer.evaluate()
    print(f"\n   Results:")
    print(f"   Precision: {eval_results.get('eval_precision', 0):.4f}")
    print(f"   Recall:    {eval_results.get('eval_recall', 0):.4f}")
    print(f"   F1:        {eval_results.get('eval_f1', 0):.4f}")

    # 9. Save final model
    print(f"\n💾 Saving fine-tuned model to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save label mapping for reference
    label_map_path = os.path.join(output_dir, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump({
            "label_to_id": LABEL_TO_ID,
            "id_to_label": {str(k): v for k, v in ID_TO_LABEL.items()},
            "entity_types": VEILNET_ENTITY_TYPES,
        }, f, indent=2)

    # Save eval results
    eval_path = os.path.join(output_dir, "eval_results.json")
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\n✅ Fine-tuning complete!")
    print(f"   Model saved to: {output_dir}")
    print(f"   Label map saved to: {label_map_path}")
    print(f"   Eval results saved to: {eval_path}")

    # 10. Detailed classification report
    print("\n📋 Detailed Classification Report:")
    predictions = trainer.predict(tokenized_val)
    preds = np.argmax(predictions.predictions, axis=2)
    labels = predictions.label_ids

    true_labels = []
    pred_labels = []
    for pred_seq, label_seq in zip(preds, labels):
        true_seq = []
        pred_seq_clean = []
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                true_seq.append(ID_TO_LABEL.get(l, "O"))
                pred_seq_clean.append(ID_TO_LABEL.get(p, "O"))
        true_labels.append(true_seq)
        pred_labels.append(pred_seq_clean)

    report = classification_report(true_labels, pred_labels)
    print(report)

    # Save report
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    return eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BERT NER for VeilNet")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=50000,
                        help="Max training samples (0 = all)")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Max token sequence length")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for the model")

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_samples=args.max_samples if args.max_samples > 0 else None,
        max_length=args.max_length,
        output_dir=args.output_dir,
    )
