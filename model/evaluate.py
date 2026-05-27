"""
VeilNet NER Model Evaluation
=============================
Evaluates a fine-tuned NER model on test data and generates
a detailed classification report.

Usage:
    python evaluate.py --model_path ./veilnet-ner-model/
    python evaluate.py --model_path ./veilnet-ner-model/ --max_samples 5000
    python evaluate.py --model_path ./veilnet-ner-model/ --test_texts "John Smith works at Google"
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
    pipeline,
)
from seqeval.metrics import classification_report, f1_score

# Import label mappings from fine_tune_ner
sys.path.insert(0, str(Path(__file__).parent))
from fine_tune_ner import (
    AI4PRIVACY_TO_VEILNET,
    LABEL_TO_ID,
    ID_TO_LABEL,
    LABEL_LIST,
    tokenize_and_align_labels,
)


def evaluate_on_dataset(
    model_path: str,
    max_samples: int = 1000,
    max_length: int = 128,
):
    """
    Evaluate the fine-tuned model on a held-out validation set.
    Returns a detailed classification report.
    """
    print(f"\n📊 Evaluating model: {model_path}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    # Load and prepare evaluation data
    print("📦 Loading evaluation data...")
    dataset = load_dataset("ai4privacy/pii-masking-300k", trust_remote_code=True)
    eval_data = dataset["train"].filter(
        lambda x: x.get("language", "").lower() == "english"
    )

    # Use the last portion as eval (not seen during training)
    total = len(eval_data)
    eval_start = max(0, total - max_samples)
    eval_data = eval_data.select(range(eval_start, total))
    print(f"   Evaluating on {len(eval_data)} samples")

    # Tokenize
    tokenized = eval_data.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, max_length),
        batched=True,
        remove_columns=eval_data.column_names,
        desc="Tokenizing",
    )

    # Run predictions
    print("🔮 Running predictions...")
    from transformers import Trainer

    trainer = Trainer(model=model, processing_class=tokenizer)
    predictions = trainer.predict(tokenized)
    preds = np.argmax(predictions.predictions, axis=2)
    labels = predictions.label_ids

    # Convert to label strings
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

    # Classification report
    report = classification_report(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)
    print(f"\nOverall F1: {f1:.4f}")

    # Save report
    output_dir = model_path
    report_path = os.path.join(output_dir, "eval_report.txt")
    with open(report_path, "w") as f:
        f.write("VeilNet NER Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
        f.write(f"\nOverall F1: {f1:.4f}\n")
    print(f"\n💾 Report saved to: {report_path}")

    return report, f1


def test_on_texts(model_path: str, texts: list[str]):
    """
    Run the fine-tuned model on custom text samples and display results.
    Useful for quick sanity checks and false-positive analysis.
    """
    print(f"\n🔬 Testing model on {len(texts)} custom texts...")
    print(f"   Model: {model_path}\n")

    ner = pipeline(
        "ner",
        model=model_path,
        aggregation_strategy="simple",
    )

    for i, text in enumerate(texts):
        print(f"─── Text {i + 1} {'─' * 50}")
        print(f"  Input: {text[:200]}{'...' if len(text) > 200 else ''}\n")

        results = ner(text)
        if not results:
            print("  No entities detected.\n")
            continue

        for entity in results:
            score = entity["score"]
            word = entity["word"]
            group = entity["entity_group"]
            emoji = "✅" if score >= 0.85 else "⚠️" if score >= 0.70 else "❌"
            print(f"  {emoji} [{group}] \"{word}\" (score: {score:.4f})")
        print()


# Default test texts for false-positive analysis
DEFAULT_TEST_TEXTS = [
    # Should detect PER + ORG + LOC
    "John Smith works at Google in New York City. His email is john@google.com.",
    # Should detect PER only — common resume-like text
    "Vinit Sharma is a software engineer with 5 years of experience in Python and Machine Learning.",
    # Should NOT flag common words as entities (false positive test)
    "The project summary includes technical skills, education, and experience sections.",
    # Should detect PER + ORG
    "Dear Mr. Patel, We are pleased to offer you a position at Amazon Web Services.",
    # Address + SSN + credit card (regex territory, but NER shouldn't false-flag)
    "SSN: 123-45-6789. Credit card: 4111-1111-1111-1111. Address: 123 Main Street.",
    # Should detect multiple PER + ORG
    "Dr. Priya Sharma reviewed the case with Prof. Rahul Gupta at Stanford University.",
    # Tricky text — common words that BERT sometimes falsely flags
    "Page 1 of 10. Section: Profile Summary. Skills: Python, JavaScript, React.",
    # Names in context vs generic words
    "Patient Name: Ananya Verma. Hospital: Apollo Hospitals. City: Mumbai.",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VeilNet NER model")
    parser.add_argument(
        "--model_path", type=str,
        default=str(Path(__file__).parent / "veilnet-ner-model"),
        help="Path to fine-tuned model directory",
    )
    parser.add_argument(
        "--max_samples", type=int, default=1000,
        help="Max evaluation samples",
    )
    parser.add_argument(
        "--test_texts", type=str, nargs="*",
        help="Custom texts to test (space-separated, use quotes)",
    )
    parser.add_argument(
        "--skip_dataset", action="store_true",
        help="Skip dataset evaluation, only run text tests",
    )

    args = parser.parse_args()

    # Run dataset evaluation
    if not args.skip_dataset:
        try:
            evaluate_on_dataset(args.model_path, args.max_samples)
        except Exception as e:
            print(f"⚠️  Dataset evaluation failed: {e}")
            print("   Run with --skip_dataset to only test on custom texts.")

    # Run custom text tests
    texts = args.test_texts if args.test_texts else DEFAULT_TEST_TEXTS
    test_on_texts(args.model_path, texts)
