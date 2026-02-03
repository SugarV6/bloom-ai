"""
Train the binary question-quality classifier on the labeled dataset.
CS499: Supervised learning component. Dataset schema: question, label, is_labeled, source, dataset_type.
Label: 1 = valid/content-related, 0 = invalid/meta/vague. NOT Bloom-level labels.
Output: Saved model (joblib) and evaluation metrics (accuracy, precision, recall, F1).
"""
import json
import sys
from pathlib import Path

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

from app.config import DATASET_PATH, QUALITY_MODEL_PATH


def load_dataset(path: str):
    """Load CSV with columns question, label, is_labeled, source, dataset_type. Use labeled rows only."""
    df = pd.read_csv(path)
    required = ["question", "label"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Dataset must have column '{col}'. Found: {list(df.columns)}")
    # Use rows with valid binary label
    df = df.dropna(subset=["question", "label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])]
    return df["question"].astype(str), df["label"]


def main():
    path = Path(DATASET_PATH)
    if not path.exists():
        print(f"Dataset not found: {path}")
        print("Create data/questions.csv with columns: question, label, is_labeled, source, dataset_type")
        sys.exit(1)

    X, y = load_dataset(DATASET_PATH)
    if len(X) < 30:
        print("Warning: very few samples. Need at least ~30 for train/val/test split.")
    # Split: 70% train, 15% validation, 15% test (stratified)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15 / 0.85, random_state=42, stratify=y_temp
    )
    print(f"Split: train {len(X_train)} ({100*len(X_train)/len(X):.1f}%), val {len(X_val)} ({100*len(X_val)/len(X):.1f}%), test {len(X_test)} ({100*len(X_test)/len(X):.1f}%)")

    # Explainable pipeline: TF-IDF + LogisticRegression (no LLM fine-tuning)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=500, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    # Validation metrics (for monitoring; not used for model selection in this script)
    y_val_pred = pipeline.predict(X_val)
    val_metrics = {
        "accuracy": round(accuracy_score(y_val, y_val_pred), 4),
        "precision": round(precision_score(y_val, y_val_pred, zero_division=0), 4),
        "recall": round(recall_score(y_val, y_val_pred, zero_division=0), 4),
        "f1": round(f1_score(y_val, y_val_pred, zero_division=0), 4),
    }
    print("Validation metrics:")
    print(json.dumps(val_metrics, indent=2))

    # Test metrics (final evaluation; reported and saved)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
    print("Test metrics (final evaluation):")
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, y_pred, target_names=["invalid (0)", "valid (1)"]))

    # Save model
    model_dir = Path(QUALITY_MODEL_PATH).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, QUALITY_MODEL_PATH)
    print(f"Model saved to {QUALITY_MODEL_PATH}")

    # Save metrics next to model for documentation (test = final; validation = monitoring)
    metrics_path = model_dir / "quality_classifier_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"test": metrics, "validation": val_metrics, "split_pct": {"train": 70, "validation": 15, "test": 15}}, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
