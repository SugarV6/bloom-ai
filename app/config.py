"""
Application configuration. Paths for dataset and trained quality classifier.
CS499: Binary question-quality classifier uses dataset with schema:
  question, label, is_labeled, source, dataset_type
Label: 1 = valid/content-related, 0 = invalid/meta/vague. Not Bloom-level labels.
"""
import os
from pathlib import Path

# Project root (parent of app/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Labeled dataset path: CSV with columns question, label, is_labeled, source, dataset_type
DATASET_PATH = os.getenv("BLOOM_DATASET_PATH") or str(PROJECT_ROOT / "data" / "all_questions_deduplicated.csv")

# Trained binary classifier artifact (joblib): used for filtering after LLM generation
QUALITY_MODEL_PATH = os.getenv("BLOOM_QUALITY_MODEL_PATH") or str(PROJECT_ROOT / "models" / "quality_classifier.joblib")
