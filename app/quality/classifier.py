"""
Inference module for the binary question-quality classifier.
CS499: Used as a filter AFTER LLM generation. Questions predicted as invalid (0) are removed.
Loads the model trained by scripts/train_quality_classifier.py.
"""
from pathlib import Path
from typing import List, Union

import joblib

from app.config import QUALITY_MODEL_PATH


# Lazy-loaded pipeline to avoid loading at import time
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        path = Path(QUALITY_MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"Quality classifier model not found at {QUALITY_MODEL_PATH}. "
                "Run: python scripts/train_quality_classifier.py"
            )
        _pipeline = joblib.load(path)
    return _pipeline


def predict(question: str) -> int:
    """
    Predict binary quality label for a single question.
    Returns: 1 = valid/content-related (keep), 0 = invalid/meta/vague (filter out).
    """
    pipe = _get_pipeline()
    # pipeline expects 1D array of strings
    return int(pipe.predict([question])[0])


def predict_batch(questions: List[str]) -> List[int]:
    """Predict labels for multiple questions. Returns list of 0s and 1s."""
    if not questions:
        return []
    pipe = _get_pipeline()
    return [int(x) for x in pipe.predict(questions)]


def filter_valid_items(items: List[dict], question_key: str = "question") -> List[dict]:
    """
    Filter a list of item dicts (e.g. LLM-generated questions) to only those
    predicted as valid (label=1). Uses item[question_key] as text for classification.
    """
    if not items:
        return []
    texts = [item.get(question_key, "") or "" for item in items]
    labels = predict_batch(texts)
    return [item for item, lab in zip(items, labels) if lab == 1]
