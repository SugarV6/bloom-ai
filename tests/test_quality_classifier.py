"""
Tests for the binary question-quality classifier (inference only).
CS499: Classifier predicts 0=invalid, 1=valid. Filter removes predicted invalid.
"""
import pytest
from unittest.mock import patch, MagicMock


# Mock joblib so we don't need a trained model on disk for unit tests
@patch("app.quality.classifier.joblib.load")
def test_predict_returns_0_or_1(mock_load):
    mock_pipe = MagicMock()
    mock_pipe.predict.return_value = [1]
    mock_load.return_value = mock_pipe
    from app.quality.classifier import predict, _get_pipeline
    # Reset module-level cache so mock is used
    import app.quality.classifier as m
    m._pipeline = None
    assert predict("What is photosynthesis?") == 1
    mock_pipe.predict.return_value = [0]
    m._pipeline = None
    mock_load.return_value = mock_pipe
    assert predict("What?") == 0


@patch("app.quality.classifier._get_pipeline")
def test_filter_valid_items_keeps_only_label_1(mock_get):
    mock_pipe = MagicMock()
    mock_pipe.predict.return_value = [1, 0, 1]
    mock_get.return_value = mock_pipe
    from app.quality.classifier import filter_valid_items
    items = [
        {"question": "Q1", "choices": []},
        {"question": "Q2", "choices": []},
        {"question": "Q3", "choices": []},
    ]
    out = filter_valid_items(items)
    assert len(out) == 2
    assert out[0]["question"] == "Q1" and out[1]["question"] == "Q3"


def test_filter_valid_items_empty():
    from app.quality.classifier import filter_valid_items
    assert filter_valid_items([]) == []
