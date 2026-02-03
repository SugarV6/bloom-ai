"""
FastAPI app for Bloom AI. Generation pipeline: LLM -> parse JSON -> quality classifier filter -> response.
CS499: Binary classifier (trained on question/label dataset) filters out invalid (label=0) questions.
Bloom level is prompt-only; we do NOT train or claim Bloom classification.
"""
import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from app.llm.generator import generate_questions
from app.quality.classifier import filter_valid_items

app = FastAPI(title="Bloom AI (Azure)", version="0.1.0")


class GenerateRequest(BaseModel):
    topic: str = Field(..., min_length=2)
    bloom: str = Field(..., min_length=2)
    qtype: str = Field("mcq")
    n: int = Field(5, ge=1, le=10)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    }


@app.post("/generate")
def generate(req: GenerateRequest):
    """
    Generate questions via LLM, then filter by binary quality classifier.
    Only questions predicted as valid (label=1) are returned.
    """
    try:
        # 1. LLM generation (Bloom target in prompt only; no Bloom training)
        items = generate_questions(
            topic=req.topic,
            bloom=req.bloom,
            qtype=req.qtype,
            n=req.n,
        )
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Model did not return valid JSON: {e}")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 2. Filter: remove questions predicted as invalid (label=0) by supervised classifier
    try:
        filtered = filter_valid_items(items, question_key="question")
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail="Quality classifier model not found. Run: python scripts/train_quality_classifier.py",
        ) from e

    return {
        "topic": req.topic,
        "target_bloom": req.bloom,
        "type": req.qtype,
        "n_requested": req.n,
        "n_returned": len(filtered),
        "items": filtered,
    }
