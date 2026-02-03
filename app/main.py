from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Bloom AI", version="0.1.0")


class GenerateRequest(BaseModel):
    topic: str
    bloom: str
    qtype: str = "mcq"
    n: int = 5


@app.get("/health")
def health():
    return {"status": "ok", "service": "bloom-ai"}


@app.post("/generate")
def generate(req: GenerateRequest):
    return {
        "topic": req.topic,
        "target_bloom": req.bloom,
        "type": req.qtype,
        "n": req.n,
        "items": [
            {
                "question": "Dummy question (LLM not connected yet).",
                "choices": ["A", "B", "C", "D"],
                "correct_index": 0,
                "explanation": "This is a placeholder response.",
                "bloom_justification": "Placeholder justification.",
                "quality_flags": ["placeholder"],
            }
        ],
    }
