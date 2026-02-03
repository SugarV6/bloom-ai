"""
LLM-based question generation. Produces Bloom-style questions (target level in prompt).
CS499: Output is then filtered by the binary quality classifier; we do NOT train Bloom levels.
"""
import json
import os
from app.llm.client import get_client
from app.llm.prompts import build_prompt


def generate_questions(topic: str, bloom: str, qtype: str = "mcq", n: int = 5) -> list:
    """
    Call LLM to generate n questions. Returns list of item dicts with keys
    question, choices, correct_index, explanation, bloom_justification, quality_flags.
    """
    client = get_client()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if not deployment:
        raise RuntimeError("AZURE_OPENAI_DEPLOYMENT must be set in .env")
    prompt = build_prompt(topic, bloom, qtype, n)
    resp = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "Return only JSON. No markdown."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=1800,
    )
    content = resp.choices[0].message.content or ""
    data = json.loads(content)
    return data.get("items", [])
