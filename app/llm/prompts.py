import json

def build_prompt(topic: str, bloom: str, qtype: str, n: int) -> str:
    example = {
        "topic": topic,
        "target_bloom": bloom,
        "type": qtype,
        "items": [
            {
                "question": "string",
                "choices": ["A", "B", "C", "D"],
                "correct_index": 0,
                "explanation": "string",
                "bloom_justification": "string",
                "quality_flags": ["no_issue"]
            }
        ]
    }

    return f"""
Return ONLY valid JSON. No markdown. No extra text.

Generate {n} {qtype.upper()} questions about:
{topic}

Target Bloom level: {bloom}

Rules:
- Always include exactly 4 choices (A-D).
- correct_index must be 0..3.
- bloom_justification must explain why the question matches the Bloom level.

Return JSON with the same structure as this example:
{json.dumps(example, indent=2)}
""".strip()
