import os
import json
import math
import urllib.request
from typing import List
from src.llm_backend.base import OLLAMA_BASE_URL, OLLAMA_INFRA, MODEL_MAP

EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")


def get_embedding(text: str, model: str = None) -> List[float]:
    if not text.strip():
        return []
    emb_model = model if model and model != "ornith" else EMBEDDING_MODEL
    if OLLAMA_INFRA:
        try:
            resolved = MODEL_MAP.get(emb_model, {}).get("ollama", emb_model)
            payload = {"model": resolved, "prompt": text}
            req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/embeddings", data=json.dumps(
                payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                return data.get("embedding", [])
        except Exception as e:
            print(f"[Embedding Error]: {e}")
            return []
    return []


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)