# src/llm_backend/__init__.py
import sys

from src.core.types import Message
from src.llm_backend.base import (
    chat, is_ollama,
    OLLAMA_INFRA, OLLAMA_BASE_URL
)
from src.llm_backend.embeddings import get_embedding, cosine_similarity

# Initialization Check (Moved here to prevent circular imports)
if OLLAMA_INFRA:
    from src.llm_backend.ollama import _check_ollama
    if _check_ollama():
        print("[LLM Backend] OLLAMA_INFRA=True  ·  Ollama reachable ✓")
    else:
        print(
            f"\n[LLM Backend] Ollama is NOT responding at {OLLAMA_BASE_URL}\n")
        sys.exit(1)
else:
    try:
        import llama_cpp
    except ImportError:
        print(
            "\n[LLM Backend] llama-cpp-python is not installed. Switching to Ollama HTTP mode.\n")
        OLLAMA_INFRA = True
        if not _check_ollama():
            sys.exit(1)
        else:
            print("[LLM Backend] Ollama reachable ✓ (Fallback)")
    else:
        from src.llm_backend.llamacpp import _scan_gguf, _llamacpp_models
        _llamacpp_models.update(_scan_gguf())
        if not _llamacpp_models:
            print(
                "\n[LLM Backend] No matching .gguf files found. Switching to Ollama HTTP mode.\n")
            OLLAMA_INFRA = True
            if not _check_ollama():
                sys.exit(1)
        else:
            print(
                f"[LLM Backend] llama.cpp ready ✓  ·  models: {list(_llamacpp_models.keys())}")
