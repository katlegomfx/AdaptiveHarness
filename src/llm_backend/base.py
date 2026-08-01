# src/llm_backend/base.py
import os
import sys
import typing
import inspect
import json
import urllib.request
import urllib.error
from typing import Any, Dict, Generator, List, Optional
from src.core.types import Message

IS_TERMUX = os.environ.get("PREFIX", "").startswith(
    "/data/data/com.termux") or os.path.exists("/data/data/com.termux/files/usr")
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OLLAMA_INFRA: bool = os.environ.get(
    "OLLAMA_INFRA", "True").strip().lower() in ("true", "1", "yes", "on",)
if IS_TERMUX:
    OLLAMA_INFRA = True

OLLAMA_BASE_URL: str = os.environ.get(
    "OLLAMA_BASE_URL", "http://localhost:11434")
MODELS_DIR: str = os.environ.get(
    "LLAMACPP_MODELS_DIR", os.path.join(os.getcwd(), "llms"))
LLAMACPP_N_CTX: int = int(os.environ.get("LLAMACPP_N_CTX", "128000"))
LLAMACPP_N_GPU_LAYERS: int = int(os.environ.get("LLAMACPP_N_GPU_LAYERS", "-1"))
LLAMACPP_N_THREADS: int = int(os.environ.get("LLAMACPP_N_THREADS", "4"))

MODEL_MAP: Dict[str, Dict[str, str]] = {
    "ornith": {"gguf": "ornith.gguf", "ollama": "ornith"}}


def is_ollama() -> bool: return OLLAMA_INFRA

def _tools_to_openai(tools: Optional[List[Any]]) -> Optional[List[Dict]]:
    if not tools:
        return None
    out: List[Dict] = []
    TYPE_MAP = {int: "integer", float: "number",
                str: "string", bool: "boolean"}
    for t in tools:
        if isinstance(t, dict) and "function" in t:
            out.append(t)
            continue
        if not callable(t):
            continue
        sig = inspect.signature(t)
        try:
            hints = typing.get_type_hints(t)
        except Exception:
            hints = {}
        doc = inspect.getdoc(t) or ""
        props: Dict[str, Any] = {}
        required: List[str] = []
        for pname, param in sig.parameters.items():
            if pname == "kwargs":
                continue
            ptype = hints.get(pname, str)
            desc = f"Parameter {pname}"
            if hasattr(ptype, "__metadata__"):
                if ptype.__metadata__:
                    desc = ptype.__metadata__[0]
                ptype = ptype.__origin__
            props[pname] = {"type": TYPE_MAP.get(
                ptype, "string"), "description": desc}
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        out.append({"type": "function", "function": {"name": t.__name__, "description": doc.split("\n")[
                   0] if doc else "", "parameters": {"type": "object", "properties": props, "required": required}}})
    return out or None

def chat(model: str, messages: List[Dict[str, Any]], tools: Optional[List[Any]] = None, think: bool = False, stream: bool = True, temperature: float = 0.7) -> Generator[Dict[str, Any], None, None]:
    if OLLAMA_INFRA:
        from src.llm_backend.ollama import _chat_ollama
        yield from _chat_ollama(model, messages, tools, think, stream, temperature)
    else:
        from src.llm_backend.llamacpp import _chat_llamacpp
        yield from _chat_llamacpp(model, messages, tools, think, stream, temperature)


