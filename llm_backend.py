"""
llm_backend.py — Single-toggle LLM backend (Ollama or llama.cpp).
"""
import json
import os
import sys
import threading
import time
import urllib.request
import urllib.error
import inspect
import typing
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OLLAMA_INFRA: bool = os.environ.get(
    "OLLAMA_INFRA", "True").strip().lower() in ("true", "1", "yes", "on",)
OLLAMA_BASE_URL: str = os.environ.get(
    "OLLAMA_BASE_URL", "http://localhost:11434")
MODELS_DIR: str = os.environ.get("LLAMACPP_MODELS_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models"),)
LLAMACPP_N_CTX: int = int(os.environ.get("LLAMACPP_N_CTX", "8192"))
LLAMACPP_N_GPU_LAYERS: int = int(os.environ.get("LLAMACPP_N_GPU_LAYERS", "-1"))
LLAMACPP_N_THREADS: int = int(os.environ.get("LLAMACPP_N_THREADS", "0"))

MODEL_MAP: Dict[str, Dict[str, str]] = {
    "ornith": {"gguf": "ornith.gguf", "ollama": "ornith"},
}
assert len(MODEL_MAP) == len(set(MODEL_MAP)), "Duplicate keys in MODEL_MAP"

# Thread safety lock for local llama.cpp execution
_llamacpp_lock = threading.Lock()


@dataclass
class Message:
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    thinking: Optional[str] = None


def _check_ollama() -> bool:
    try:
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/tags", headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def _scan_gguf() -> Dict[str, str]:
    found: Dict[str, str] = {}
    if not os.path.isdir(MODELS_DIR):
        return found
    disk_files: Dict[str, str] = {}
    for fname in os.listdir(MODELS_DIR):
        if fname.lower().endswith(".gguf"):
            disk_files[fname[:-5].lower()
                       ] = os.path.abspath(os.path.join(MODELS_DIR, fname))
    for logical, mapping in MODEL_MAP.items():
        stem = mapping["gguf"].lower().replace(".gguf", "")
        if stem in disk_files:
            found[logical] = disk_files[stem]
    return found


_llamacpp_models: Dict[str, str] = {}
_llamacpp_instances: Dict[str, Any] = {}

if OLLAMA_INFRA:
    if _check_ollama():
        print("[LLM Backend] OLLAMA_INFRA=True  ·  Ollama reachable ✓")
    else:
        print(
            f"\n[LLM Backend] Ollama is NOT responding at {OLLAMA_BASE_URL}\n")
        sys.exit(1)
else:
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        print("\n[LLM Backend] llama-cpp-python is not installed.\n")
        sys.exit(1)

    _llamacpp_models = _scan_gguf()
    if not _llamacpp_models:
        print("\n[LLM Backend] No matching .gguf files found.\n")
        sys.exit(1)

    print(
        f"[LLM Backend] llama.cpp ready ✓  ·  models: {list(_llamacpp_models.keys())}")


def is_ollama() -> bool:
    return OLLAMA_INFRA


def available_models() -> List[str]:
    if OLLAMA_INFRA:
        return list(MODEL_MAP.keys())
    return list(_llamacpp_models.keys())


def _get_llama(model_name: str):
    if model_name in _llamacpp_instances:
        return _llamacpp_instances[model_name]

    path = _llamacpp_models.get(model_name)
    if not path:
        raise ValueError(f"Model '{model_name}' not found.")

    import llama_cpp
    print(f"[LLM Backend] Loading '{model_name}'...", end=" ", flush=True)
    t0 = time.time()
    kwargs: Dict[str, Any] = {
        "model_path": path, "n_ctx": LLAMACPP_N_CTX, "n_gpu_layers": LLAMACPP_N_GPU_LAYERS,
        "verbose": False, "chat_format": "chatml",
    }
    if LLAMACPP_N_THREADS > 0:
        kwargs["n_threads"] = LLAMACPP_N_THREADS

    instance = llama_cpp.Llama(**kwargs)
    print(f"done ({time.time() - t0:.1f}s)")
    _llamacpp_instances[model_name] = instance
    return instance


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
        out.append({
            "type": "function",
            "function": {
                "name": t.__name__, "description": doc.split("\n")[0] if doc else "",
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        })
    return out or None


def chat(
    model: str, messages: List[Dict[str, Any]], tools: Optional[List[Any]] = None,
    think: bool = False, stream: bool = True
) -> Generator[Dict[str, Any], None, None]:
    if OLLAMA_INFRA:
        yield from _chat_ollama(model, messages, tools, think, stream)
    else:
        yield from _chat_llamacpp(model, messages, tools, think, stream)


def _chat_ollama(model, messages, tools, think, stream):
    from ollama import chat as ollama_chat
    resolved = MODEL_MAP.get(model, {}).get("ollama", model)
    # Ollama server handles concurrent requests natively
    for chunk in ollama_chat(model=resolved, messages=messages, tools=tools, think=think, stream=stream):
        yield chunk


def _chat_llamacpp(model, messages, tools, think, stream):
    instance = _get_llama(model)
    openai_tools = _tools_to_openai(tools)

    kwargs: Dict[str, Any] = {
        "messages": messages, "stream": True, "temperature": 0.6, "top_p": 0.9,
    }
    if openai_tools:
        kwargs["tools"] = openai_tools
        kwargs["tool_choice"] = "auto"

    # Serialize llama.cpp calls to prevent thread crashes (segfaults)
    with _llamacpp_lock:
        raw = instance.create_chat_completion(**kwargs)
        tc_buf: Dict[int, Dict[str, Any]] = {}

        for chunk in raw:
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content") or None

            for frag in (delta.get("tool_calls") or []):
                idx = frag.get("index", 0)
                if idx not in tc_buf:
                    tc_buf[idx] = {"id": frag.get("id", f"call_{idx}"), "type": "function", "function": {
                        "name": "", "arguments": ""}}
                fn = frag.get("function", {})
                if fn.get("name"):
                    tc_buf[idx]["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    tc_buf[idx]["function"]["arguments"] += fn["arguments"]

            if content:
                yield {"message": {"role": "assistant", "content": content, "thinking": None, "tool_calls": None}}

    if tc_buf:
        final = []
        for idx in sorted(tc_buf):
            entry = tc_buf[idx]
            raw_args = entry["function"]["arguments"]
            try:
                parsed = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                parsed = {"_raw": raw_args}
            final.append({"id": entry["id"], "type": "function", "function": {
                         "name": entry["function"]["name"], "arguments": parsed}})
        yield {"message": {"role": "assistant", "content": None, "thinking": None, "tool_calls": final}}


def chat_sync(model: str, messages: List[Dict], tools=None, think=False) -> Message:
    if OLLAMA_INFRA:
        from ollama import chat as ollama_chat
        resolved = MODEL_MAP.get(model, {}).get("ollama", model)
        resp = ollama_chat(model=resolved, messages=messages,
                           tools=tools, think=think, stream=False)
        m = resp.get("message", resp) if isinstance(resp, dict) else resp
        return Message(
            role=getattr(m, "role", "assistant"), content=getattr(m, "content", None),
            tool_calls=getattr(m, "tool_calls", None), thinking=getattr(m, "thinking", None),
        )
    else:
        parts_c, parts_t, final_tc = [], [], None
        for chunk in chat(model, messages, tools, think, stream=True):
            m = chunk["message"]
            if m.get("content"):
                parts_c.append(m["content"])
            if m.get("thinking"):
                parts_t.append(m["thinking"])
            if m.get("tool_calls") is not None:
                final_tc = m["tool_calls"]
        return Message(
            role="assistant", content="".join(parts_c) or None,
            tool_calls=final_tc, thinking="".join(parts_t) or None,
        )
