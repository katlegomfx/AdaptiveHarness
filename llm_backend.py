# llm_backend.py
"""
llm_backend.py — Single-toggle LLM backend (Ollama via urllib or llama.cpp) + Embeddings.
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
import math
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

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
MODELS_DIR: str = os.environ.get("LLAMACPP_MODELS_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models"))
LLAMACPP_N_CTX: int = int(os.environ.get("LLAMACPP_N_CTX", "8192"))
LLAMACPP_N_GPU_LAYERS: int = int(os.environ.get("LLAMACPP_N_GPU_LAYERS", "-1"))
LLAMACPP_N_THREADS: int = int(os.environ.get("LLAMACPP_N_THREADS", "0"))

MODEL_MAP: Dict[str, Dict[str, str]] = {
    "ornith": {"gguf": "ornith.gguf", "ollama": "ornith"},
}
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")

_llamacpp_lock = threading.Lock()
_llamacpp_models: Dict[str, str] = {}
_llamacpp_instances: Dict[str, Any] = {}


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


if OLLAMA_INFRA:
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
            print(
                f"\n[LLM Backend] Ollama is NOT responding at {OLLAMA_BASE_URL}. Cannot proceed.\n")
            sys.exit(1)
        else:
            print("[LLM Backend] Ollama reachable ✓ (Fallback)")
    else:
        _llamacpp_models = _scan_gguf()
        if not _llamacpp_models:
            print(
                "\n[LLM Backend] No matching .gguf files found. Switching to Ollama HTTP mode.\n")
            OLLAMA_INFRA = True
            if not _check_ollama():
                sys.exit(1)
        else:
            print(
                f"[LLM Backend] llama.cpp ready ✓  ·  models: {list(_llamacpp_models.keys())}")


def is_ollama() -> bool: return OLLAMA_INFRA


def available_models() -> List[str]:
    return list(MODEL_MAP.keys()) if OLLAMA_INFRA else list(_llamacpp_models.keys())


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


def chat(model: str, messages: List[Dict[str, Any]], tools: Optional[List[Any]] = None, think: bool = False, stream: bool = True) -> Generator[Dict[str, Any], None, None]:
    if OLLAMA_INFRA:
        yield from _chat_ollama(model, messages, tools, think, stream)
    else:
        yield from _chat_llamacpp(model, messages, tools, think, stream)


def _chat_ollama(model, messages, tools, think, stream):
    resolved = MODEL_MAP.get(model, {}).get("ollama", model)
    openai_tools = _tools_to_openai(tools)

    payload = {
        "model": resolved,
        "messages": messages,
        "stream": True
    }
    if openai_tools:
        payload["tools"] = openai_tools
    if think:
        payload["think"] = think

    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        resp = urllib.request.urlopen(req, timeout=None)
    except urllib.error.HTTPError as e:
        # Read the error body for Ollama's detailed message
        error_body = e.read().decode("utf-8", errors="replace")
        raise urllib.error.HTTPError(
            e.url, e.code,
            f"{e.reason} | Model: {resolved} | think: {think} | tools: {bool(openai_tools)} | Body: {error_body}",
            e.headers, None
        )

    with resp:
        tool_calls_buf = {}
        for line in resp:
            # ... rest unchanged ...
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = chunk.get("message", {})
            content = msg.get("content")
            thinking = msg.get("thinking")
            tcs = msg.get("tool_calls")

            if content:
                yield {"message": {"role": "assistant", "content": content, "thinking": None, "tool_calls": None}}
            if thinking:
                yield {"message": {"role": "assistant", "content": None, "thinking": thinking, "tool_calls": None}}

            if tcs:
                for i, tc in enumerate(tcs):
                    args = tc.get("function", {}).get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except:
                            pass
                    tc["function"]["arguments"] = args
                    tool_calls_buf[i] = tc

            if chunk.get("done"):
                break

    if tool_calls_buf:
        final_tcs = [tool_calls_buf[i] for i in sorted(tool_calls_buf.keys())]
        yield {"message": {"role": "assistant", "content": None, "thinking": None, "tool_calls": final_tcs}}


def _chat_llamacpp(model, messages, tools, think, stream):
    instance = _get_llama(model)
    openai_tools = _tools_to_openai(tools)
    kwargs: Dict[str, Any] = {
        "messages": messages, "stream": True, "temperature": 0.6, "top_p": 0.9,
    }
    if openai_tools:
        kwargs["tools"] = openai_tools
        kwargs["tool_choice"] = "auto"

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
        resolved = MODEL_MAP.get(model, {}).get("ollama", model)
        openai_tools = _tools_to_openai(tools)
        payload = {
            "model": resolved,
            "messages": messages,
            "stream": False
        }
        if openai_tools:
            payload["tools"] = openai_tools
        if think:
            payload["think"] = think

        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                m = json.loads(resp.read())
                msg = m.get("message", m)
                tcs = msg.get("tool_calls")
                if tcs:
                    for tc in tcs:
                        args = tc.get("function", {}).get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except:
                                pass
                        tc["function"]["arguments"] = args
                return Message(
                    role=msg.get("role", "assistant"),
                    content=msg.get("content"),
                    tool_calls=tcs,
                    thinking=msg.get("thinking")
                )
        except Exception as e:
            return Message(role="assistant", content=f"Error: {str(e)}")
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


def get_embedding(text: str, model: str = None) -> List[float]:
    """Fetches embedding vector for a given text using standard urllib."""
    if not text.strip():
        return []
    # Use dedicated embedding model, not the chat model
    emb_model = model if model and model != "ornith" else EMBEDDING_MODEL
    if OLLAMA_INFRA:
        try:
            resolved = MODEL_MAP.get(emb_model, {}).get("ollama", emb_model)
            payload = {"model": resolved, "prompt": text}
            req = urllib.request.Request(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                return data.get("embedding", [])
        except Exception as e:
            print(f"[Embedding Error]: {e}")
            return []
    else:
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