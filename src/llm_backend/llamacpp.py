import os
import sys
import threading
import time
import json
from typing import Any, Dict, List, Optional
from src.llm_backend.base import MODELS_DIR, MODEL_MAP, LLAMACPP_N_CTX, LLAMACPP_N_GPU_LAYERS, LLAMACPP_N_THREADS, _tools_to_openai

_llamacpp_lock = threading.Lock()
_llamacpp_models: Dict[str, str] = {}
_llamacpp_instances: Dict[str, Any] = {}


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


def _get_llama(model_name: str):
    if model_name in _llamacpp_instances:
        return _llamacpp_instances[model_name]
    path = _llamacpp_models.get(model_name)
    if not path:
        raise ValueError(f"Model '{model_name}' not found.")
    import llama_cpp
    print(f"[LLM Backend] Loading '{model_name}'...", end=" ", flush=True)
    t0 = time.time()
    kwargs: Dict[str, Any] = {"model_path": path, "n_ctx": LLAMACPP_N_CTX,
                              "n_gpu_layers": LLAMACPP_N_GPU_LAYERS, "verbose": False}
    if LLAMACPP_N_THREADS > 0:
        kwargs["n_threads"] = LLAMACPP_N_THREADS
    instance = llama_cpp.Llama(**kwargs)
    print(f"done ({time.time() - t0:.1f}s)")
    _llamacpp_instances[model_name] = instance
    return instance


def _chat_llamacpp(model, messages, tools, think, stream, temperature):
    instance = _get_llama(model)
    openai_tools = _tools_to_openai(tools)
    # Use the passed temperature instead of hardcoded 0.5
    kwargs: Dict[str, Any] = {"messages": messages,
                              "stream": True, "temperature": temperature, "top_p": 0.2,
                              "repeat_penalty": 1.15}
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
            if isinstance(raw_args, str):
                try:
                    parsed = json.loads(raw_args) if raw_args.strip() else {}
                except json.JSONDecodeError:
                    parsed = {"_raw": raw_args}
            else:
                parsed = raw_args if isinstance(raw_args, dict) else {}
            final.append({"id": entry["id"], "type": "function", "function": {
                         "name": entry["function"]["name"], "arguments": parsed}})
        yield {"message": {"role": "assistant", "content": None, "thinking": None, "tool_calls": final}}
