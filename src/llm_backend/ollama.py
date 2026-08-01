import json
import urllib.request
import urllib.error
from src.llm_backend.base import OLLAMA_BASE_URL, MODEL_MAP, _tools_to_openai


def _check_ollama() -> bool:
    try:
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/tags", headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def _chat_ollama(model, messages, tools, think, stream, temperature):
    resolved = MODEL_MAP.get(model, {}).get("ollama", model)
    openai_tools = _tools_to_openai(tools)
    # Add options temperature
    payload = {"model": resolved, "messages": messages,
               "stream": True, "options": {"temperature": temperature}}
    if openai_tools:
        payload["tools"] = openai_tools
    if think:
        payload["think"] = think

    req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/chat", data=json.dumps(
        payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=None)
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise urllib.error.HTTPError(
            e.url, e.code, f"{e.reason} | Model: {resolved} | Body: {error_body}", e.headers, None)

    with resp:
        tool_calls_buf = {}
        for line in resp:
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
