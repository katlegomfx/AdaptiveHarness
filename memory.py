# 3.5 Token-Based Memory Condensation
from llm_backend import chat


def condense_history(
    messages: list,
    model_name: str = "ornith",
    max_window: int = 10,
    token_threshold: int = 6000,
    log_stream_func=None,
) -> list:
    """Summarizes older conversation turns while streaming visible output to stdout/log."""
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = None

    def count_tokens(text):
        if enc:
            return len(enc.encode(text))
        return len(text) // 4

    total_tokens = sum(count_tokens(str(m.get("content", "")))
                       for m in messages)
    if total_tokens < token_threshold and len(messages) <= max_window * 2:
        return messages

    system_msg = messages[0] if messages[0].get("role") == "system" else None
    start_idx = 1 if system_msg else 0

    recent_messages = messages[-max_window:]
    older_messages = messages[start_idx:-max_window]

    if not older_messages:
        return messages

    summary_prompt = [
        {
            "role": "system",
            "content": "Summarize the key facts, findings, decisions, and tool outputs from this conversation snippet concisely.",
        },
        {
            "role": "user",
            "content": f"Conversation history snippet:\n{older_messages}",
        },
    ]

    header_msg = "\n-> [Memory Condenser] Summarizing long-term conversation context..."
    if log_stream_func:
        log_stream_func(header_msg + "\n")
    else:
        print(header_msg)

    summary_content = ""
    try:
        stream = chat(model=model_name, messages=summary_prompt, stream=True)
        for chunk in stream:
            msg = (
                chunk.get("message")
                if isinstance(chunk, dict)
                else getattr(chunk, "message", None)
            )
            if not msg:
                continue
            content = getattr(msg, "content", None) or (
                msg.get("content") if isinstance(msg, dict) else None
            )
            if content:
                summary_content += content
                if log_stream_func:
                    log_stream_func(content)
                else:
                    print(content, end="", flush=True)

        done_msg = "\n-> [Memory Condenser] Context compression complete.\n"
        if log_stream_func:
            log_stream_func(done_msg)
        else:
            print(done_msg)

        condensed_summary = {
            "role": "system",
            "content": f"[Progress Context Summary]: {summary_content.strip()}",
        }

        new_history = []
        if system_msg:
            new_history.append(system_msg)
        new_history.append(condensed_summary)
        new_history.extend(recent_messages)
        return new_history

    except Exception as e:
        err_msg = f"\n-> [Memory Condenser Fallback] Summarization skipped: {e}\n"
        if log_stream_func:
            log_stream_func(err_msg)
        else:
            print(err_msg)
        return [system_msg] + recent_messages if system_msg else recent_messages
