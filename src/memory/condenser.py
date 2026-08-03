# memory.py
import os
from src.llm_backend import chat
from src.config import SUMMARY_TEMP


def condense_history(
    messages: list,
    model_name: str = "ornith",
    max_window: int = 10,
    token_threshold: int = 64000,
    log_stream_func=None,
) -> tuple[list, str | None]:
    """Summarizes older conversation turns. Returns (recent_history, summary_string)."""
    enc = None
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        pass

    def count_tokens(text):
        if enc:
            return len(enc.encode(text))
        return len(text) // 4

    system_msg = messages[0] if messages and messages[0].get(
        "role") == "system" else None
    start_idx = 1 if system_msg else 0

    recent_messages = messages[-max_window:]
    older_messages = messages[start_idx:-max_window]

    if not older_messages or len(older_messages) < 3:
        return messages, None

    total_tokens = sum(count_tokens(str(m.get("content", "")))
                       for m in messages)
    if total_tokens < token_threshold and len(messages) <= max_window * 2:
        return messages, None

    summary_prompt = [
        {
            "role": "system",
            "content": "Summarize the conversation snippet to preserve context for an AI agent. "
                       "CRITICAL: You MUST retain exact file paths, exact error messages, code snippets, "
                       "and specific variable names. Do not generalize technical details. "
                       "List the actions taken, the outcomes, and any unresolved issues.",
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

    think_enabled = os.environ.get(
        "THINK_ENABLED", "false").strip().lower() in ("true", "1", "yes", "on")
    summary_content = ""

    in_thinking = False
    in_content = False

    try:
        stream = chat(model=model_name, messages=summary_prompt,
                      think=think_enabled, stream=True, temperature=SUMMARY_TEMP)

        for chunk in stream:
            msg = (
                chunk.get("message")
                if isinstance(chunk, dict)
                else getattr(chunk, "message", None)
            )
            if not msg:
                continue

            chunk_thinking = getattr(msg, "thinking", None) or (
                msg.get("thinking") if isinstance(msg, dict) else None)
            content = getattr(msg, "content", None) or (
                msg.get("content") if isinstance(msg, dict) else None)

            # Stream thinking to TUI/stdout
            if chunk_thinking:
                if not in_thinking:
                    in_thinking = True
                    if log_stream_func:
                        log_stream_func("\n[Memory Condenser Thinking]\n")
                    else:
                        print("\n[Memory Condenser Thinking]\n", end="")
                if log_stream_func:
                    log_stream_func(chunk_thinking)
                else:
                    print(chunk_thinking, end="")

            # Stream summary to TUI/stdout
            if content:
                if in_thinking:
                    in_thinking = False
                    if log_stream_func:
                        log_stream_func("\n[Summary]\n")
                    else:
                        print("\n[Summary]\n", end="")
                if not in_content:
                    in_content = True
                if log_stream_func:
                    log_stream_func(content)
                else:
                    print(content, end="", flush=True)
                summary_content += content

        done_msg = "\n-> [Memory Condenser] Context compression complete.\n"
        if log_stream_func:
            log_stream_func(done_msg)
        else:
            print(done_msg)

        # Reconstruct recent history ensuring system_msg is at index 0
        new_recent = []
        if system_msg:
            new_recent.append(system_msg)
        for msg in recent_messages:
            if system_msg and msg == system_msg:
                continue
            new_recent.append(msg)

        return new_recent, summary_content.strip() or None

    except Exception as e:
        err_msg = f"\n-> [Memory Condenser Fallback] Summarization skipped: {e}\n"
        if log_stream_func:
            log_stream_func(err_msg)
        else:
            print(err_msg)
        return messages, None
