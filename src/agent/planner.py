# src/agent/planner.py
import inspect
import json
from src.tools.registry import TOOL_REGISTRY, REGISTRY_LOCK
from src.llm_backend import chat, get_embedding
from src.config import META_PROMPT_MODEL, META_PROMPT_TEMP
from src.agent.context import build_aspect_context
from src.memory.storage import save_aspect_memory
from src.memory.reflection import _extract_json_from_text


def generate_task_profile(agent, user_prompt: str, query_emb: list = None) -> tuple:
    """Merges planning and directive generation into one LLM call."""
    agent.emit("-> [Pass 0] Synthesizing Task Profile (Plan + Directives)...")

    with REGISTRY_LOCK:
        active_tools = list(TOOL_REGISTRY.values())
    tools_info = "\n".join(
        [f"- {t.__name__}: {inspect.getdoc(t) or 'No description'}" for t in active_tools])

    context_messages = build_aspect_context(
        agent, "planner", user_prompt, recent_n=4, query_emb=query_emb)

    messages = [
        {"role": "system",
            "content": f"You are an expert planner. Output a JSON object with 'plan' (step-by-step string) and 'directives' (3-5 concise rules string). Do not execute tools, write code, or answer the user's prompt.\n\nAvailable tools:\n{tools_info}"},
        *context_messages,
        {"role": "user", "content": user_prompt}
    ]

    try:
        stream = chat(model=META_PROMPT_MODEL, messages=messages,
                      stream=True, temperature=META_PROMPT_TEMP)
        buf = agent._print_stream_and_get_content(
            stream, header="[Task Profile]\n", end="\n\n")

        data = _extract_json_from_text(buf)
        if data:
            plan = data.get("plan", "")
            directives = data.get("directives", "")
            if plan:
                plan_emb = get_embedding(
                    plan) if query_emb is None else query_emb
                save_aspect_memory("planner", agent.session_id,
                                   plan, embedding=plan_emb)
            return plan, directives, tools_info

        # Fallback if JSON parsing fails
        agent.emit(
            "   [Task Profile Fallback]: Failed to parse JSON. Using raw text as plan.\n")
        return buf, "Focus on safe, correct code execution and modular tool design.", tools_info

    except Exception as e:
        agent.emit(f"   [Task Profile Failed]: {e}\n")
        return "", "Focus on safe, correct code execution and modular tool design.", tools_info
