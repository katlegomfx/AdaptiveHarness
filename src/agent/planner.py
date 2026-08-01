# src/agent/planner.py
import inspect
from src.tools.registry import TOOL_REGISTRY, REGISTRY_LOCK
from src.llm_backend import chat, get_embedding
from src.config import META_PROMPT_MODEL
from src.agent.context import build_aspect_context
from src.memory.storage import save_aspect_memory


def generate_plan(agent, user_prompt: str) -> str:
    agent.emit("-> [Pass 0] Synthesizing execution plan...")
    with REGISTRY_LOCK:
        active_tools = list(TOOL_REGISTRY.values())
    tools_info = "\n".join(
        [f"- {t.__name__}: {inspect.getdoc(t) or 'No description'}" for t in active_tools])

    context_messages = build_aspect_context(
        agent, "planner", user_prompt, recent_n=4)

    plan_messages = [
        {"role": "system", "content": f"You are an expert planner. Create a concise step-by-step plan. Identify what tools need to be used. CRITICAL: DO NOT execute tools, write code, or answer the user's prompt. Just list the steps.\n\nAvailable tools:\n{tools_info}"},
        *context_messages,  # Inject the recent conversation here!
        {"role": "user", "content": user_prompt}
    ]
    try:
        stream = chat(model=META_PROMPT_MODEL,
                      messages=plan_messages, stream=True)
        plan = agent._print_stream_and_get_content(
            stream, header="[Plan]\n", end="\n\n")
        if plan:
            save_aspect_memory("planner", agent.session_id,
                               plan, embedding=get_embedding(plan))
        return plan, tools_info
    except Exception as e:
        agent.emit(f"   [Planning Failed]: {e}\n")
        return "", tools_info


def generate_dynamic_system_prompt(agent, goal: str, plan: str, tools_info: str) -> str:
    agent.emit("-> [Pass 1] Synthesizing task-specific System Prompt...")

    # Get recent context
    context_messages = build_aspect_context(
        agent, "meta_prompter", goal, recent_n=2)

    meta_messages = [
        {"role": "system", "content": "You are a tactical directive generator for an AI agent. Output 3-5 concise rules."},
        *context_messages,  # Inject context here!
        {"role": "user", "content": f"Goal: {goal}\n\nExecution Plan:\n{plan}\n\nCRITICAL INSTRUCTIONS:\n- DO NOT answer the goal yourself.\n- DO NOT guess file contents or hallucinate data.\n- ONLY output 3-5 tactical directives for the main agent to follow."}
    ]
    # ... rest of the function remains the same
    try:
        stream = chat(model=META_PROMPT_MODEL,
                      messages=meta_messages, stream=True)
        dynamic_rules = agent._print_stream_and_get_content(
            stream, header="[Meta-Prompt Directives]\n", end="\n\n-> [Pass 1 Complete] Dynamic instructions generated.\n\n")
        if dynamic_rules:
            save_aspect_memory("meta_prompter", agent.session_id,
                               dynamic_rules, embedding=get_embedding(dynamic_rules))

        return dynamic_rules.strip()
    except Exception as e:
        agent.emit(
            f"\n-> [Pass 1 Fallback] Meta-prompt generation skipped: {e}\n\n")
        return "Focus on safe, correct code execution and modular tool design."
