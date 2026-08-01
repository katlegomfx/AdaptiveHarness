# src/memory/reflection.py
import json
import re
from src.llm_backend import chat, get_embedding
from src.memory.storage import save_learning, save_aspect_memory
from src.agent.context import build_aspect_context  # <-- ADD IMPORT
from src.config import IMPROVEMENT_GUIDE_PATH, META_PROMPT_MODEL, META_PROMPT_TEMP


def reflect_on_task(agent, user_prompt: str, messages: list, plan: str = ""):
    agent.emit("\n-> [Post-Mortem] Reflecting on task execution...\n")

    # USE ASPECT CONTEXT to remember past reflections
    context_messages = build_aspect_context(
        agent, "reflector", user_prompt, recent_n=2)

    reflect_messages = [
        {"role": "system",
            "content": "You are a reflection agent. Analyze the conversation and the original plan. Did the tools work? Were any redundant? What should be remembered for future tasks? Output a JSON with 'learnings' (list of strings) and 'improvements' (string). CRITICAL: OUTPUT ONLY VALID JSON. Do not output <think> tags or markdown."},
        *context_messages,
        {"role": "user",
            "content": f"Original Prompt: {user_prompt}\n\nOriginal Plan:\n{plan}\n\nConversation:\n{json.dumps(messages[-10:], default=str)}"}
    ]
    try:
        stream = chat(model=META_PROMPT_MODEL,
                      messages=reflect_messages, stream=True, temperature=META_PROMPT_TEMP)
        content_buffer = agent._print_stream_and_get_content(
            stream, header="[Reflection]\n", end="\n")

        if not content_buffer.strip():
            agent.emit(
                "   [Reflection Skipped]: Model returned an empty response.\n")
            return

        data = _extract_json_from_text(content_buffer)
        if not data:
            agent.emit(
                "   [Reflection Failed]: Could not extract valid JSON from model output.\n")
            return

        for learning in data.get("learnings", []):
            emb = get_embedding(learning)
            save_learning(learning, embedding=emb)

        improvements = data.get("improvements", "")
        if improvements:
            with open(IMPROVEMENT_GUIDE_PATH, "a", encoding="utf-8") as f:
                f.write(f"\n## Task: {user_prompt}\n{improvements}\n")

        # SAVE THIS REFLECTION TO ASPECT MEMORY
        reflection_summary = f"Task: {user_prompt}\nLearnings: {json.dumps(data.get('learnings', []))}"
        save_aspect_memory("reflector", agent.session_id, reflection_summary,
                           embedding=get_embedding(reflection_summary))

        agent.emit("   [Reflection Complete]: Learnings saved.\n")
    except Exception as e:
        agent.emit(f"   [Reflection Failed]: {e}\n")

def _extract_json_from_text(text: str) -> dict | None:
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        text = match.group(1)
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        json_str = text[start:end+1]
        try:
            sanitized_str = re.sub(r'(?<!\\)\n', r'\\n', json_str)
            sanitized_str = re.sub(r'(?<!\\)\t', r'\\t', sanitized_str)
            return json.loads(sanitized_str)
        except json.JSONDecodeError:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                return None
    return None
