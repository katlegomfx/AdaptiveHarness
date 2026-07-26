# goals.py
import json
import os
import re
from typing import Dict, List, Optional

from llm_backend import chat, get_embedding, cosine_similarity
from observability.logger import logger
from storage import (
    add_long_term_goal, get_next_long_term_goal, list_long_term_goals,
    log_system_improvement, mark_goal_decomposed, update_goal_embedding
)

GOAL_MODEL = "ornith"

INTROSPECTABLE_FILES = [
    "main.py", "dynamic_tools.py", "memory.py", "llm_backend.py",
    "sandbox.py", "storage.py", "safety_net.py",
    "runtime/result.py", "observability/logger.py", "observability/metrics.py",
]


def extract_goals_from_conversation(user_prompt: str, conversation: list) -> List[str]:
    messages = [
        {
            "role": "system",
            "content": (
                "You extract LONG-TERM goals from agent conversations. "
                "A long-term goal is something the user wants accomplished eventually but "
                "not necessarily right now (e.g. 'we should eventually support X', "
                "'I want the agent to be able to Y'). "
                "Return ONLY a raw JSON array of self-contained goal strings. "
                "Return [] if none."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User prompt: {user_prompt}\n\n"
                f"Conversation tail:\n{json.dumps(conversation[-6:], default=str)}"
            ),
        },
    ]
    try:
        buf = ""
        for chunk in chat(model=GOAL_MODEL, messages=messages, stream=True):
            msg = chunk.get("message") if isinstance(
                chunk, dict) else getattr(chunk, "message", None)
            if not msg:
                continue
            c = getattr(msg, "content", None) or (
                msg.get("content") if isinstance(msg, dict) else None)
            if c:
                buf += c
        cleaned = buf.strip().replace("```json", "").replace("```", "").strip()
        goals = json.loads(cleaned)
        if isinstance(goals, list):
            return [g.strip() for g in goals if isinstance(g, str) and g.strip()]
    except Exception as e:
        logger.warning(f"Goal extraction failed: {e}")
    return []


def decompose_goal(goal_text: str) -> List[str]:
    """Ask the LLM to split a large goal into 2-4 smaller, actionable sub-tasks."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a task decomposition agent. Break down the provided large goal into 2-4 smaller, "
                "actionable sub-tasks that can be executed sequentially. "
                "Return ONLY a raw JSON array of strings."
            ),
        },
        {"role": "user", "content": f"Large Goal: {goal_text}"},
    ]
    try:
        buf = ""
        for chunk in chat(model=GOAL_MODEL, messages=messages, stream=True):
            msg = chunk.get("message") if isinstance(
                chunk, dict) else getattr(chunk, "message", None)
            if not msg:
                continue
            c = getattr(msg, "content", None) or (
                msg.get("content") if isinstance(msg, dict) else None)
            if c:
                buf += c
        cleaned = buf.strip().replace("```json", "").replace("```", "").strip()
        sub_tasks = json.loads(cleaned)
        if isinstance(sub_tasks, list) and sub_tasks:
            return [s.strip() for s in sub_tasks if isinstance(s, str) and s.strip()]
    except Exception as e:
        logger.warning(f"Goal decomposition failed: {e}")
    return []


def find_system_improvement_opportunity() -> Optional[Dict]:
    file_contents: Dict[str, str] = {}
    for f in INTROSPECTABLE_FILES:
        if os.path.exists(f):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    file_contents[f] = fp.read()
            except Exception:
                pass

    markers: List[str] = []
    for f, content in file_contents.items():
        for i, line in enumerate(content.splitlines(), 1):
            if re.search(r"\b(TODO|FIXME|HACK|XXX|BUG)\b", line, re.IGNORECASE):
                markers.append(f"{f}:{i} -> {line.strip()}")
    marker_block = "\n".join(
        markers[:25]) if markers else "(No explicit TODO/FIXME markers found.)"

    src_dump = json.dumps(file_contents, indent=2)[:8000]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a self-improvement analyzer for an autonomous Python agent. "
                "Identify ONE small, concrete, safe improvement to make to the codebase. "
                "Look for: bugs, error-handling gaps, missing features, robustness issues, "
                "TODO/FIXME markers, performance problems, or code smells. "
                "Return a JSON object with keys: "
                "'file' (str), 'issue' (str), 'suggested_action' (str), 'priority' (int 1-10). "
                "Do NOT propose large rewrites. Pick the smallest valuable win."
            ),
        },
        {"role": "user",
            "content": f"Source files (truncated):\n{src_dump}\n\nPre-identified markers:\n{marker_block}"},
    ]
    try:
        buf = ""
        for chunk in chat(model=GOAL_MODEL, messages=messages, stream=True):
            msg = chunk.get("message") if isinstance(
                chunk, dict) else getattr(chunk, "message", None)
            if not msg:
                continue
            c = getattr(msg, "content", None) or (
                msg.get("content") if isinstance(msg, dict) else None)
            if c:
                buf += c
        cleaned = buf.strip().replace("```json", "").replace("```", "").strip()
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start:end + 1])
    except Exception as e:
        logger.warning(f"Self-improvement scan failed: {e}")
    return None


def select_autonomous_task(allow_self_edit: bool = False, current_state_emb: list = None) -> Dict:
    """Return one of:
         {"type": "long_term_goal",   "goal": {...}}
         {"type": "system_improvement","improvement": {...}}
         {"type": "none"}
    """
    goals = list_long_term_goals(status="pending")
    if not goals:
        if allow_self_edit:
            imp = find_system_improvement_opportunity()
            if imp:
                return {"type": "system_improvement", "improvement": imp}
        return {"type": "none"}

    # 1. Embedding-based Retrieval (if state context provided)
    selected_goal = None
    if current_state_emb:
        best_score = -1.0
        for g in goals:
            if not g.get("embedding"):
                g["embedding"] = get_embedding(g["goal_text"])
                update_goal_embedding(g["id"], g["embedding"])

            score = cosine_similarity(current_state_emb, g["embedding"])
            if score > best_score:
                best_score = score
                selected_goal = g
    else:
        # Fallback to priority/oldest if no embeddings
        selected_goal = goals[0]

    if selected_goal:
        # 2. Goal Decomposition
        # If it's a high-level goal and hasn't been decomposed, break it down.
        # (We can treat anything with priority <= 5 as needing decomposition if complex,
        # but let's just try to decompose any goal that doesn't have children yet)
        has_children = any(g.get("parent_id") ==
                           selected_goal["id"] for g in goals)
        if not has_children and not selected_goal.get("decomposed"):
            sub_tasks = decompose_goal(selected_goal["goal_text"])
            if sub_tasks and len(sub_tasks) > 1:
                for st in sub_tasks:
                    st_emb = get_embedding(st)
                    add_long_term_goal(
                        st, priority=selected_goal["priority"] - 1, source="decomposition", parent_id=selected_goal["id"], embedding=st_emb)
                mark_goal_decomposed(selected_goal["id"])
                # Fetch the newly added highest priority child
                return {"type": "long_term_goal", "goal": get_next_long_term_goal()}

        return {"type": "long_term_goal", "goal": selected_goal}

    return {"type": "none"}
