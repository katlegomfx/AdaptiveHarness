# src/ui/commands.py
import os
import json
import sqlite3
from src.config import DB_PATH, PATCH_DIR
from src.memory.storage import add_long_term_goal, mark_goal_completed, list_long_term_goals
from src.tools.registry import TOOL_REGISTRY, REGISTRY_LOCK


def handle_special_commands(agent, user_input: str) -> bool:
    """Checks if the user input is a special command. Returns True if handled, False otherwise."""
    lowered = user_input.lower().strip()

    if lowered in ("help", "commands"):
        agent.emit("\n=== Available Commands ===\n")
        agent.emit("  goal: <text>            - Save a long-term goal\n")
        agent.emit("  priority goal: <text>   - Save a high-priority goal\n")
        agent.emit("  goals                   - List all goals\n")
        agent.emit("  complete goal <id>      - Mark a goal as done\n")
        agent.emit("  delete goal <id>        - Delete a specific goal\n")
        agent.emit(
            "  reset goal <id>         - Reset a blocked/failed goal to pending\n")
        agent.emit("  clear goals             - Delete ALL goals\n")
        agent.emit("  list tools              - Show all registered tools\n")
        agent.emit(
            "  list patches            - Show pending source code patches\n")
        agent.emit("  approve patch <file>    - Apply a pending patch\n")
        agent.emit(
            "  clear history           - Wipe the current conversation memory\n")
        agent.emit("  exit / quit             - Shutdown the agent\n\n")
        return True

    if lowered.startswith("goal:"):
        text = user_input[5:].strip()
        if text:
            gid = add_long_term_goal(text, priority=5, source="user")
            agent.emit(f"   [Goal Saved] #{gid}: {text}\n")
        return True

    if lowered.startswith("priority goal:"):
        text = user_input[len("priority goal:"):].strip()
        if text:
            gid = add_long_term_goal(text, priority=9, source="user")
            agent.emit(f"   [Priority Goal Saved] #{gid}: {text}\n")
        return True

    if lowered in ("goals", "list goals"):
        goals = list_long_term_goals()
        agent.emit(f"\n=== Long-Term Goals ({len(goals)}) ===\n")
        for g in goals:
            agent.emit(
                f"  [#{g['id']}] [{g['status']}] P:{g['priority']} attempts:{g['attempts']} — {g['goal_text']}\n")
        agent.emit("\n")
        return True

    if lowered.startswith("complete goal"):
        try:
            gid = int(lowered.split()[-1])
            mark_goal_completed(gid, "Marked complete by user")
            agent.emit(f"   [Goal #{gid} marked complete]\n")
        except (ValueError, IndexError):
            agent.emit("   Usage: complete goal <id>\n")
        return True

    if lowered.startswith("delete goal"):
        try:
            gid = int(lowered.split()[-1])
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(
                    "DELETE FROM long_term_goals WHERE id = ?", (gid,))
                conn.commit()
            agent.emit(f"   [Goal #{gid} deleted]\n")
        except (ValueError, IndexError):
            agent.emit("   Usage: delete goal <id>\n")
        return True

    if lowered.startswith("reset goal"):
        try:
            gid = int(lowered.split()[-1])
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(
                    "UPDATE long_term_goals SET status = 'pending', attempts = 0, failure_reason = '' WHERE id = ?", (gid,))
                conn.commit()
            agent.emit(f"   [Goal #{gid} reset to pending]\n")
        except (ValueError, IndexError):
            agent.emit("   Usage: reset goal <id>\n")
        return True

    if lowered == "clear goals":
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("DELETE FROM long_term_goals")
                conn.commit()
            agent.emit("   [Success] All long-term goals have been cleared.\n")
        except Exception as e:
            agent.emit(f"   [Error clearing goals]: {e}\n")
        return True

    if lowered in ("tools", "list tools"):
        with REGISTRY_LOCK:
            tools = list(TOOL_REGISTRY.keys())
        agent.emit(f"\n=== Active Tools ({len(tools)}) ===\n")
        for t in tools:
            agent.emit(f"  - {t}\n")
        agent.emit("\n")
        return True

    if lowered == "clear history":
        agent.history = []
        agent.turn_count = 0
        agent.emit(
            "   [Success] Conversation history cleared. Starting fresh.\n")
        return True

    if lowered == "list patches":
        if os.path.exists(PATCH_DIR):
            patches = [f for f in os.listdir(PATCH_DIR) if f.endswith(".json")]
            agent.emit(f"\n=== Pending Patches ({len(patches)}) ===\n")
            for p in patches:
                agent.emit(f"  - {p}\n")
            agent.emit("Use: approve patch <filename>\n\n")
        else:
            agent.emit("No pending patches.\n")
        return True

    if lowered.startswith("approve patch"):
        try:
            patch_file = lowered.split(" ", 2)[2]
            patch_path = os.path.join(PATCH_DIR, patch_file)
            if not os.path.exists(patch_path):
                agent.emit(f"   Error: Patch file '{patch_file}' not found.\n")
                return True
            with open(patch_path, "r", encoding="utf-8") as f:
                patch_data = json.load(f)
            agent.emit(f"   [Applying Patch] {patch_file}...\n")
            agent.autonomous_mode = False
            call = {"function": {"name": "edit_source_file",
                                 "arguments": patch_data["args"]}}
            res = agent._execute_once(call, patch_data["trace_id"])
            if res.is_success:
                os.remove(patch_path)
                agent.emit(f"   [Patch Applied & Removed]: {res.value}\n")
            else:
                agent.emit(f"   [Patch Failed]: {res.value}\n")
        except Exception as e:
            agent.emit(f"   [Approve Patch Error]: {e}\n")
        return True

    return False
