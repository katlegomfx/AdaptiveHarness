import os
import sys
import time
import json
from src.agent.autonomous import extract_goals_from_conversation
from src.agent.planner import generate_plan
from src.memory.reflection import reflect_on_task
from src.memory.storage import add_long_term_goal, mark_goal_completed, retrieve_learnings, list_long_term_goals
from src.llm_backend import get_embedding
from src.config import PATCH_DIR, AUTONOMOUS_COOLDOWN_SECONDS, MAX_AUTONOMOUS_CYCLES_IN_A_ROW


def handle_special_commands(agent, user_input: str) -> bool:
    lowered = user_input.lower().strip()
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


def run_blocking_loop(agent, use_tui: bool = False) -> None:
    tui = None
    if use_tui:
        from src.ui.tui import CursesTUI, StdoutRedirector
        tui = CursesTUI()
        agent.on_stream = tui.stream_handler
        agent.on_input = tui.input_handler
        sys.stdout = StdoutRedirector(tui)
    try:
        while True:
            try:
                if tui:
                    user_input = tui.get_input_async(3600)
                    if user_input is None:
                        continue
                    user_input = user_input.strip()
                else:
                    user_input = input("User > ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    break
                if handle_special_commands(agent, user_input):
                    continue
                plan, tools_info = generate_plan(agent=agent, user_prompt=user_input)
                learnings = retrieve_learnings(
                    user_input, query_emb=get_embedding(user_input))
                agent.run_agent_loop(user_input, plan, tools_info, learnings)
                reflect_on_task(agent, user_input, agent.history)
            except (KeyboardInterrupt, SystemExit):
                agent.emit("\nExiting...\n")
                sys.exit(0)
    finally:
        if tui:
            sys.stdout = sys.__stdout__
            tui.cleanup()


def run_scheduled_loop(agent, idle_timeout: int, allow_self_edit: bool, use_tui: bool = False) -> None:
    from src.ui.async_input import AsyncInputReader
    tui = None
    if use_tui:
        from src.ui.tui import CursesTUI, StdoutRedirector
        tui = CursesTUI()
        agent.on_stream = tui.stream_handler
        agent.on_input = tui.input_handler
        sys.stdout = StdoutRedirector(tui)
    try:
        reader = None
        if not tui:
            reader = AsyncInputReader()
        consecutive_autonomous = 0
        agent.emit(
            f"Autonomous Mode: ENABLED  (idle timeout: {idle_timeout}s, self-edit: {'ON' if allow_self_edit else 'OFF'})\n")
        pending = list_long_term_goals(status="pending")
        if pending:
            agent.emit(f"Pending long-term goals: {len(pending)}\n")
            for g in pending[:3]:
                agent.emit(f"  [#{g['id']}] {g['goal_text']}\n")
        agent.emit("\n")
        while True:
            try:
                if tui:
                    user_input = tui.get_input_async(idle_timeout)
                    if user_input is not None:
                        user_input = user_input.strip()
                else:
                    agent.emit("User > ", end="")
                    sys.stdout.flush()
                    user_input = reader.get_input(timeout=idle_timeout)
                if user_input is None:
                    consecutive_autonomous += 1
                    if consecutive_autonomous > MAX_AUTONOMOUS_CYCLES_IN_A_ROW:
                        agent.emit(
                            f"\n[Safety] Hit {MAX_AUTONOMOUS_CYCLES_IN_A_ROW} consecutive autonomous cycles. Pausing for user.\n")
                        consecutive_autonomous = 0
                        continue
                    agent.run_autonomous_cycle(allow_self_edit=allow_self_edit)
                    time.sleep(AUTONOMOUS_COOLDOWN_SECONDS)
                    continue
                consecutive_autonomous = 0
                agent.emit("\n")
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    break
                if handle_special_commands(agent, user_input):
                    continue
                plan, tools_info = generate_plan(agent=agent, user_prompt=user_input)
                learnings = retrieve_learnings(
                    user_input, query_emb=get_embedding(user_input))
                agent.run_agent_loop(user_input, plan, tools_info, learnings)
                reflect_on_task(agent, user_input, agent.history)
                try:
                    new_goals = extract_goals_from_conversation(
                        user_input, agent.history, agent=agent
                    )
                    for g in new_goals:
                        gid = add_long_term_goal(
                            g, priority=5, source="reflection", embedding=get_embedding(g))
                        agent.emit(
                            f"   [Goal Captured from conversation] #{gid}: {g}\n")


                except Exception:
                    pass
            except (KeyboardInterrupt, SystemExit):
                agent.emit("\nExiting...\n")
                sys.exit(0)
    finally:
        if tui:
            sys.stdout = sys.__stdout__
            tui.cleanup()
