# src/ui/cli.py
import os
import sys
import time
import json
import threading
import queue

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text

from src.agent.autonomous import extract_goals_from_conversation
from src.agent.planner import generate_task_profile
from src.memory.reflection import reflect_on_task
from src.memory.storage import add_long_term_goal, mark_goal_completed, retrieve_learnings, list_long_term_goals
from src.llm_backend import get_embedding
from src.config import PATCH_DIR, AUTONOMOUS_COOLDOWN_SECONDS, MAX_AUTONOMOUS_CYCLES_IN_A_ROW
from src.ui.commands import handle_special_commands

# Initialize Rich Console
console = Console()


def rich_stream_handler(text: str):
    """Parses agent output and formats it with Rich."""
    # If it's just whitespace/newlines, print it raw to maintain spacing
    if not text.strip():
        if "\n" in text:
            console.print(text, end="")
        return

    # 1. Tool Results -> Blue Panel
    if text.startswith("   Result ["):
        console.print(Panel(text.strip(), title="Tool Result",
                      border_style="blue", expand=False))
    # 2. Execution Logs -> Green
    elif any(kw in text for kw in ["-> [In-Process Execution]", "-> [Sandbox Execution]", "Model requested tool"]):
        console.print(text, style="bold green", end="")
    # 3. Planning/Pass 0/1 -> Cyan
    elif any(kw in text for kw in ["-> [Pass", "[Task Profile]", "[Plan]", "[Meta-Prompt Directives]", "Pass 1 Complete"]):
        console.print(text, style="bold cyan", end="")
    # 4. Safety & Errors -> Red
    elif any(kw in text for kw in ["[Safety]", "Error", "Failed", "Runtime Failure", "Reverted", "CRITICAL ERROR"]):
        console.print(text, style="bold red", end="")
    # 5. Warnings/Fallback -> Yellow
    elif any(kw in text for kw in ["[Fallback Parser]", "[Warning]", "Skipped", "Defaulting"]):
        console.print(text, style="bold yellow", end="")
    # 6. Memory/Reflection -> Magenta
    elif any(kw in text for kw in ["[Reflection]", "[Goal Extractor]", "[Memory Condenser", "[Summary]", "Learnings"]):
        console.print(text, style="magenta", end="")
    # 7. Special Command Outputs -> White/Bold
    elif text.startswith("===") or text.startswith("   [") or text.startswith("  -") or text.startswith("   Usage"):
        console.print(text, style="bold white", end="")
    # 8. Final Response (No tags, just raw text from LLM) -> Plain White
    else:
        # FIX: Do not use Markdown() on streaming chunks.
        # It treats each chunk as a separate block and breaks them onto new lines.
        # Just print the text directly to maintain a smooth stream.
        console.print(text, style="white", end="")


def run_blocking_loop(agent, use_tui: bool = False) -> None:
    if use_tui:
        from src.ui.tui import CursesTUI, StdoutRedirector
        tui = CursesTUI()
        agent.on_stream = tui.stream_handler
        agent.on_input = tui.input_handler
        sys.stdout = StdoutRedirector(tui)
    else:
        agent.on_stream = rich_stream_handler

    user_input_queue = queue.Queue()

    def agent_worker():
        while True:
            user_input = user_input_queue.get()
            if user_input == "__EXIT__":
                break
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break
            if handle_special_commands(agent, user_input):
                continue

            query_emb = get_embedding(user_input)
            plan, directives, tools_info = generate_task_profile(
                agent=agent, user_prompt=user_input, query_emb=query_emb)
            learnings = retrieve_learnings(user_input, query_emb=query_emb)
            agent.run_agent_loop(user_input, plan, directives,
                                 tools_info, learnings, query_emb=query_emb)

            def background_post_mortem():
                try:
                    reflect_on_task(agent, user_input, agent.history, plan)
                    new_goals = extract_goals_from_conversation(
                        user_input, agent.history, agent=agent)
                    for g in new_goals:
                        gid = add_long_term_goal(
                            g, priority=5, source="reflection", embedding=get_embedding(g))
                        console.print(
                            f"   [Goal Captured from conversation] #{gid}: {g}", style="magenta")
                except Exception as e:
                    console.print(
                        f"   [Background Task Error]: {e}", style="bold red")

            threading.Thread(target=background_post_mortem,
                             daemon=True).start()

    worker = threading.Thread(target=agent_worker, daemon=True)
    worker.start()

    try:
        if use_tui:
            while worker.is_alive():
                user_input = tui.get_input_non_blocking()
                if user_input is not None:
                    if user_input.lower() in ("exit", "quit"):
                        user_input_queue.put("__EXIT__")
                        break
                    user_input_queue.put(user_input)
                time.sleep(0.01)
        else:
            while worker.is_alive():
                user_input = console.input("[bold cyan]User > [/bold cyan]")
                if user_input.lower() in ("exit", "quit"):
                    user_input_queue.put("__EXIT__")
                    break
                user_input_queue.put(user_input)
    finally:
        if use_tui:
            sys.stdout = sys.__stdout__
            tui.cleanup()


def run_scheduled_loop(agent, idle_timeout: int, allow_self_edit: bool, use_tui: bool = False) -> None:
    from src.ui.async_input import AsyncInputReader
    if use_tui:
        from src.ui.tui import CursesTUI, StdoutRedirector
        tui = CursesTUI()
        agent.on_stream = tui.stream_handler
        agent.on_input = tui.input_handler
        sys.stdout = StdoutRedirector(tui)
    else:
        agent.on_stream = rich_stream_handler

    if use_tui:
        user_input_queue = queue.Queue()

        def agent_worker():
            consecutive_autonomous = 0
            while True:
                try:
                    try:
                        user_input = user_input_queue.get(timeout=idle_timeout)
                        if user_input == "__EXIT__":
                            break
                    except queue.Empty:
                        user_input = None

                    if user_input is None:
                        consecutive_autonomous += 1
                        if consecutive_autonomous > MAX_AUTONOMOUS_CYCLES_IN_A_ROW:
                            agent.emit(
                                f"\n[Safety] Hit {MAX_AUTONOMOUS_CYCLES_IN_A_ROW} consecutive autonomous cycles. Pausing for user.\n")
                            consecutive_autonomous = 0
                            user_input = user_input_queue.get()
                            if user_input == "__EXIT__":
                                break
                            user_input_queue.put(user_input)
                            continue

                        agent.run_autonomous_cycle(
                            allow_self_edit=allow_self_edit)
                        time.sleep(AUTONOMOUS_COOLDOWN_SECONDS)
                        continue

                    consecutive_autonomous = 0
                    if not user_input:
                        continue
                    if user_input.lower() in ("exit", "quit"):
                        break

                    if handle_special_commands(agent, user_input):
                        continue

                    query_emb = get_embedding(user_input)
                    plan, directives, tools_info = generate_task_profile(
                        agent=agent, user_prompt=user_input, query_emb=query_emb)
                    learnings = retrieve_learnings(
                        user_input, query_emb=query_emb)
                    agent.run_agent_loop(
                        user_input, plan, directives, tools_info, learnings, query_emb=query_emb)

                    def background_post_mortem():
                        try:
                            reflect_on_task(agent, user_input,
                                            agent.history, plan)
                            new_goals = extract_goals_from_conversation(
                                user_input, agent.history, agent=agent)
                            for g in new_goals:
                                gid = add_long_term_goal(
                                    g, priority=5, source="reflection", embedding=get_embedding(g))
                                agent.emit(
                                    f"   [Goal Captured from conversation] #{gid}: {g}\n")
                        except Exception as e:
                            agent.emit(f"   [Background Task Error]: {e}\n")

                    threading.Thread(
                        target=background_post_mortem, daemon=True).start()

                except Exception as e:
                    agent.emit(f"\n[Worker Error] {e}\n")

        worker = threading.Thread(target=agent_worker, daemon=True)
        worker.start()

        try:
            while worker.is_alive():
                user_input = tui.get_input_non_blocking()
                if user_input is not None:
                    if user_input.lower() in ("exit", "quit"):
                        user_input_queue.put("__EXIT__")
                        break
                    user_input_queue.put(user_input)
                time.sleep(0.01)
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            sys.stdout = sys.__stdout__
            tui.cleanup()

    else:
        reader = AsyncInputReader()
        consecutive_autonomous = 0
        console.print(
            f"Autonomous Mode: ENABLED  (idle timeout: {idle_timeout}s, self-edit: {'ON' if allow_self_edit else 'OFF'})", style="bold yellow")
        pending = list_long_term_goals(status="pending")
        if pending:
            console.print(
                f"Pending long-term goals: {len(pending)}", style="yellow")
            for g in pending[:3]:
                console.print(
                    f"  [#{g['id']}] {g['goal_text']}", style="yellow")

        while True:
            try:
                console.print("User > ", style="bold cyan", end="")
                sys.stdout.flush()
                user_input = reader.get_input(timeout=idle_timeout)

                if user_input is None:
                    consecutive_autonomous += 1
                    if consecutive_autonomous > MAX_AUTONOMOUS_CYCLES_IN_A_ROW:
                        console.print(
                            f"\n[Safety] Hit {MAX_AUTONOMOUS_CYCLES_IN_A_ROW} consecutive autonomous cycles. Pausing for user.", style="bold red")
                        consecutive_autonomous = 0
                        continue
                    agent.run_autonomous_cycle(allow_self_edit=allow_self_edit)
                    time.sleep(AUTONOMOUS_COOLDOWN_SECONDS)
                    continue

                consecutive_autonomous = 0
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    break
                if handle_special_commands(agent, user_input):
                    continue

                query_emb = get_embedding(user_input)
                plan, directives, tools_info = generate_task_profile(
                    agent=agent, user_prompt=user_input, query_emb=query_emb)
                learnings = retrieve_learnings(user_input, query_emb=query_emb)
                agent.run_agent_loop(
                    user_input, plan, directives, tools_info, learnings, query_emb=query_emb)

                def background_post_mortem():
                    try:
                        reflect_on_task(agent, user_input, agent.history, plan)
                        new_goals = extract_goals_from_conversation(
                            user_input, agent.history, agent=agent)
                        for g in new_goals:
                            gid = add_long_term_goal(
                                g, priority=5, source="reflection", embedding=get_embedding(g))
                            console.print(
                                f"   [Goal Captured from conversation] #{gid}: {g}", style="magenta")
                    except Exception as e:
                        console.print(
                            f"   [Background Task Error]: {e}", style="bold red")

                threading.Thread(target=background_post_mortem,
                                 daemon=True).start()

            except (KeyboardInterrupt, SystemExit):
                console.print("\nExiting...", style="bold red")
                sys.exit(0)
