import argparse
import signal
import sys
import uuid
from dotenv import load_dotenv

from src.agent.agent import Agent
from src.tools.registry import TOOL_REGISTRY, REGISTRY_LOCK
from src.memory.storage import init_db, list_sessions
from src.llm_backend import is_ollama
from src.ui.cli import run_blocking_loop, run_scheduled_loop
from src.config import REASONING_MODEL, META_PROMPT_MODEL, TESTER_MODEL, IDLE_TIMEOUT_SECONDS

# Import tools to register them
import src.tools.builtin
import src.tools.dynamic
import src.tools.sandbox

load_dotenv()


def main():
    signal.signal(signal.SIGINT, lambda s, f: (sys.stderr.write(
        "\n[Interrupt] Force shutting down...\n"), sys.exit(0)))
    init_db()

    parser = argparse.ArgumentParser(description="Ollama Dynamic Tool Agent")
    parser.add_argument("--session", type=str, help="Session ID to resume")
    parser.add_argument("--list-sessions", action="store_true")
    parser.add_argument("--yolo", action="store_true",
                        help="Skip HITL approval")
    parser.add_argument("--no-autonomous", action="store_true",
                        help="Disable scheduler; block on input (legacy behavior)")
    parser.add_argument("--idle-timeout", type=int, default=IDLE_TIMEOUT_SECONDS,
                        help=f"Seconds to wait for input before autonomous wake-up (default: {IDLE_TIMEOUT_SECONDS})")
    parser.add_argument("--autonomous-self-edit", action="store_true",
                        help="DANGER: allow autonomous cycles to edit source files without approval. Without this flag, only long-term goals are pursued autonomously.")
    parser.add_argument("--tui", action="store_true",
                        help="Enable Text User Interface (curses)")
    args = parser.parse_args()

    if args.list_sessions:
        sessions = list_sessions()
        print("\n=== Saved Agent Sessions ===")
        if not sessions:
            print("No saved sessions found.")
        else:
            for s in sessions:
                print(
                    f"ID: {s['session_id']} | Turns: {s['latest_turn']} | Last Active: {s['timestamp']}")
        sys.exit(0)

    agent = Agent(session_id=args.session or str(
        uuid.uuid4())[:8], yolo_mode=args.yolo)

    if args.session:
        agent.load_state()
    else:
        agent.emit(
            f"=== Ollama Dynamic Agent Ready | Session: {agent.session_id} ===\n")

    agent.emit(
        f"Reasoning: '{REASONING_MODEL}' | Meta: '{META_PROMPT_MODEL}' | Tester: '{TESTER_MODEL}'\n")
    with REGISTRY_LOCK:
        agent.emit(f"Active Tools: {list(TOOL_REGISTRY.keys())}\n")
    agent.emit(f"LLM Backend: {'Ollama' if is_ollama() else 'llama.cpp'}\n")

    if args.no_autonomous:
        run_blocking_loop(agent, use_tui=args.tui)
    else:
        run_scheduled_loop(agent, idle_timeout=args.idle_timeout,
                           allow_self_edit=args.autonomous_self_edit, use_tui=args.tui)


if __name__ == "__main__":
    main()
