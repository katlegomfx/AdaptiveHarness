# environmentBots

An autonomous, multi-model AI agent framework that dynamically creates, validates, registers, and executes custom Python tools on demand. It features a non-blocking scheduler, long-term goal management, self-improvement capabilities, and an optional Text User Interface (TUI).

It supports **two LLM backends** selectable via environment variables:
1. **Ollama** (Default - uses native `urllib` HTTP requests, no extra dependencies)
2. **llama.cpp** (For local CPU/GPU inference without Ollama)

## Architecture Highlights

1. **Pluggable LLM Backend & Models**: Controlled via `.env`. Easily swap between Ollama and llama.cpp, and configure specific models for reasoning, planning, testing, and summarization.
2. **Autonomous Scheduling**: Non-blocking input loop. When idle, the agent autonomously selects and works on long-term goals or system self-improvements.
3. **Dynamic Tool Creation & Persistence**: Writes Python code, inspects AST security rules, saves to disk (`custom_tools/`), and imports into runtime memory. Includes **semantic deduplication** using embeddings to prevent creating the same tool twice.
4. **Subprocess Sandbox Execution**: Tool calls execute inside isolated Python sub-processes via STDIN IPC.
5. **State Persistence & Long-Term Memory**: Conversations and learnings are saved to SQLite. Uses vector embeddings to retrieve relevant past learnings.
6. **Self-Healing & Safety Net**: If a tool crashes, the agent attempts to debug and patch it automatically. Core file edits are tracked and can be auto-reverted on crashes.
7. **Termux / Android Compatible**: Detects Termux and gracefully handles missing optional dependencies (like `tiktoken` or `llama-cpp-python`).

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure your backend and models (.env)
# By default, it uses Ollama. To use llama.cpp instead:
#    Change OLLAMA_INFRA=False in .env
#    Place your .gguf models in the ./llms/ directory

# 3. Run new agent session (Default: scheduled mode, 2-min idle timeout)
python start.py

# 4. Run with the Text User Interface (curses)
python start.py --tui

# 5. Run with auto approvals (Skip Human-In-The-Loop)
python start.py --yolo

# List active saved sessions
python start.py --list-sessions

# Resume a specific session
python start.py --session <SESSION_ID>

# Tighter idle timeout (30s), allow autonomous source-code edits
python start.py --idle-timeout 30 --autonomous-self-edit

# Legacy blocking behavior (no scheduler)
python start.py --no-autonomous

```

## Example requests:
What files and folders are in the current directory?

## Suggestion: