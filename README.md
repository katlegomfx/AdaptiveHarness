# environmentBots

An autonomous, multi-model AI agent framework that dynamically creates, validates, registers, and executes custom Python tools on demand. 

It supports **two LLM backends** selectable via environment variables:
1. **Ollama** (Default)
2. **llama.cpp** (For local CPU/GPU inference without Ollama)

## Architecture Highlights

1. **Pluggable LLM Backend**: Controlled via `.env`. Routes all model calls seamlessly through `llm_backend.py`.
2. **Meta-Prompt Synthesis (Pass 1)**: Utilizes a lightweight model to generate streaming goal-specific directives.
3. **Dynamic Tool Creation & Persistence**: Writes Python code, inspects AST security rules, saves to disk (`custom_tools/`), and imports into runtime memory.
4. **Subprocess Sandbox Execution**: Tool calls execute inside isolated Python sub-processes.
5. **State Persistence**: Conversations saved to SQLite for session resumption.

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure your backend (.env)
# By default, it uses Ollama. To use llama.cpp instead:
#    Change OLLAMA_INFRA=False in .env
#    Place your .gguf models in the ./models/ directory

# 3. Run new agent session
python start.py

# 4. Run new agent session
python start.py --yolo

# List active saved sessions
python start.py --list-sessions

# Resume a specific session
python start.py --session <SESSION_ID>
```

## Example requests:
What files and folders are in the current directory?

## Suggestion: