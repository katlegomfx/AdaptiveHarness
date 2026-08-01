# src/safety_net.py
import os
import sys
import shutil
import hashlib
import time
import json
from src.config import BACKUP_DIR, BASE_DIR

STACK_FILE = os.path.join(BACKUP_DIR, "stack.json")

SRC_FILES = [
    "start.py", "requirements.txt", ".env.example",
    "src/config.py", "src/safety_net.py",
    "src/agent/agent.py", "src/agent/planner.py", "src/agent/parser.py", "src/agent/prompts.py", "src/agent/autonomous.py",
    "src/core/result.py", "src/core/types.py", "src/core/logger.py", "src/core/metrics.py",
    "src/llm_backend/base.py", "src/llm_backend/ollama.py", "src/llm_backend/llamacpp.py", "src/llm_backend/embeddings.py",
    "src/tools/registry.py", "src/tools/builtin.py", "src/tools/dynamic.py", "src/tools/sandbox.py",
    "src/memory/storage.py", "src/memory/condenser.py", "src/memory/reflection.py",
    "src/ui/cli.py", "src/ui/tui.py", "src/ui/async_input.py"
]


def _load_stack():
    if os.path.exists(STACK_FILE):
        with open(STACK_FILE, "r") as f:
            return json.load(f)
    return []


def _save_stack(stack):
    with open(STACK_FILE, "w") as f:
        json.dump(stack, f)


def snapshot(description: str) -> bool:
    os.makedirs(BACKUP_DIR, exist_ok=True)
    backup_id = hashlib.sha1(
        f"{description}{time.time()}".encode()).hexdigest()[:10]
    backup_path = os.path.join(BACKUP_DIR, backup_id)
    os.makedirs(backup_path, exist_ok=True)
    for f in SRC_FILES:
        abs_f = os.path.join(BASE_DIR, f)
        if os.path.exists(abs_f):
            dest = os.path.join(backup_path, f)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            try:
                shutil.copy2(abs_f, dest)
            except Exception:
                pass
    with open(os.path.join(backup_path, "desc.txt"), "w", encoding="utf-8") as f:
        f.write(description)
    stack = _load_stack()
    stack.append(backup_id)
    _save_stack(stack)
    return True


def revert() -> str:
    stack = _load_stack()
    if not stack:
        return ""
    backup_id = stack.pop()
    _save_stack(stack)
    backup_path = os.path.join(BACKUP_DIR, backup_id)
    if not os.path.exists(backup_path):
        return ""
    desc = ""
    desc_file = os.path.join(backup_path, "desc.txt")
    if os.path.exists(desc_file):
        with open(desc_file, "r", encoding="utf-8") as f:
            desc = f.read()
    for f in SRC_FILES:
        abs_f = os.path.join(backup_path, f)
        if os.path.exists(abs_f):
            dest = os.path.join(BASE_DIR, f)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(abs_f, dest)
    shutil.rmtree(backup_path)
    return f"Reverted: {desc}"


def validate_python_syntax(filepath: str) -> tuple[bool, str]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            compile(f.read(), filepath, 'exec')
        return True, ""
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"
