# 3.7 Make revert() Non-Destructive
import subprocess
import os
import sys
import shutil
import hashlib
import time

BACKUP_DIR = ".safety_net_backups"
_backup_stack = []


def is_git_repo() -> bool:
    return os.path.exists(".git")


def snapshot(description: str) -> bool:
    """Creates an atomic backup of the current state before an edit."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    backup_id = hashlib.sha1(
        f"{description}{time.time()}".encode()).hexdigest()[:10]
    backup_path = os.path.join(BACKUP_DIR, backup_id)
    os.makedirs(backup_path, exist_ok=True)

    if os.path.exists("custom_tools"):
        try:
            shutil.copytree("custom_tools", os.path.join(
                backup_path, "custom_tools"), dirs_exist_ok=True)
        except Exception:
            pass

    for f in ["main.py", "sandbox.py", "dynamic_tools.py", "memory.py", "llm_backend.py", "storage.py", "safety_net.py"]:
        if os.path.exists(f):
            try:
                shutil.copy2(f, os.path.join(backup_path, f))
            except Exception:
                pass

    with open(os.path.join(backup_path, "desc.txt"), "w", encoding="utf-8") as f:
        f.write(description)

    _backup_stack.append(backup_id)
    return True


def revert() -> str:
    """Reverts to the previous state and returns the description of what was removed."""
    if not _backup_stack:
        return ""
    backup_id = _backup_stack.pop()
    backup_path = os.path.join(BACKUP_DIR, backup_id)

    if not os.path.exists(backup_path):
        return ""

    desc = ""
    desc_file = os.path.join(backup_path, "desc.txt")
    if os.path.exists(desc_file):
        with open(desc_file, "r", encoding="utf-8") as f:
            desc = f.read()

    if os.path.exists(os.path.join(backup_path, "custom_tools")):
        if os.path.exists("custom_tools"):
            shutil.rmtree("custom_tools")
        shutil.copytree(os.path.join(
            backup_path, "custom_tools"), "custom_tools")

    for f in ["main.py", "sandbox.py", "dynamic_tools.py", "memory.py", "llm_backend.py", "storage.py", "safety_net.py"]:
        if os.path.exists(os.path.join(backup_path, f)):
            shutil.copy2(os.path.join(backup_path, f), f)

    shutil.rmtree(backup_path)
    return f"Reverted: {desc}"


def validate_python_syntax(filepath: str) -> tuple[bool, str]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            compile(f.read(), filepath, 'exec')
        return True, ""
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"
