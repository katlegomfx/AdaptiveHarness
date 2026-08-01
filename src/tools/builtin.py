import os
import json
import subprocess
import signal
import sys
import shutil
import time
from typing import Dict
from src.tools.registry import register_tool

PROCESS_REGISTRY_FILE = "process_registry.json"
SECRETS_FILE = ".agent_secrets.json"


def _load_secrets() -> Dict[str, str]:
    if os.path.exists(SECRETS_FILE):
        with open(SECRETS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_secrets(secrets: Dict[str, str]):
    with open(SECRETS_FILE, "w", encoding="utf-8") as f:
        json.dump(secrets, f, indent=2)


def _update_process_registry(processes: dict):
    with open(PROCESS_REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(processes, f, indent=2)


def _load_process_registry() -> dict:
    if os.path.exists(PROCESS_REGISTRY_FILE):
        with open(PROCESS_REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@register_tool
def write_file(filepath: str, content: str, **kwargs) -> str:
    dir_name = os.path.dirname(filepath)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    return f"Successfully wrote {len(content)} characters to {filepath}."


@register_tool
def read_file(filepath: str, **kwargs) -> str:
    if not os.path.exists(filepath):
        return f"Error: File '{filepath}' not found."
    if os.path.isdir(filepath):
        return f"Error: '{filepath}' is a directory, not a file."
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file '{filepath}': {str(e)}"


@register_tool
def list_directory(filepath: str = ".", **kwargs) -> str:
    """Lists all files and folders in the specified directory path. Defaults to current directory."""
    # Accept 'path' as a fallback if the LLM uses it instead of 'filepath'
    target_dir = kwargs.get("path", filepath)

    if not os.path.exists(target_dir):
        return f"Error: Directory '{target_dir}' not found."
    if not os.path.isdir(target_dir):
        return f"Error: '{target_dir}' is not a directory."
    try:
        entries = os.listdir(target_dir)
        return str(entries)
    except Exception as e:
        return f"Error listing directory '{target_dir}': {str(e)}"


@register_tool
def delete_file(filepath: str, **kwargs) -> str:
    if not os.path.exists(filepath):
        return f"Error: Path '{filepath}' not found."
    try:
        if os.path.isfile(filepath):
            os.remove(filepath)
            return f"Successfully deleted file '{filepath}'."
        elif os.path.isdir(filepath):
            os.rmdir(filepath)
            return f"Successfully deleted empty directory '{filepath}'."
        else:
            return f"Error: '{filepath}' is neither a file nor a directory."
    except Exception as e:
        return f"Error deleting '{filepath}': {str(e)}"


@register_tool
def make_directory(path: str, **kwargs) -> str:
    try:
        os.makedirs(path, exist_ok=True)
        return f"Successfully created directory '{path}'."
    except Exception as e:
        return f"Error creating directory '{path}': {str(e)}"


@register_tool
def move_file(source: str, destination: str, **kwargs) -> str:
    if not os.path.exists(source):
        return f"Error: Source '{source}' not found."
    try:
        shutil.move(source, destination)
        return f"Successfully moved '{source}' to '{destination}'."
    except Exception as e:
        return f"Error moving '{source}' to '{destination}': {str(e)}"


@register_tool
def copy_file(source: str, destination: str, **kwargs) -> str:
    if not os.path.exists(source):
        return f"Error: Source '{source}' not found."
    try:
        if os.path.isdir(source):
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
        return f"Successfully copied '{source}' to '{destination}'."
    except Exception as e:
        return f"Error copying '{source}' to '{destination}': {str(e)}"


@register_tool
def get_file_info(filepath: str, **kwargs) -> str:
    if not os.path.exists(filepath):
        return f"Error: Path '{filepath}' not found."
    try:
        stat_info = os.stat(filepath)
        size = stat_info.st_size
        mtime = time.strftime('%Y-%m-%d %H:%M:%S',
                              time.localtime(stat_info.st_mtime))
        return f"Path: {filepath}\nSize: {size} bytes\nModified: {mtime}"
    except Exception as e:
        return f"Error getting info for '{filepath}': {str(e)}"


@register_tool
def start_background_process(script_path: str, **kwargs) -> str:
    if not os.path.exists(script_path):
        return f"Error: Script '{script_path}' not found."
    log_file = f"{os.path.basename(script_path)}.log"
    log_fh = open(log_file, "a", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            [sys.executable, script_path], stdout=log_fh, stderr=log_fh, cwd=os.getcwd())
        registry = _load_process_registry()
        registry[str(proc.pid)] = {"script": script_path,
                                   "log_file": log_file, "status": "running"}
        _update_process_registry(registry)
        return f"Successfully started background process. PID: {proc.pid}. Logs writing to {log_file}."
    except Exception as e:
        return f"Failed to start process: {str(e)}"


@register_tool
def check_process_status(pid: str, **kwargs) -> str:
    registry = _load_process_registry()
    if pid not in registry:
        return f"Error: Process {pid} not found in registry."
    info = registry[pid]
    log_file = info["log_file"]
    try:
        os.kill(int(pid), 0)
        status = "running"
    except OSError:
        status = "stopped/dead"
        info["status"] = status
        _update_process_registry(registry)
    logs = "(No logs found)"
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            logs = "".join(lines[-10:])
    return f"Process {pid} Status: {status}\nLast 10 log lines:\n{logs}"


@register_tool
def stop_background_process(pid: str, **kwargs) -> str:
    registry = _load_process_registry()
    if pid not in registry:
        return f"Error: Process {pid} not found in registry."
    try:
        os.kill(int(pid), signal.SIGTERM)
        registry[pid]["status"] = "terminated"
        _update_process_registry(registry)
        return f"Successfully sent SIGTERM to process {pid}."
    except Exception as e:
        return f"Failed to stop process {pid}: {str(e)}"


@register_tool
def install_package(package_name: str, **kwargs) -> str:
    if not package_name or " " in package_name:
        return "Error: Invalid package name."
    if ";" in package_name or "&" in package_name or "|" in package_name:
        return "Error: Invalid characters in package name."
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "install",
                                package_name], capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            return f"Successfully installed '{package_name}'."
        else:
            return f"Failed to install '{package_name}'.\nStderr: {result.stderr}"
    except Exception as e:
        return f"Error running pip install: {str(e)}"


@register_tool
def set_secret(key: str, value: str, **kwargs) -> str:
    if not key or not value:
        return "Error: Key and value must be provided."
    secrets = _load_secrets()
    secrets[key] = value
    _save_secrets(secrets)
    return f"Secret '{key}' saved successfully."


@register_tool
def get_secret(key: str, **kwargs) -> str:
    secrets = _load_secrets()
    if key in secrets:
        return secrets[key]
    return f"Error: Secret '{key}' not found."
