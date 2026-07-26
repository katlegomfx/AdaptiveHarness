# dynamic_tools.py
import json
import signal
import subprocess
import ast
import importlib
import inspect
import math
import os
import re
import shutil
import sys
import threading
import time
from typing import Callable, Dict, List, Tuple
from runtime.result import ResultStatus

TOOL_REGISTRY: Dict[str, Callable] = {}
REGISTRY_LOCK = threading.RLock()
CUSTOM_TOOLS_WRITE_LOCK = threading.RLock()
CUSTOM_TOOLS_DIR = os.path.join(os.path.dirname(__file__), "custom_tools")

META_TOOLS = {"create_tool", "update_tool", "edit_source_file", "write_file"}

BANNED_IMPORTS = {"ctypes", "pickle", "socket", "builtins"}
BANNED_FUNCTIONS = {"eval", "exec", "compile", "__import__"}


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


class SecurityASTVisitor(ast.NodeVisitor):
    def __init__(self, banned_imports: set, banned_funcs: set):
        self.banned_imports = banned_imports
        self.banned_funcs = banned_funcs
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            root_module = alias.name.split(".")[0]
            if root_module in self.banned_imports:
                self.violations.append(
                    f"Security Violation: Import of '{alias.name}' is prohibited.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            root_module = node.module.split(".")[0]
            if root_module in self.banned_imports:
                self.violations.append(
                    f"Security Violation: Import from '{node.module}' is prohibited.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name in self.banned_funcs:
            self.violations.append(
                f"Security Violation: Dynamic evaluation call '{func_name}()' is prohibited.")
        self.generic_visit(node)


def register_tool(func: Callable):
    with REGISTRY_LOCK:
        TOOL_REGISTRY[func.__name__] = func
    return func


def validate_python_code(python_code: str) -> Tuple[bool, str]:
    try:
        tree = ast.parse(python_code)
    except SyntaxError as e:
        return False, f"Syntax error in tool code: {str(e)}"

    visitor = SecurityASTVisitor(BANNED_IMPORTS, BANNED_FUNCTIONS)
    visitor.visit(tree)

    if visitor.violations:
        return False, " | ".join(visitor.violations)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if not ast.get_docstring(node):
                return False, f"Validation Error: Function '{node.name}' must include a docstring."
            continue
        if isinstance(node, ast.ClassDef):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.If):
            test = node.test
            is_main_check = False
            if isinstance(test, ast.Compare):
                if isinstance(test.left, ast.Name) and test.left.id == '__name__':
                    is_main_check = True
                elif test.comparators and isinstance(test.comparators[0], ast.Name) and test.comparators[0].id == '__name__':
                    is_main_check = True
            if is_main_check:
                continue
            return False, "Validation Error: Module-level executable code is prohibited. If you need to test locally, wrap calls in `if __name__ == '__main__':`."
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Call):
                return False, "Validation Error: Module-level function calls are prohibited."
            continue
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.value, ast.Call):
                return False, "Validation Error: Module-level function calls are prohibited."
            continue
        if isinstance(node, ast.Expr):
            return False, "Validation Error: Module-level executable expressions (like print()) are prohibited."

        return False, "Validation Error: Module-level executable code is prohibited."

    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, bool):
                return False, f"Validation Error: Function must not return bare booleans. Return descriptive error strings instead."

    return True, "Validation successful"


def extract_docstring_and_name(python_code: str) -> Tuple[str, str]:
    try:
        tree = ast.parse(python_code)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                doc = ast.get_docstring(node) or ""
                return node.name, doc
    except Exception:
        pass
    return "", ""


def check_semantic_duplicates(new_name: str, new_doc: str, threshold: float = 0.85) -> Tuple[bool, str]:
    """Checks for duplicates using LLM embeddings for high accuracy."""
    with REGISTRY_LOCK:
        for name, func in TOOL_REGISTRY.items():
            if name == new_name:
                continue
            existing_doc = inspect.getdoc(func) or ""

            try:
                from llm_backend import get_embedding, cosine_similarity
                v1 = get_embedding(new_name + " " + new_doc)
                v2 = get_embedding(name + " " + existing_doc)
                sim = cosine_similarity(v1, v2)
                if sim >= threshold:
                    return True, f"Duplicate detected! Capabilities closely match existing tool '{name}' (similarity score: {sim:.2f})."
            except Exception:
                # Fallback to simple token overlap if embeddings fail
                tokens1 = set(re.findall(
                    r"\w+", (new_name + " " + new_doc).lower()))
                tokens2 = set(re.findall(
                    r"\w+", (name + " " + existing_doc).lower()))
                if not tokens1 or not tokens2:
                    continue
                intersection = tokens1.intersection(tokens2)
                union = tokens1.union(tokens2)
                similarity = len(intersection) / len(union) if union else 0.0
                if similarity >= 0.75:
                    return True, f"Duplicate detected! Capabilities closely match existing tool '{name}' (similarity score: {similarity:.2f})."
    return False, ""


def extract_defined_functions(python_code: str) -> list[str]:
    try:
        tree = ast.parse(python_code)
        return [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    except Exception:
        return []


def ensure_package_init():
    os.makedirs(CUSTOM_TOOLS_DIR, exist_ok=True)
    init_file = os.path.join(CUSTOM_TOOLS_DIR, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w", encoding="utf-8") as f:
            f.write("# Auto-generated package init\n")


def load_persisted_tools():
    ensure_package_init()
    for filename in os.listdir(CUSTOM_TOOLS_DIR):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = filename[:-3]
            package_module = f"custom_tools.{module_name}"
            try:
                if package_module in sys.modules:
                    module = importlib.reload(sys.modules[package_module])
                else:
                    module = importlib.import_module(package_module)
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if inspect.isfunction(attr) and attr.__module__ == package_module and not attr_name.startswith("_"):
                        with REGISTRY_LOCK:
                            TOOL_REGISTRY[attr_name] = attr
                        print(f"[Registry] Loaded tool: '{attr_name}'")
            except Exception as e:
                print(f"[Registry] Failed loading '{filename}': {e}")


def get_relevant_tools(user_query: str, top_k: int = 5) -> list[Callable]:
    with REGISTRY_LOCK:
        tools = list(TOOL_REGISTRY.values())
        if len(tools) <= top_k:
            return tools

        query_tokens = set(re.findall(r"\w+", user_query.lower()))
        scored_tools = []
        meta_tools_found = []

        for func in tools:
            name = func.__name__
            if name in META_TOOLS:
                meta_tools_found.append(func)
                continue

            doc = inspect.getdoc(func) or ""
            text = f"{name} {doc}".lower()
            text_tokens = set(re.findall(r"\w+", text))
            overlap = len(query_tokens.intersection(text_tokens))
            score = overlap / \
                math.sqrt(len(text_tokens) + 1) if text_tokens else 0.0
            scored_tools.append((score, func))

        scored_tools.sort(key=lambda x: x[0], reverse=True)

        dynamic_to_take = top_k - len(meta_tools_found)
        if dynamic_to_take < 0:
            dynamic_to_take = 0

        result = meta_tools_found + [func for _,
                                     func in scored_tools[:dynamic_to_take]]
        return result[:top_k]


def _extract_sample_args(test_inputs):
    sample_args_list = test_inputs if isinstance(
        test_inputs, list) else [test_inputs]
    extracted_args = []

    for sample_args in sample_args_list:
        if isinstance(sample_args, dict):
            if "args" in sample_args and isinstance(sample_args["args"], dict):
                extracted_args.append(sample_args["args"])
            else:
                found = False
                for meta_key in ["test_cases", "test1", "sample_args", "edge_cases", "edge_case"]:
                    if meta_key in sample_args:
                        val = sample_args[meta_key]
                        if isinstance(val, dict):
                            extracted_args.append(val)
                            found = True
                            break
                        elif isinstance(val, list) and val and isinstance(val[0], dict):
                            extracted_args.append(val[0])
                            found = True
                            break
                if not found:
                    extracted_args.append(sample_args)
        else:
            extracted_args.append({})

    return extracted_args


@register_tool
def create_tool(tool_name: str, python_code: str, test_inputs: dict = None) -> str:
    """Creates, registers, and persists a brand-new Python tool after 3-stage validation."""
    from sandbox import execute_tool_in_sandbox
    from safety_net import snapshot, revert

    is_valid, validation_msg = validate_python_code(python_code)
    if not is_valid:
        return f"Tool creation rejected in Stage 1 (AST Verification): {validation_msg}"

    defined_funcs = extract_defined_functions(python_code)
    if not defined_funcs:
        return "Error: No top-level function definition found in python_code."

    actual_func_name = tool_name if tool_name in defined_funcs else defined_funcs[0]
    func_name, docstring = extract_docstring_and_name(python_code)

    with REGISTRY_LOCK:
        if actual_func_name in TOOL_REGISTRY and tool_name != f"update_{actual_func_name}":
            return f"Error: Tool '{actual_func_name}' already exists. Use 'update_tool' to patch bugs."

    is_dup, dup_msg = check_semantic_duplicates(actual_func_name, docstring)
    if is_dup:
        return f"Tool creation rejected in Stage 2 (Deduplication Check): {dup_msg}"

    try:
        ensure_package_init()
        with CUSTOM_TOOLS_WRITE_LOCK:
            file_path = os.path.join(
                CUSTOM_TOOLS_DIR, f"{actual_func_name}.py")
            snapshot(f"Before creating tool {actual_func_name}")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(python_code)
            importlib.invalidate_caches()

        extracted_args = _extract_sample_args(test_inputs)
        for sample_args in extracted_args:
            test_result = execute_tool_in_sandbox(
                actual_func_name, sample_args, timeout_seconds=15)
            if not test_result.is_success:
                with CUSTOM_TOOLS_WRITE_LOCK:
                    revert()
                return f"Tool creation rejected in Stage 3 (Sandbox Test Run Failed). Code automatically reverted.\n{test_result.value}"

        with CUSTOM_TOOLS_WRITE_LOCK:
            package_module = f"custom_tools.{actual_func_name}"
            if package_module in sys.modules:
                module = importlib.reload(sys.modules[package_module])
            else:
                module = importlib.import_module(package_module)

            func = getattr(module, actual_func_name, None)
            if not func or not callable(func):
                revert()
                return f"Error: Target function '{actual_func_name}' could not be loaded into runtime. Reverted."

            with REGISTRY_LOCK:
                TOOL_REGISTRY[actual_func_name] = func
                if tool_name != actual_func_name:
                    TOOL_REGISTRY[tool_name] = func

        return f"Successfully verified and registered tool '{actual_func_name}'. Passed AST, Deduplication, and Sandbox Verification stages."
    except Exception as e:
        revert()
        return f"Failed to create tool '{tool_name}': {str(e)}. Reverted."


@register_tool
def update_tool(tool_name: str, python_code: str, test_inputs: dict = None) -> str:
    """Updates an existing tool in the registry after security and sandbox validation."""
    from sandbox import execute_tool_in_sandbox
    from safety_net import snapshot, revert

    is_valid, validation_msg = validate_python_code(python_code)
    if not is_valid:
        return f"Update rejected in Stage 1 (AST Verification): {validation_msg}"

    defined_funcs = extract_defined_functions(python_code)
    if not defined_funcs:
        return "Error: No top-level function definition found in python_code."

    actual_func_name = tool_name if tool_name in defined_funcs else defined_funcs[0]

    with REGISTRY_LOCK:
        if actual_func_name not in TOOL_REGISTRY:
            return f"Error: Tool '{actual_func_name}' does not exist. Use create_tool instead."

    try:
        with CUSTOM_TOOLS_WRITE_LOCK:
            file_path = os.path.join(
                CUSTOM_TOOLS_DIR, f"{actual_func_name}.py")
            snapshot(f"Before updating tool {actual_func_name}")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(python_code)
            importlib.invalidate_caches()

        extracted_args = _extract_sample_args(test_inputs)
        for sample_args in extracted_args:
            test_result = execute_tool_in_sandbox(
                actual_func_name, sample_args, timeout_seconds=15)
            if not test_result.is_success:
                with CUSTOM_TOOLS_WRITE_LOCK:
                    revert()
                return f"Update rejected (Sandbox Failed). Automatically reverted.\n{test_result.value}"

        with CUSTOM_TOOLS_WRITE_LOCK:
            package_module = f"custom_tools.{actual_func_name}"
            if package_module in sys.modules:
                module = importlib.reload(sys.modules[package_module])
            else:
                module = importlib.import_module(package_module)

            func = getattr(module, actual_func_name, None)
            if not func or not callable(func):
                revert()
                return f"Error: Updated '{actual_func_name}' could not be loaded. Reverted."

            with REGISTRY_LOCK:
                TOOL_REGISTRY[actual_func_name] = func
                if tool_name != actual_func_name:
                    TOOL_REGISTRY[tool_name] = func

        return f"Successfully updated and verified tool '{actual_func_name}'."
    except Exception as e:
        revert()
        return f"Failed to update tool '{tool_name}': {str(e)}. Reverted."


@register_tool
def edit_source_file(filepath: str, search_string: str, replace_string: str, **kwargs) -> str:
    """Applies a precise string replacement patch to a local project file."""
    from safety_net import snapshot, revert, validate_python_syntax

    if not os.path.exists(filepath):
        return f"Error: File '{filepath}' not found."
    if not search_string.strip():
        return "Error: search_string cannot be empty."

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    occurrences = content.count(search_string)
    if occurrences == 0:
        return f"Error: search_string not found in {filepath}."
    if occurrences > 1:
        return f"Error: Found {occurrences} matches. Please make search_string more specific."

    snapshot(f"Before editing {filepath}")

    new_content = content.replace(search_string, replace_string, 1)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

    if filepath.endswith(".py"):
        is_valid, err_msg = validate_python_syntax(filepath)
        if not is_valid:
            revert()
            return f"Error: Patch introduced a syntax error. Automatically reverted. Details: {err_msg}"

    return f"Successfully patched '{filepath}'. 1 replacement made."


@register_tool
def write_file(filepath: str, content: str, **kwargs) -> str:
    """Writes content to a file, creating parent directories if needed."""
    dir_name = os.path.dirname(filepath)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    return f"Successfully wrote {len(content)} characters to {filepath}."


@register_tool
def read_file(filepath: str, **kwargs) -> str:
    """Reads the content of a file and returns it as a string."""
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
def list_directory(path: str = ".", **kwargs) -> str:
    """Lists all files and folders in the specified directory path. Defaults to current directory."""
    if not os.path.exists(path):
        return f"Error: Directory '{path}' not found."
    if not os.path.isdir(path):
        return f"Error: '{path}' is not a directory."
    try:
        entries = os.listdir(path)
        return str(entries)
    except Exception as e:
        return f"Error listing directory '{path}': {str(e)}"


@register_tool
def delete_file(filepath: str, **kwargs) -> str:
    """Deletes a file. Can also delete empty directories."""
    if not os.path.exists(filepath):
        return f"Error: Path '{filepath}' not found."
    try:
        if os.path.isfile(filepath):
            os.remove(filepath)
            return f"Successfully deleted file '{filepath}'."
        elif os.path.isdir(filepath):
            os.rmdir(filepath)  # Only removes empty directories
            return f"Successfully deleted empty directory '{filepath}'."
        else:
            return f"Error: '{filepath}' is neither a file nor a directory."
    except Exception as e:
        return f"Error deleting '{filepath}': {str(e)}"


@register_tool
def make_directory(path: str, **kwargs) -> str:
    """Creates a new directory, including parent directories if they don't exist."""
    try:
        os.makedirs(path, exist_ok=True)
        return f"Successfully created directory '{path}'."
    except Exception as e:
        return f"Error creating directory '{path}': {str(e)}"


@register_tool
def move_file(source: str, destination: str, **kwargs) -> str:
    """Moves or renames a file or directory from source to destination."""
    if not os.path.exists(source):
        return f"Error: Source '{source}' not found."
    try:
        shutil.move(source, destination)
        return f"Successfully moved '{source}' to '{destination}'."
    except Exception as e:
        return f"Error moving '{source}' to '{destination}': {str(e)}"


@register_tool
def copy_file(source: str, destination: str, **kwargs) -> str:
    """Copies a file or directory from source to destination."""
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
    """Retrieves metadata (size, modification time) for a file or directory."""
    if not os.path.exists(filepath):
        return f"Error: Path '{filepath}' not found."
    try:
        stat_info = os.stat(filepath)
        size = stat_info.st_size
        mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat_info.st_mtime))
        return f"Path: {filepath}\nSize: {size} bytes\nModified: {mtime}"
    except Exception as e:
        return f"Error getting info for '{filepath}': {str(e)}"


@register_tool
def start_background_process(script_path: str, **kwargs) -> str:
    """Starts a Python script as a long-running background process. Returns the PID."""
    if not os.path.exists(script_path):
        return f"Error: Script '{script_path}' not found."

    log_file = f"{os.path.basename(script_path)}.log"
    log_fh = open(log_file, "a", encoding="utf-8")

    try:
        proc = subprocess.Popen(
            [sys.executable, script_path],
            stdout=log_fh,
            stderr=log_fh,
            cwd=os.getcwd()
        )

        registry = _load_process_registry()
        registry[str(proc.pid)] = {
            "script": script_path,
            "log_file": log_file,
            "status": "running"
        }
        _update_process_registry(registry)

        return f"Successfully started background process. PID: {proc.pid}. Logs writing to {log_file}."
    except Exception as e:
        return f"Failed to start process: {str(e)}"


@register_tool
def check_process_status(pid: str, **kwargs) -> str:
    """Checks the status and last 10 lines of logs for a background process."""
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
    """Terminates a running background process by PID."""
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
    """Installs a Python package using pip in the current environment."""
    if not package_name or " " in package_name:
        return "Error: Invalid package name."

    if ";" in package_name or "&" in package_name or "|" in package_name:
        return "Error: Invalid characters in package name."

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0:
            return f"Successfully installed '{package_name}'."
        else:
            return f"Failed to install '{package_name}'.\nStderr: {result.stderr}"
    except Exception as e:
        return f"Error running pip install: {str(e)}"


@register_tool
def set_secret(key: str, value: str, **kwargs) -> str:
    """Saves a secret (like an API key or password) securely to the agent's secrets file."""
    if not key or not value:
        return "Error: Key and value must be provided."

    secrets = _load_secrets()
    secrets[key] = value
    _save_secrets(secrets)
    return f"Secret '{key}' saved successfully."


@register_tool
def get_secret(key: str, **kwargs) -> str:
    """Retrieves a saved secret by its key."""
    secrets = _load_secrets()
    if key in secrets:
        return secrets[key]
    return f"Error: Secret '{key}' not found."


load_persisted_tools()