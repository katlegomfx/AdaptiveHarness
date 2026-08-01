# src/tools/dynamic.py
import ast
import importlib
import inspect
import json
import os
import re
import sys
import threading
from typing import Callable, List, Tuple

from src.config import CUSTOM_TOOLS_DIR
from src.tools.registry import TOOL_REGISTRY, REGISTRY_LOCK, register_tool

CUSTOM_TOOLS_WRITE_LOCK = threading.RLock()
BANNED_IMPORTS = {"ctypes", "pickle", "socket", "builtins"}
BANNED_FUNCTIONS = {"eval", "exec", "compile", "__import__"}


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
                from src.llm_backend import get_embedding, cosine_similarity
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
    from src.tools.sandbox import execute_tool_in_sandbox
    from src.safety_net import snapshot, revert

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
    from src.tools.sandbox import execute_tool_in_sandbox
    from src.safety_net import snapshot, revert

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
    from src.safety_net import snapshot, revert, validate_python_syntax

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


# Initialize dynamic tools on load
load_persisted_tools()
