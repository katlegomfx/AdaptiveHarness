import ast
import importlib
import inspect
import math
import os
import re
import sys
import threading
from typing import Callable, Dict, List, Tuple
from runtime.result import ResultStatus

TOOL_REGISTRY: Dict[str, Callable] = {}
REGISTRY_LOCK = threading.RLock()
# 3.8 Parallel Execution Race on custom_tools/
CUSTOM_TOOLS_WRITE_LOCK = threading.RLock()
CUSTOM_TOOLS_DIR = os.path.join(os.path.dirname(__file__), "custom_tools")

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

    # 5.6 validate_python_code Regex for Boolean Returns is Fragile
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


def check_semantic_duplicates(new_name: str, new_doc: str, threshold: float = 0.75) -> Tuple[bool, str]:
    with REGISTRY_LOCK:
        for name, func in TOOL_REGISTRY.items():
            if name == new_name:
                continue
            existing_doc = inspect.getdoc(func) or ""
            tokens1 = set(re.findall(
                r"\w+", (new_name + " " + new_doc).lower()))
            tokens2 = set(re.findall(
                r"\w+", (name + " " + existing_doc).lower()))
            if not tokens1 or not tokens2:
                continue
            intersection = tokens1.intersection(tokens2)
            union = tokens1.union(tokens2)
            similarity = len(intersection) / len(union) if union else 0.0
            if similarity >= threshold:
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
        for func in tools:
            name = func.__name__
            doc = inspect.getdoc(func) or ""
            text = f"{name} {doc}".lower()
            text_tokens = set(re.findall(r"\w+", text))
            if name in ("create_tool", "update_tool"):
                score = 999.0
            else:
                overlap = len(query_tokens.intersection(text_tokens))
                score = overlap / \
                    math.sqrt(len(text_tokens) + 1) if text_tokens else 0.0
            scored_tools.append((score, func))
        scored_tools.sort(key=lambda x: x[0], reverse=True)
        return [func for _, func in scored_tools[:top_k]]


def _extract_sample_args(test_inputs):
    """Helper to extract arguments from various test input formats."""
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

        # 3.8 Parallel Execution Race on custom_tools/
        with CUSTOM_TOOLS_WRITE_LOCK:
            file_path = os.path.join(
                CUSTOM_TOOLS_DIR, f"{actual_func_name}.py")
            snapshot(f"Before creating tool {actual_func_name}")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(python_code)
            importlib.invalidate_caches()

        # 5.5 Adversarial Tester Generates Only One Test Case
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


load_persisted_tools()
