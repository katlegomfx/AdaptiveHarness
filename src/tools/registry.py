import threading
from typing import Callable, Dict

TOOL_REGISTRY: Dict[str, Callable] = {}
REGISTRY_LOCK = threading.RLock()

META_TOOLS = {"create_tool", "update_tool", "edit_source_file", "write_file"}

BUILTIN_TOOLS = {
    "create_tool", "update_tool", "edit_source_file", "write_file",
    "read_file", "list_directory", "delete_file", "make_directory",
    "move_file", "copy_file", "get_file_info", "start_background_process",
    "check_process_status", "stop_background_process", "install_package",
    "set_secret", "get_secret",
}


def is_builtin_tool(name: str) -> bool:
    return name in BUILTIN_TOOLS


def register_tool(func: Callable):
    with REGISTRY_LOCK:
        TOOL_REGISTRY[func.__name__] = func
    return func


def get_relevant_tools(user_query: str, top_k: int = 5) -> list[Callable]:
    import inspect
    import math
    import re
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
            score = overlap / (math.sqrt(len(text_tokens) + 1)
                               if text_tokens else 0.0)
            scored_tools.append((score, func))

        scored_tools.sort(key=lambda x: x[0], reverse=True)

        dynamic_to_take = top_k - len(meta_tools_found)
        if dynamic_to_take < 0:
            dynamic_to_take = 0

        result = meta_tools_found + [func for _,
                                     func in scored_tools[:dynamic_to_take]]
        return result[:top_k]
