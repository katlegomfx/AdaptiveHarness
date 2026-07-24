import argparse
import signal
from concurrent.futures import ThreadPoolExecutor
import datetime
import inspect
import json
import os
import re
import sys
import uuid
import atexit
import time
import logging

import dynamic_tools
from dynamic_tools import REGISTRY_LOCK, TOOL_REGISTRY, get_relevant_tools
from memory import condense_history
from llm_backend import Message, chat, chat_sync, is_ollama
from sandbox import execute_tool_in_sandbox
from storage import init_db, list_sessions, load_latest_checkpoint, save_checkpoint, save_learning, retrieve_learnings
from runtime.result import ToolResult, ResultStatus
from observability.logger import logger, JsonFormatter
from observability.metrics import metrics

REASONING_MODEL = "ornith"
META_PROMPT_MODEL = "ornith"
TESTER_MODEL = "ornith"
SUMMARY_MODEL = "ornith"
LOG_FILE_PATH = "agent_execution.log"

MAX_HEAL_RETRIES = 2
MAX_AGENT_ITERATIONS = 25
YOLO_MODE = False

META_TOOLS = ("create_tool", "update_tool", "edit_source_file")

BASE_SYSTEM_PROMPT = (
    "You are an autonomous AI agent framework running in a Python environment.\n"
    "1. If the user asks for information or an action for which NO tool exists, you MUST call `create_tool`.\n"
    "2. CRITICAL - FUNCTION FORMAT: `python_code` MUST contain a complete `def` definition.\n"
    "3. FLEXIBLE SIGNATURES: Always include `**kwargs` as the final parameter.\n"
    "4. PARAMETER INVOCATION: BEFORE calling any registered tool, inspect its required arguments.\n"
    "5. RECOVERY: If `create_tool` returns an error, analyze the trace and call `create_tool` again with corrected code.\n"
    "6. EXECUTION LOOP: Once `create_tool` succeeds, call the newly registered tool.\n"
    "7. FILE I/O ENCODING: ALWAYS specify encoding='utf-8' explicitly.\n"
    "8. ERROR REPORTING: Tools MUST return descriptive error strings on failure, never bare booleans.\n"
    "9. EXISTING TOOLS: Before creating a tool, check if a tool with similar capabilities already exists.\n"
    "10. SELF-HEALING: If a tool execution fails with a traceback, the system will attempt to automatically patch the tool code.\n"
    "11. SELF-IMPROVEMENT: You can edit your own core source files using `edit_source_file`.\n"
    "12. TOOL OUTPUT QUALITY: Tools MUST return the actual data the user requested (e.g., file contents, search results, calculations), NOT just metadata (like success booleans or content lengths). If a tool returns metadata instead of content, it is broken.\n"
    "13. TOOL CORRECTION: If a tool runs successfully but returns the wrong data format (e.g., returning a dictionary when a string is expected), you MUST use `update_tool` to fix the return statement immediately.\n"
    "14. NO MODULE-LEVEL EXECUTION: Do NOT include module-level executable code (like calling the function or printing) in `python_code`. The sandbox runner imports the module and calls the function automatically. If you want to include local tests, wrap them in `if __name__ == '__main__':`.\n"
)

_log_fh = open(LOG_FILE_PATH, "a", encoding="utf-8", buffering=1)
atexit.register(_log_fh.close)

handler = logging.FileHandler("agent.jsonl")
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)


def _handle_interrupt(signum, frame):
    sys.stderr.write("\n[Interrupt] Force shutting down...\n")
    sys.stderr.flush()
    raise SystemExit(0)


def log_and_stream(text: str, end: str = "", flush: bool = True):
    sys.stdout.write(text + (end if end != "" else ""))
    if flush:
        sys.stdout.flush()
    _log_fh.write(text + (end if end != "" else ""))


def log_structured(level: str, msg: str, **kwargs):
    extra = {k: v for k, v in kwargs.items() if k in [
        "session_id", "tool", "turn", "trace_id"]}
    logger.log(getattr(logging, level.upper(), logging.INFO), msg, extra=extra)


def sanitize_tool_calls(tool_calls: list) -> list:
    if not tool_calls:
        return []
    sanitized = []
    for call in tool_calls:
        if isinstance(call, dict):
            sanitized.append(call)
        elif hasattr(call, "model_dump"):
            sanitized.append(call.model_dump())
        elif hasattr(call, "dict"):
            sanitized.append(call.dict())
        else:
            func = getattr(call, "function", None)
            sanitized.append({
                "function": {
                    "name": getattr(func, "name", ""),
                    "arguments": getattr(func, "arguments", {}),
                }
            })
    return sanitized


def generate_plan(user_prompt: str) -> str:
    log_and_stream("-> [Pass 0] Synthesizing execution plan...\n")
    with REGISTRY_LOCK:
        active_tools = list(TOOL_REGISTRY.values())
    tools_info = "\n".join(
        [f"- {t.__name__}: {inspect.getdoc(t) or 'No description'}" for t in active_tools])

    plan_messages = [
        {"role": "system", "content": f"You are an expert planner. Create a concise step-by-step plan for the user's request. Identify what tools need to be created or used.\n\nCurrently available tools:\n{tools_info}"},
        {"role": "user", "content": user_prompt}
    ]
    try:
        stream = chat(model=META_PROMPT_MODEL,
                      messages=plan_messages, stream=True)
        log_and_stream("[Plan]\n")
        plan = ""
        for chunk in stream:
            msg = chunk.get("message") if isinstance(
                chunk, dict) else getattr(chunk, "message", None)
            if not msg:
                continue
            chunk_content = getattr(msg, "content", None) or (
                msg.get("content") if isinstance(msg, dict) else None)
            if chunk_content:
                log_and_stream(chunk_content)
                plan += chunk_content
        log_and_stream("\n\n")
        return plan
    except Exception as e:
        log_and_stream(f"   [Planning Failed]: {e}\n")
        return ""


def generate_dynamic_system_prompt(goal: str) -> str:
    log_and_stream("-> [Pass 1] Synthesizing task-specific System Prompt...\n")
    cwd = os.getcwd()
    try:
        entries = os.listdir(cwd)
        dirs = [d for d in entries if os.path.isdir(os.path.join(cwd, d))]
        files = [f for f in entries if not os.path.isdir(os.path.join(cwd, f))]
        dir_context = f"Current Working Directory: {cwd}\nDirectories: {dirs}\nFiles: {files}"
    except Exception:
        dir_context = "Current Working Directory: Unknown"

    meta_messages = [
        {
            "role": "system",
            "content": (
                "You are an expert meta-prompt engineer for an autonomous Python agent. "
                "Analyze the user's core intent and goal, and write 3-5 concise, high-impact tactical directives. "
                "CRITICAL: Do NOT answer the user's prompt. Do NOT guess file contents. Do NOT hallucinate data. "
                "Only output the tactical directives the main agent must follow.\n\n"
                f"Environment Context:\n{dir_context}"
            ),
        },
        {"role": "user", "content": f"Target Goal:\n{goal}"},
    ]

    dynamic_rules = ""
    try:
        stream = chat(model=META_PROMPT_MODEL,
                      messages=meta_messages, stream=True)
        log_and_stream("[Meta-Prompt Directives]\n")
        for chunk in stream:
            msg = chunk.get("message") if isinstance(
                chunk, dict) else getattr(chunk, "message", None)
            if not msg:
                continue
            chunk_content = getattr(msg, "content", None) or (
                msg.get("content") if isinstance(msg, dict) else None)
            if chunk_content:
                log_and_stream(chunk_content)
                dynamic_rules += chunk_content
        log_and_stream(
            "\n\n-> [Pass 1 Complete] Dynamic instructions generated.\n\n")
        return dynamic_rules.strip()
    except Exception as e:
        log_and_stream(
            f"\n-> [Pass 1 Fallback] Meta-prompt generation skipped: {e}\n\n")
        return "Focus on safe, correct code execution and modular tool design."


def generate_adversarial_test_cases(tool_name: str, python_code: str) -> list[dict]:
    log_and_stream(
        f"-> [Pass 1.5 - Adversarial Tester Agent] Generating edge-case verification tests for '{tool_name}'...\n")
    tester_messages = [
        {
            "role": "system",
            "content": (
                "You are an Adversarial QA Testing Agent. Inspect the provided Python function signature. "
                "Return ONLY a raw JSON array of test case objects. Each object must have a 'name' and 'args' key. "
                "The 'args' keys MUST EXACTLY match the parameter names of the function. "
                "Generate 3 distinct test cases: happy path, empty input, and special characters/edge case. "
                "Use REAL, plausible values for file paths or inputs (e.g., 'README.md', '', '/tmp/nonexistent.txt'). "
                "Do NOT invent paths under /test/ that don't exist."
            ),
        },
        {"role": "user", "content": f"Target Function Code:\n{python_code}"},
    ]
    try:
        stream = chat(model=TESTER_MODEL,
                      messages=tester_messages, stream=True)
        content_buffer = ""
        for chunk in stream:
            msg = chunk.get("message") if isinstance(
                chunk, dict) else getattr(chunk, "message", None)
            if not msg:
                continue
            chunk_content = getattr(msg, "content", None) or (
                msg.get("content") if isinstance(msg, dict) else None)
            if chunk_content:
                content_buffer += chunk_content

        cleaned = content_buffer.strip().replace(
            "```json", "").replace("```", "").strip()
        parsed_args = json.loads(cleaned)
        if isinstance(parsed_args, list):
            log_and_stream(
                f"   [Adversarial Tests Generated]: {parsed_args}\n")
            return parsed_args
    except Exception as e:
        log_and_stream(
            f"   [Adversarial Tester Warning]: Failed generating custom tests ({e}). Defaulting to standard empty args.\n")
    return [{"name": "default", "args": {}}]


def resolve_tool_function(requested_name: str):
    with REGISTRY_LOCK:
        if requested_name in TOOL_REGISTRY:
            return requested_name, TOOL_REGISTRY[requested_name]
        normalized_target = requested_name.lower().replace("_", "").replace("-", "")
        for registered_name, func in TOOL_REGISTRY.items():
            normalized_reg = registered_name.lower().replace("_", "").replace("-", "")
            if normalized_target == normalized_reg:
                return registered_name, func
    return None, None


def ask_model_stream(messages: list, tools: list) -> Message:
    stream = chat(model=REASONING_MODEL, messages=messages,
                  tools=tools, think=True, stream=True)
    in_thinking = False
    in_content = False
    assembled_content = ""
    assembled_thinking = ""
    assembled_tool_calls = []

    for chunk in stream:
        msg = chunk.get("message") if isinstance(
            chunk, dict) else getattr(chunk, "message", None)
        if not msg:
            continue
        chunk_thinking = getattr(msg, "thinking", None) or (
            msg.get("thinking") if isinstance(msg, dict) else None)
        chunk_content = getattr(msg, "content", None) or (
            msg.get("content") if isinstance(msg, dict) else None)
        chunk_tools = getattr(msg, "tool_calls", None) or (
            msg.get("tool_calls") if isinstance(msg, dict) else None)

        if chunk_thinking:
            if not in_thinking:
                in_thinking = True
                log_and_stream("\n[Thinking]\n")
            log_and_stream(chunk_thinking)
            assembled_thinking += chunk_thinking
            continue

        if chunk_content:
            if in_thinking:
                in_thinking = False
                log_and_stream("\n")
            if not in_content:
                in_content = True
                log_and_stream("Assistant: ")
            log_and_stream(chunk_content)
            assembled_content += chunk_content

        if chunk_tools:
            if in_thinking or in_content:
                log_and_stream("\n")
                in_thinking = False
                in_content = False
            for call in chunk_tools:
                if call not in assembled_tool_calls:
                    assembled_tool_calls.append(call)
                    func_name = getattr(call.function, "name", "") if hasattr(
                        call, "function") else call.get("function", {}).get("name", "")
                    func_args = getattr(call.function, "arguments", {}) if hasattr(
                        call, "function") else call.get("function", {}).get("arguments", {})
                    log_and_stream(
                        f"-> Model requested tool: '{func_name}' | Args: {func_args}\n")

    log_and_stream("\n")
    return Message(
        role="assistant",
        content=assembled_content,
        tool_calls=sanitize_tool_calls(
            assembled_tool_calls) if assembled_tool_calls else None,
        thinking=assembled_thinking if assembled_thinking else None,
    )


def _get_call_details(call) -> tuple[str, dict]:
    if hasattr(call, "function"):
        return call.function.name, call.function.arguments
    elif isinstance(call, dict) and "function" in call:
        return call["function"]["name"], call["function"]["arguments"]
    return getattr(call, "name", ""), getattr(call, "arguments", {})


def _execute_once(call, trace_id: str) -> ToolResult:
    raw_func_name, args = _get_call_details(call)
    resolved_name, tool_func = resolve_tool_function(raw_func_name)

    if not tool_func:
        return ToolResult(ResultStatus.NOT_FOUND, f"Error: Tool '{raw_func_name}' is not registered.")

    if resolved_name in META_TOOLS and not YOLO_MODE:
        user_approval = input(
            f"\n> Agent wants to execute '{resolved_name}' with args: {args}. Approve? [y/N] ")
        if user_approval.lower() != 'y':
            return ToolResult(ResultStatus.VALIDATION_FAILURE, "Error: User denied execution.")

    if resolved_name not in META_TOOLS:
        try:
            sig = inspect.signature(tool_func)
            required_params = [
                p_name for p_name, p in sig.parameters.items()
                if p.default == inspect.Parameter.empty
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            ]
            if required_params and not args:
                return ToolResult(
                    ResultStatus.VALIDATION_FAILURE,
                    f"Tool Invocation Error: '{resolved_name}' requires parameters: {required_params}, but was called with empty arguments."
                )
        except Exception:
            pass

    start_time = time.time()
    status = ResultStatus.SUCCESS
    try:
        if resolved_name in META_TOOLS:
            log_and_stream(
                f"-> Executing meta-tool in process: '{resolved_name}' [Trace: {trace_id}]...\n")

            if resolved_name in ("create_tool", "update_tool"):
                target_tool = args.get("tool_name", "dynamic_func")
                code = args.get("python_code", "")
                test_cases = generate_adversarial_test_cases(target_tool, code)
                args["test_inputs"] = test_cases

            result_str = tool_func(**args)

            if "Successfully" in result_str:
                status = ResultStatus.SUCCESS
            else:
                status = ResultStatus.VALIDATION_FAILURE
            return ToolResult(status, result_str)
        else:
            log_and_stream(
                f"-> [Sandbox Execution] Tool: '{resolved_name}' [Trace: {trace_id}]...\n")
            return execute_tool_in_sandbox(resolved_name, args, timeout_seconds=120, ephemeral=False)
    except Exception as e:
        return ToolResult(ResultStatus.RUNTIME_FAILURE, f"Execution Error: {str(e)}")
    finally:
        duration = time.time() - start_time
        metrics.record_tool(resolved_name, duration,
                            status == ResultStatus.SUCCESS)
        log_structured("INFO", "Tool executed", tool=resolved_name,
                       trace_id=trace_id, success=status == ResultStatus.SUCCESS)


def execute_single_tool_call(call, max_heals=2) -> tuple[str, ToolResult]:
    heal_attempts = 0
    current_call = call
    trace_id = str(uuid.uuid4())[:8]

    while True:
        result = _execute_once(current_call, trace_id)
        resolved_name, _ = resolve_tool_function(
            _get_call_details(current_call)[0])

        is_dynamic_tool = resolved_name not in META_TOOLS

        if not is_dynamic_tool or result.is_success or result.status == ResultStatus.NOT_FOUND or heal_attempts >= max_heals:
            return resolved_name, result

        heal_attempts += 1
        log_and_stream(
            f"\n!!! [Self-Heal Triggered] Tool '{resolved_name}' crashed. Attempting automated patch ({heal_attempts}/{max_heals})...\n")
        metrics.record_heal(False)

        try:
            tool_path = os.path.join(
                dynamic_tools.CUSTOM_TOOLS_DIR, f"{resolved_name}.py")
            with open(tool_path, 'r', encoding='utf-8') as f:
                broken_code = f.read()
        except Exception:
            return resolved_name, result

        heal_messages = [
            {"role": "system", "content": "You are an autonomous debugging agent. A tool just crashed. Analyze the traceback and the code. Return ONLY the complete, fixed Python code for the function. Do not include explanations."},
            {"role": "user", "content": f"Tool Name: {resolved_name}\n\nTraceback/Error:\n{result.value}\n\nBroken Code:\n{broken_code}"}
        ]

        try:
            log_and_stream("   [Self-Heal] Generating fixed code:\n")
            stream = chat(model=META_PROMPT_MODEL,
                          messages=heal_messages, stream=True)
            fixed_code = ""
            for chunk in stream:
                msg = chunk.get("message") if isinstance(
                    chunk, dict) else getattr(chunk, "message", None)
                if not msg:
                    continue
                chunk_content = getattr(msg, "content", None) or (
                    msg.get("content") if isinstance(msg, dict) else None)
                if chunk_content:
                    log_and_stream(chunk_content)
                    fixed_code += chunk_content
            log_and_stream("\n")

            fixed_code = fixed_code.replace(
                "```python", "").replace("```", "").strip()
        except Exception as e:
            log_and_stream(
                f"   [Self-Heal Failed] LLM could not generate fix: {e}\n")
            return resolved_name, result

        if not fixed_code:
            return resolved_name, result

        heal_call = {
            "function": {
                "name": "update_tool",
                "arguments": {
                    "tool_name": resolved_name,
                    "python_code": fixed_code
                }
            }
        }

        heal_result = _execute_once(heal_call, trace_id)

        if heal_result.is_success:
            log_and_stream(
                "   [Self-Heal Success] Patch applied. Retrying original execution...\n")
            metrics.record_heal(True)
        else:
            log_and_stream(
                "   [Self-Heal Failed] Patch was rejected by sandbox validator.\n")
            result.value += f"\n\n[Self-Heal Attempt Failed]: {heal_result.value}"
            return resolved_name, result


def _extract_json_from_text(text: str) -> dict | None:
    """Helper to robustly extract the first JSON object block from LLM output."""
    # Try to find a ```json ... ``` block first
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        text = match.group(1)

    # Fallback to finding the first { ... } block
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        json_str = text[start:end+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None


def reflect_on_task(user_prompt: str, messages: list):
    """6.3 Reflection / Post-Mortem"""
    log_and_stream("\n-> [Post-Mortem] Reflecting on task execution...\n")
    reflect_messages = [
        {"role": "system",
            "content": "You are a reflection agent. Analyze the conversation. Did the tools work? Were any redundant? What should be remembered for future tasks? Output a JSON with 'learnings' (list of strings) and 'improvements' (string)."},
        {"role": "user",
            "content": f"Original Prompt: {user_prompt}\n\nConversation:\n{json.dumps(messages[-10:], default=str)}"}
    ]
    try:
        log_and_stream("[Reflection]\n")
        stream = chat(model=META_PROMPT_MODEL,
                      messages=reflect_messages, stream=True)
        content_buffer = ""
        for chunk in stream:
            msg = chunk.get("message") if isinstance(
                chunk, dict) else getattr(chunk, "message", None)
            if not msg:
                continue
            chunk_content = getattr(msg, "content", None) or (
                msg.get("content") if isinstance(msg, dict) else None)
            if chunk_content:
                log_and_stream(chunk_content)
                content_buffer += chunk_content

        log_and_stream("\n")

        if not content_buffer.strip():
            log_and_stream(
                "   [Reflection Skipped]: Model returned an empty response.\n")
            return

        data = _extract_json_from_text(content_buffer)
        if not data:
            log_and_stream(
                "   [Reflection Failed]: Could not extract valid JSON from model output.\n")
            return

        for learning in data.get("learnings", []):
            save_learning(learning)

        improvements = data.get("improvements", "")
        if improvements:
            with open("improvement_guide.md", "a", encoding="utf-8") as f:
                f.write(f"\n## Task: {user_prompt}\n{improvements}\n")

        log_and_stream("   [Reflection Complete]: Learnings saved.\n")

    except Exception as e:
        log_and_stream(f"   [Reflection Failed]: {e}\n")


def run_agent_loop(
    user_prompt: str, session_id: str, messages: list = None, turn_count: int = 0, plan: str = "", learnings: list = None
) -> tuple[list, int]:
    log_and_stream(
        f"\n=== User Turn: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log_and_stream(f"User > {user_prompt}\n\n")

    dynamic_instructions = generate_dynamic_system_prompt(user_prompt)
    plan_str = f"\n\nExecution Plan:\n{plan}" if plan else ""
    learnings_str = f"\n\nRelevant Past Learnings:\n{learnings}" if learnings else ""

    combined_system_prompt = {
        "role": "system",
        "content": f"{BASE_SYSTEM_PROMPT}\n\nTask Specific Directives:\n{dynamic_instructions}{plan_str}{learnings_str}",
    }

    if messages is None:
        messages = [combined_system_prompt]
    else:
        if messages[0].get("role") == "system":
            messages[0] = combined_system_prompt
        else:
            messages.insert(0, combined_system_prompt)

    messages.append({"role": "user", "content": user_prompt})

    iteration = 0
    tool_call_history = []

    while iteration < MAX_AGENT_ITERATIONS:
        iteration += 1

        messages = condense_history(
            messages, model_name=SUMMARY_MODEL, log_stream_func=log_and_stream)
        active_tools = get_relevant_tools(user_prompt, top_k=5)
        assistant_message = ask_model_stream(messages, active_tools)

        msg_dict = {
            "role": assistant_message.role,
            "content": assistant_message.content or "",
        }
        if assistant_message.tool_calls:
            msg_dict["tool_calls"] = assistant_message.tool_calls
        if hasattr(assistant_message, "thinking") and assistant_message.thinking:
            msg_dict["thinking"] = assistant_message.thinking

        messages.append(msg_dict)

        if not assistant_message.tool_calls:
            turn_count += 1
            save_checkpoint(session_id, turn_count, messages)
            break

        tool_calls = assistant_message.tool_calls
        results = []

        # Track tool name AND arguments to prevent false loop positives
        # (e.g., reading multiple different files is not a loop)
        current_tool_calls = []
        for tc in tool_calls:
            func_name = tc.get("function", {}).get("name", "")
            func_args = tc.get("function", {}).get("arguments", {})
            try:
                args_str = json.dumps(func_args, sort_keys=True)
            except Exception:
                args_str = str(func_args)
            current_tool_calls.append((func_name, args_str))

        tool_call_history.extend(current_tool_calls)

        loop_detected = False
        if len(tool_call_history) > 3:
            recent_calls = tool_call_history[-3:]
            # A true loop is calling the exact same tool with the exact same arguments
            if len(set(recent_calls)) == 1 and recent_calls[0][0] not in META_TOOLS:
                loop_detected = True
                log_and_stream(
                    f"\n[Safety] Loop detected: Agent called '{recent_calls[0][0]}' with identical args 3 times in a row. Forcing reflection.\n")
                for tc in tool_calls:
                    messages.append({
                        "role": "tool",
                        "name": tc.get("function", {}).get("name", ""),
                        "content": "CRITICAL ERROR: You are stuck in a loop calling this tool with the exact same arguments. The tool may be returning the wrong data (e.g., metadata instead of actual content). Do NOT call this tool again. Instead, use `update_tool` to fix the tool's logic so it returns the actual content/data the user requested, NOT just metadata or success booleans."
                    })
                tool_call_history.clear()
                turn_count += 1
                save_checkpoint(session_id, turn_count, messages)
                continue

        if len(tool_calls) == 1:
            results.append(execute_single_tool_call(tool_calls[0]))
        else:
            has_meta_tools = any(
                (tc.get("function", {}).get("name") in META_TOOLS)
                for tc in tool_calls
            )
            if has_meta_tools:
                log_and_stream(
                    "-> Detected meta-tools in batch. Forcing sequential execution...\n")
                results = [execute_single_tool_call(tc) for tc in tool_calls]
            else:
                log_and_stream(
                    f"-> Parallelizing {len(tool_calls)} safe tool calls...\n")
                with ThreadPoolExecutor(max_workers=min(len(tool_calls), 5)) as executor:
                    results = list(executor.map(
                        execute_single_tool_call, tool_calls))

        for tool_name, result in results:
            log_and_stream(f"   Result [{tool_name}]: {result.value}\n\n")
            messages.append({
                "role": "tool",
                "name": tool_name,
                "content": result.value,
            })

        turn_count += 1
        save_checkpoint(session_id, turn_count, messages)
    else:
        log_and_stream(
            f"[Safety] Reached max iterations ({MAX_AGENT_ITERATIONS}). Forcing stop.\n")
        messages.append(
            {"role": "system", "content": "Forced stop: iteration budget exhausted."})

    return messages, turn_count


def main():
    global YOLO_MODE
    signal.signal(signal.SIGINT, _handle_interrupt)
    init_db()

    parser = argparse.ArgumentParser(description="Ollama Dynamic Tool Agent")
    parser.add_argument("--session", type=str, help="Session ID to resume")
    parser.add_argument("--list-sessions", action="store_true",
                        help="List all past saved sessions")
    parser.add_argument("--yolo", action="store_true",
                        help="Skip human-in-the-loop approval")
    args = parser.parse_args()

    YOLO_MODE = args.yolo

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

    session_id = args.session or str(uuid.uuid4())[:8]
    history = None

    if args.session:
        restored = load_latest_checkpoint(session_id)
        if restored:
            history = restored
            log_and_stream(
                f"=== Resumed Session: {session_id} ({len(history)} messages loaded) ===\n")
        else:
            log_and_stream(
                f"=== Session '{session_id}' not found. Starting fresh session. ===\n")
            session_id = str(uuid.uuid4())[:8]
    else:
        log_and_stream(
            f"=== Ollama Dynamic Agent Ready | Session: {session_id} ===\n")

    log_and_stream(
        f"Reasoning Model: '{REASONING_MODEL}' | Meta-Prompt Model: '{META_PROMPT_MODEL}' | Tester Model: '{TESTER_MODEL}'\n")
    with REGISTRY_LOCK:
        active_tools = list(TOOL_REGISTRY.keys())
    log_and_stream(f"Active Registry Tools: {active_tools}\n\n")
    log_and_stream(
        f"LLM Backend: {'Ollama' if is_ollama() else 'llama.cpp'}\n\n")

    turn_count = len(history) if history else 0

    while True:
        try:
            user_input = input("User > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break

            plan = generate_plan(user_input)
            learnings = retrieve_learnings(user_input)

            history, turn_count = run_agent_loop(
                user_input, session_id, history, turn_count, plan, learnings
            )

            reflect_on_task(user_input, history)

        except (KeyboardInterrupt, SystemExit):
            log_and_stream("\nExiting...\n")
            sys.exit(0)


if __name__ == "__main__":
    main()
