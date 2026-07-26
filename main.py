# main.py
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
from typing import Callable, Optional, List, Dict, Any
import urllib.error
from dotenv import load_dotenv

import dynamic_tools
from dynamic_tools import REGISTRY_LOCK, TOOL_REGISTRY, get_relevant_tools
from memory import condense_history
from llm_backend import Message, chat, chat_sync, is_ollama, get_embedding
from sandbox import execute_tool_in_sandbox
from storage import init_db, list_sessions, load_latest_checkpoint, save_checkpoint, save_learning, retrieve_learnings, add_long_term_goal
from runtime.result import ToolResult, ResultStatus
from observability.logger import logger, JsonFormatter
from observability.metrics import metrics

load_dotenv()

# Load model configurations from environment variables, fallback to "ornith"
REASONING_MODEL = os.environ.get("REASONING_MODEL", "ornith")
META_PROMPT_MODEL = os.environ.get("META_PROMPT_MODEL", "ornith")
TESTER_MODEL = os.environ.get("TESTER_MODEL", "ornith")
SUMMARY_MODEL = os.environ.get("SUMMARY_MODEL", "ornith")
THINK_ENABLED = os.environ.get(
    "THINK_ENABLED", "false").strip().lower() in ("true", "1", "yes", "on")
LOG_FILE_PATH = "agent_execution.log"

MAX_HEAL_RETRIES = 2
MAX_AGENT_ITERATIONS = 25
META_TOOLS = ("create_tool", "update_tool", "edit_source_file", "write_file")

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
    "15. STANDALONE SCRIPTS & FILES: If the user asks you to create a file, script, or project, use the `write_file` tool. Do NOT use `create_tool` for creating static files or standalone scripts. `create_tool` is ONLY for creating reusable Python functions that you will call repeatedly.\n"
)

_log_fh = open(LOG_FILE_PATH, "a", encoding="utf-8", buffering=1)
atexit.register(_log_fh.close)

handler = logging.FileHandler("agent.jsonl")
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)


class Agent:
    def __init__(self, session_id: str, yolo_mode: bool = False):
        self.session_id = session_id
        self.yolo_mode = yolo_mode
        self.autonomous_mode = False
        self.history: List[Dict] = []
        self.turn_count = 0

        self.on_stream: Callable[[str], None] = self._default_stream_handler
        self.on_input: Callable[[str], str] = self._default_input_handler

    def _default_stream_handler(self, text: str):
        sys.stdout.write(text)
        sys.stdout.flush()
        _log_fh.write(text)

    def _default_input_handler(self, prompt: str) -> str:
        return input(prompt)

    def emit(self, text: str, end: str = "\n"):
        self.on_stream(f"{text}{end}")

    def log_structured(self, level: str, msg: str, **kwargs):
        extra = {k: v for k, v in kwargs.items() if k in [
            "session_id", "tool", "turn", "trace_id"]}
        logger.log(getattr(logging, level.upper(),
                   logging.INFO), msg, extra=extra)

    def sanitize_tool_calls(self, tool_calls: list) -> list:
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
                sanitized.append({"function": {"name": getattr(
                    func, "name", ""), "arguments": getattr(func, "arguments", {})}})
        return sanitized

    def _print_stream_and_get_content(self, stream, header: str = None, end: str = "\n\n") -> str:
        """Helper to consume a chat stream, print thinking & content phases, and return final content."""
        if header:
            self.emit(header, end="")

        in_thinking = False
        in_content = False
        content = ""

        for chunk in stream:
            msg = chunk.get("message") if isinstance(
                chunk, dict) else getattr(chunk, "message", None)
            if not msg:
                continue

            chunk_thinking = getattr(msg, "thinking", None) or (
                msg.get("thinking") if isinstance(msg, dict) else None)
            chunk_content = getattr(msg, "content", None) or (
                msg.get("content") if isinstance(msg, dict) else None)

            if chunk_thinking:
                if not in_thinking:
                    in_thinking = True
                    self.emit("\n[Thinking]\n", end="")
                self.emit(chunk_thinking, end="")

            if chunk_content:
                if in_thinking:
                    in_thinking = False
                    self.emit("\n[Response]\n", end="")
                if not in_content:
                    in_content = True
                self.emit(chunk_content, end="")
                content += chunk_content

        if in_thinking or in_content:
            self.emit(end)

        return content

    def generate_plan(self, user_prompt: str) -> str:
        self.emit("-> [Pass 0] Synthesizing execution plan...")
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
            plan = self._print_stream_and_get_content(
                stream, header="[Plan]\n", end="\n\n")
            return plan, tools_info
        except Exception as e:
            self.emit(f"   [Planning Failed]: {e}\n")
            return "", tools_info

    def generate_dynamic_system_prompt(self, goal: str, plan: str, tools_info: str) -> str:
        self.emit("-> [Pass 1] Synthesizing task-specific System Prompt...")
        cwd = os.getcwd()
        try:
            entries = os.listdir(cwd)
            dirs = [d for d in entries if os.path.isdir(os.path.join(cwd, d))]
            files = [f for f in entries if not os.path.isdir(
                os.path.join(cwd, f))]
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
                    f"Environment Context:\n{dir_context}\n\n"
                    f"Available Tools:\n{tools_info}\n\n"
                    f"Execution Plan:\n{plan}"
                ),
            },
            {"role": "user", "content": f"Target Goal:\n{goal}"},
        ]
        try:
            stream = chat(model=META_PROMPT_MODEL,
                          messages=meta_messages, stream=True)
            dynamic_rules = self._print_stream_and_get_content(
                stream, header="[Meta-Prompt Directives]\n", end="\n\n-> [Pass 1 Complete] Dynamic instructions generated.\n\n")
            return dynamic_rules.strip()
        except Exception as e:
            self.emit(
                f"\n-> [Pass 1 Fallback] Meta-prompt generation skipped: {e}\n\n")
            return "Focus on safe, correct code execution and modular tool design."

    def generate_adversarial_test_cases(self, tool_name: str, python_code: str) -> list[dict]:
        self.emit(
            f"-> [Pass 1.5 - Adversarial Tester Agent] Generating edge-case verification tests for '{tool_name}'...\n")
        tester_messages = [
            {"role": "system",
                "content": "You are an Adversarial QA Testing Agent. Inspect the provided Python function signature. Return ONLY a raw JSON array of test case objects. Each object must have a 'name' and 'args' key. The 'args' keys MUST EXACTLY match the parameter names of the function. Generate 3 distinct test cases: happy path, empty input, and special characters/edge case. Use REAL, plausible values for file paths or inputs (e.g., 'README.md', '', '/tmp/nonexistent.txt'). Do NOT invent paths under /test/ that don't exist."},
            {"role": "user", "content": f"Target Function Code:\n{python_code}"},
        ]
        try:
            stream = chat(model=TESTER_MODEL,
                          messages=tester_messages, stream=True)
            content_buffer = self._print_stream_and_get_content(
                stream, header="[Adversarial Tester]\n", end="\n")
            cleaned = content_buffer.strip().replace(
                "```json", "").replace("```", "").strip()
            parsed_args = json.loads(cleaned)
            if isinstance(parsed_args, list):
                self.emit(f"   [Adversarial Tests Generated]: {parsed_args}\n")
                return parsed_args
        except Exception as e:
            self.emit(
                f"   [Adversarial Tester Warning]: Failed generating custom tests ({e}). Defaulting to standard empty args.\n")
        return [{"name": "default", "args": {}}]

    def resolve_tool_function(self, requested_name: str):
        with REGISTRY_LOCK:
            if requested_name in TOOL_REGISTRY:
                return requested_name, TOOL_REGISTRY[requested_name]
            normalized_target = requested_name.lower().replace("_", "").replace("-", "")
            for registered_name, func in TOOL_REGISTRY.items():
                normalized_reg = registered_name.lower().replace("_", "").replace("-", "")
                if normalized_target == normalized_reg:
                    return registered_name, func
        return None, None

    def ask_model_stream(self, messages: list, tools: list) -> Message:
        try:
            stream = chat(model=REASONING_MODEL, messages=messages,
                          tools=tools, think=THINK_ENABLED, stream=True)
        except urllib.error.HTTPError as e:
            if e.code == 400 and THINK_ENABLED:
                # Retry without think — model may not support thinking mode
                self.emit(
                    "   [Warning] Model rejected think=true. Retrying without thinking mode...\n")
                stream = chat(model=REASONING_MODEL, messages=messages,
                              tools=tools, think=False, stream=True)
            else:
                raise

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
                    self.emit("\n[Thinking]\n", end="")
                self.emit(chunk_thinking, end="")
                assembled_thinking += chunk_thinking
                continue

            if chunk_content:
                if in_thinking:
                    in_thinking = False
                    self.emit("\n[Response]\n", end="")
                if not in_content:
                    in_content = True
                self.emit(chunk_content, end="")
                assembled_content += chunk_content

            if chunk_tools:
                if in_thinking or in_content:
                    self.emit("\n")
                    in_thinking = False
                    in_content = False
                for call in chunk_tools:
                    if call not in assembled_tool_calls:
                        assembled_tool_calls.append(call)
                        func_name = getattr(call.function, "name", "") if hasattr(
                            call, "function") else call.get("function", {}).get("name", "")
                        func_args = getattr(call.function, "arguments", {}) if hasattr(
                            call, "function") else call.get("function", {}).get("arguments", {})
                        self.emit(
                            f"-> Model requested tool: '{func_name}' | Args: {func_args}\n")

        self.emit("\n")
        return Message(
            role="assistant", content=assembled_content,
            tool_calls=self.sanitize_tool_calls(
                assembled_tool_calls) if assembled_tool_calls else None,
            thinking=assembled_thinking if assembled_thinking else None,
        )

    def _get_call_details(self, call) -> tuple[str, dict]:
        if hasattr(call, "function"):
            return call.function.name, call.function.arguments
        elif isinstance(call, dict) and "function" in call:
            return call["function"]["name"], call["function"]["arguments"]
        return getattr(call, "name", ""), getattr(call, "arguments", {})

    def _execute_once(self, call, trace_id: str) -> ToolResult:
        raw_func_name, args = self._get_call_details(call)
        resolved_name, tool_func = self.resolve_tool_function(raw_func_name)

        if not tool_func:
            return ToolResult(ResultStatus.NOT_FOUND, f"Error: Tool '{raw_func_name}' is not registered.")

        if self.autonomous_mode and resolved_name == "edit_source_file":
            patch_id = str(uuid.uuid4())[:8]
            patch_dir = "pending_patches"
            os.makedirs(patch_dir, exist_ok=True)
            patch_path = os.path.join(patch_dir, f"patch_{patch_id}.json")

            with open(patch_path, "w", encoding="utf-8") as f:
                json.dump({
                    "trace_id": trace_id,
                    "args": args,
                    "timestamp": datetime.datetime.now().isoformat()
                }, f, indent=2)

            msg = f"Autonomous mode: Source edit intercepted. Patch saved to {patch_path}. Awaiting human approval (`approve patch {os.path.basename(patch_path)}`)."
            self.emit(f"-> {msg}\n")
            return ToolResult(ResultStatus.SUCCESS, msg)

        if resolved_name in META_TOOLS and not self.yolo_mode:
            user_approval = self.on_input(
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
                    return ToolResult(ResultStatus.VALIDATION_FAILURE, f"Tool Invocation Error: '{resolved_name}' requires parameters: {required_params}, but was called with empty arguments.")
            except Exception:
                pass

        start_time = time.time()
        status = ResultStatus.SUCCESS
        try:
            if resolved_name in META_TOOLS:
                self.emit(
                    f"-> Executing meta-tool in process: '{resolved_name}' [Trace: {trace_id}]...\n")
                if resolved_name in ("create_tool", "update_tool"):
                    target_tool = args.get("tool_name", "dynamic_func")
                    code = args.get("python_code", "")
                    test_cases = self.generate_adversarial_test_cases(
                        target_tool, code)
                    args["test_inputs"] = test_cases

                result_str = tool_func(**args)
                if "Successfully" in result_str:
                    status = ResultStatus.SUCCESS
                else:
                    status = ResultStatus.VALIDATION_FAILURE
                return ToolResult(status, result_str)
            else:
                self.emit(
                    f"-> [Sandbox Execution] Tool: '{resolved_name}' [Trace: {trace_id}]...\n")
                return execute_tool_in_sandbox(resolved_name, args, timeout_seconds=120, ephemeral=False)
        except Exception as e:
            return ToolResult(ResultStatus.RUNTIME_FAILURE, f"Execution Error: {str(e)}")
        finally:
            duration = time.time() - start_time
            metrics.record_tool(resolved_name, duration,
                                status == ResultStatus.SUCCESS)
            self.log_structured("INFO", "Tool executed", tool=resolved_name,
                                trace_id=trace_id, success=status == ResultStatus.SUCCESS)

    def execute_single_tool_call(self, call, max_heals=2) -> tuple[str, ToolResult]:
        heal_attempts = 0
        current_call = call
        trace_id = str(uuid.uuid4())[:8]

        while True:
            result = self._execute_once(current_call, trace_id)
            resolved_name, _ = self.resolve_tool_function(
                self._get_call_details(current_call)[0])

            is_dynamic_tool = resolved_name not in META_TOOLS

            if not is_dynamic_tool or result.is_success or result.status == ResultStatus.NOT_FOUND or heal_attempts >= max_heals:
                return resolved_name, result

            heal_attempts += 1
            self.emit(
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
                self.emit("   [Self-Heal] Generating fixed code:\n", end="")
                stream = chat(model=META_PROMPT_MODEL,
                              messages=heal_messages, stream=True)
                fixed_code = self._print_stream_and_get_content(
                    stream, header="", end="\n")
                fixed_code = fixed_code.replace(
                    "```python", "").replace("```", "").strip()
            except Exception as e:
                self.emit(
                    f"   [Self-Heal Failed] LLM could not generate fix: {e}\n")
                return resolved_name, result

            if not fixed_code:
                return resolved_name, result

            heal_call = {"function": {"name": "update_tool", "arguments": {
                "tool_name": resolved_name, "python_code": fixed_code}}}
            heal_result = self._execute_once(heal_call, trace_id)

            if heal_result.is_success:
                self.emit(
                    "   [Self-Heal Success] Patch applied. Retrying original execution...\n")
                metrics.record_heal(True)
            else:
                self.emit(
                    "   [Self-Heal Failed] Patch was rejected by sandbox validator.\n")
                result.value += f"\n\n[Self-Heal Attempt Failed]: {heal_result.value}"
                return resolved_name, result

    def _extract_json_from_text(self, text: str) -> dict | None:
        match = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if match:
            text = match.group(1)
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end+1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                return None
        return None

    def reflect_on_task(self, user_prompt: str, messages: list):
        self.emit("\n-> [Post-Mortem] Reflecting on task execution...\n")
        reflect_messages = [
            {"role": "system",
                "content": "You are a reflection agent. Analyze the conversation. Did the tools work? Were any redundant? What should be remembered for future tasks? Output a JSON with 'learnings' (list of strings) and 'improvements' (string)."},
            {"role": "user",
                "content": f"Original Prompt: {user_prompt}\n\nConversation:\n{json.dumps(messages[-10:], default=str)}"}
        ]
        try:
            stream = chat(model=META_PROMPT_MODEL,
                          messages=reflect_messages, stream=True)
            content_buffer = self._print_stream_and_get_content(
                stream, header="[Reflection]\n", end="\n")

            if not content_buffer.strip():
                self.emit(
                    "   [Reflection Skipped]: Model returned an empty response.\n")
                return

            data = self._extract_json_from_text(content_buffer)
            if not data:
                self.emit(
                    "   [Reflection Failed]: Could not extract valid JSON from model output.\n")
                return

            for learning in data.get("learnings", []):
                emb = get_embedding(learning)
                save_learning(learning, embedding=emb)

            improvements = data.get("improvements", "")
            if improvements:
                with open("improvement_guide.md", "a", encoding="utf-8") as f:
                    f.write(f"\n## Task: {user_prompt}\n{improvements}\n")

            self.emit("   [Reflection Complete]: Learnings saved.\n")
        except Exception as e:
            self.emit(f"   [Reflection Failed]: {e}\n")

    def run_agent_loop(self, user_prompt: str, plan: str, tools_info: str, learnings: list = None) -> None:
        self.emit(
            f"\n=== User Turn: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        self.emit(f"User > {user_prompt}\n\n")

        dynamic_instructions = self.generate_dynamic_system_prompt(
            user_prompt, plan, tools_info)
        plan_str = f"\n\nExecution Plan:\n{plan}" if plan else ""
        learnings_str = f"\n\nRelevant Past Learnings:\n{learnings}" if learnings else ""

        combined_system_prompt = {
            "role": "system",
            "content": f"{BASE_SYSTEM_PROMPT}\n\nTask Specific Directives:\n{dynamic_instructions}{plan_str}{learnings_str}",
        }

        if not self.history:
            self.history = [combined_system_prompt]
        else:
            if self.history[0].get("role") == "system":
                self.history[0] = combined_system_prompt
            else:
                self.history.insert(0, combined_system_prompt)

        self.history.append({"role": "user", "content": user_prompt})

        iteration = 0
        tool_call_history = []

        while iteration < MAX_AGENT_ITERATIONS:
            iteration += 1

            self.history = condense_history(
                self.history, model_name=SUMMARY_MODEL, log_stream_func=self.on_stream)
            active_tools = get_relevant_tools(user_prompt, top_k=5)
            assistant_message = self.ask_model_stream(
                self.history, active_tools)

            msg_dict = {"role": assistant_message.role,
                        "content": assistant_message.content or ""}
            if assistant_message.tool_calls:
                msg_dict["tool_calls"] = assistant_message.tool_calls
            if hasattr(assistant_message, "thinking") and assistant_message.thinking:
                msg_dict["thinking"] = assistant_message.thinking

            self.history.append(msg_dict)

            if not assistant_message.tool_calls:
                self.turn_count += 1
                save_checkpoint(self.session_id, self.turn_count, self.history)
                break

            tool_calls = assistant_message.tool_calls
            results = []

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
                if len(set(recent_calls)) == 1 and recent_calls[0][0] not in META_TOOLS:
                    loop_detected = True
                    self.emit(
                        f"\n[Safety] Loop detected: Agent called '{recent_calls[0][0]}' with identical args 3 times in a row. Forcing reflection.\n")
                    for tc in tool_calls:
                        self.history.append({
                            "role": "tool",
                            "name": tc.get("function", {}).get("name", ""),
                            "content": "CRITICAL ERROR: You are stuck in a loop calling this tool with the exact same arguments. The tool may be returning the wrong data (e.g., metadata instead of actual content). Do NOT call this tool again. Instead, use `update_tool` to fix the tool's logic so it returns the actual content/data the user requested, NOT just metadata or success booleans."
                        })
                    tool_call_history.clear()
                    self.turn_count += 1
                    save_checkpoint(self.session_id,
                                    self.turn_count, self.history)
                    continue

            if len(tool_calls) == 1:
                results.append(self.execute_single_tool_call(tool_calls[0]))
            else:
                has_meta_tools = any((tc.get("function", {}).get(
                    "name") in META_TOOLS) for tc in tool_calls)
                if has_meta_tools:
                    self.emit(
                        "-> Detected meta-tools in batch. Forcing sequential execution...\n")
                    results = [self.execute_single_tool_call(
                        tc) for tc in tool_calls]
                else:
                    self.emit(
                        f"-> Parallelizing {len(tool_calls)} safe tool calls...\n")
                    with ThreadPoolExecutor(max_workers=min(len(tool_calls), 5)) as executor:
                        results = list(executor.map(
                            self.execute_single_tool_call, tool_calls))

            for tool_name, result in results:
                self.emit(f"   Result [{tool_name}]: {result.value}\n\n")
                self.history.append(
                    {"role": "tool", "name": tool_name, "content": result.value})

            self.turn_count += 1
            save_checkpoint(self.session_id, self.turn_count, self.history)
        else:
            self.emit(
                f"[Safety] Reached max iterations ({MAX_AGENT_ITERATIONS}). Forcing stop.\n")
            self.history.append(
                {"role": "system", "content": "Forced stop: iteration budget exhausted."})

    def _is_goal_satisfied(self, goal_text: str) -> bool:
        verification = [
            {"role": "system", "content": "You verify whether a stated goal has been accomplished based on the recent agent conversation. Return JSON: {\"satisfied\": true|false, \"reason\": \"...\"}."},
            {"role": "user",
                "content": f"Goal: {goal_text}\n\nRecent conversation:\n{json.dumps(self.history[-8:], default=str)}"}
        ]
        try:
            stream = chat(model=META_PROMPT_MODEL,
                          messages=verification, stream=True)
            buf = self._print_stream_and_get_content(
                stream, header="[Goal Verification]\n", end="\n")
            data = self._extract_json_from_text(buf)
            if data:
                return bool(data.get("satisfied", False))
        except Exception:
            pass
        return False

    def run_autonomous_cycle(self, allow_self_edit: bool = False) -> bool:
        from goals import select_autonomous_task
        from storage import mark_goal_attempted, mark_goal_completed, mark_goal_blocked, log_system_improvement

        self.autonomous_mode = True
        self.emit(
            "\n=== [Autonomous Mode] No user input. Selecting task... ===\n")

        state_text = " ".join([m.get("content", "")
                              for m in self.history[-4:] if m.get("content")])
        state_emb = get_embedding(state_text)

        task = select_autonomous_task(
            allow_self_edit=allow_self_edit, current_state_emb=state_emb)

        if task["type"] == "none":
            self.emit(
                "   [Autonomous] Nothing to do (no goals, self-edit disabled). Sleeping.\n")
            self.autonomous_mode = False
            return False

        if task["type"] == "long_term_goal":
            goal = task["goal"]
            gid, gtext, attempts = goal["id"], goal["goal_text"], goal["attempts"]
            self.emit(
                f"   [Autonomous] Long-term goal #{gid} (attempt {attempts + 1}): {gtext}\n")
            mark_goal_attempted(gid)

            prompt = f"[Autonomous Task — Long-Term Goal #{gid}]\n{gtext}"
            try:
                plan, tools_info = self.generate_plan(prompt)
                learnings = retrieve_learnings(
                    gtext, query_emb=get_embedding(gtext))
                self.run_agent_loop(prompt, plan, tools_info, learnings)
                self.reflect_on_task(prompt, self.history)

                if self._is_goal_satisfied(gtext):
                    mark_goal_completed(
                        gid, "Verified complete by LLM after autonomous cycle.")
                    self.emit(
                        f"   [Autonomous] Goal #{gid} marked COMPLETED.\n")
                self.autonomous_mode = False
                return True
            except Exception as e:
                self.emit(f"   [Autonomous] Goal #{gid} failed: {e}\n")
                if attempts + 1 >= 3:
                    mark_goal_blocked(
                        gid, f"Blocked after {attempts + 1} attempts: {e}")
                    self.emit(f"   [Autonomous] Goal #{gid} marked BLOCKED.\n")
                self.autonomous_mode = False
                return False

        if task["type"] == "system_improvement":
            imp = task["improvement"]
            fpath = imp.get("file", "")
            issue = imp.get("issue", "")
            action = imp.get("suggested_action", "")
            self.emit(f"   [Autonomous] Self-improvement target: {fpath}\n")
            self.emit(f"     Issue:  {issue}\n")
            self.emit(f"     Action: {action}\n")

            prompt = (
                f"[Autonomous Task — System Self-Improvement]\n"
                f"File: {fpath}\nIssue: {issue}\nSuggested action: {action}\n\n"
                f"Use `edit_source_file` to apply a MINIMAL, SAFE patch. "
                f"Do not rewrite large sections. After patching, read the modified "
                f"region back to confirm the change is correct."
            )
            try:
                plan, tools_info = self.generate_plan(prompt)
                learnings = retrieve_learnings(
                    f"improve {fpath}", query_emb=get_embedding(f"improve {fpath}"))
                self.run_agent_loop(prompt, plan, tools_info, learnings)
                self.reflect_on_task(prompt, self.history)
                log_system_improvement(fpath, issue, "applied")
                self.autonomous_mode = False
                return True
            except Exception as e:
                self.emit(f"   [Autonomous] Self-improvement failed: {e}\n")
                log_system_improvement(fpath, issue, f"failed: {e}")
                self.autonomous_mode = False
                return False
        self.autonomous_mode = False
        return False

    def load_state(self):
        restored = load_latest_checkpoint(self.session_id)
        if restored:
            self.history = restored
            self.turn_count = len(self.history)
            self.emit(
                f"=== Resumed Session: {self.session_id} ({len(self.history)} messages loaded) ===\n")
        else:
            self.emit(
                f"=== Session '{self.session_id}' not found. Starting fresh session. ===\n")
            self.session_id = str(uuid.uuid4())[:8]


IDLE_TIMEOUT_SECONDS = 240
AUTONOMOUS_COOLDOWN_SECONDS = 120
MAX_AUTONOMOUS_CYCLES_IN_A_ROW = 12


def handle_special_commands(agent: "Agent", user_input: str) -> bool:
    from goals import list_long_term_goals, extract_goals_from_conversation
    from storage import add_long_term_goal, mark_goal_completed

    lowered = user_input.lower().strip()

    if lowered.startswith("goal:"):
        text = user_input[5:].strip()
        if text:
            gid = add_long_term_goal(text, priority=5, source="user")
            agent.emit(f"   [Goal Saved] #{gid}: {text}\n")
        return True

    if lowered.startswith("priority goal:"):
        text = user_input[len("priority goal:"):].strip()
        if text:
            gid = add_long_term_goal(text, priority=9, source="user")
            agent.emit(f"   [Priority Goal Saved] #{gid}: {text}\n")
        return True

    if lowered in ("goals", "list goals"):
        goals = list_long_term_goals()
        agent.emit(f"\n=== Long-Term Goals ({len(goals)}) ===\n")
        for g in goals:
            agent.emit(
                f"  [#{g['id']}] [{g['status']}] P:{g['priority']} attempts:{g['attempts']} — {g['goal_text']}\n")
        agent.emit("\n")
        return True

    if lowered.startswith("complete goal"):
        try:
            gid = int(lowered.split()[-1])
            mark_goal_completed(gid, "Marked complete by user")
            agent.emit(f"   [Goal #{gid} marked complete]\n")
        except (ValueError, IndexError):
            agent.emit("   Usage: complete goal <id>\n")
        return True

    if lowered == "list patches":
        patch_dir = "pending_patches"
        if os.path.exists(patch_dir):
            patches = [f for f in os.listdir(patch_dir) if f.endswith(".json")]
            agent.emit(f"\n=== Pending Patches ({len(patches)}) ===\n")
            for p in patches:
                agent.emit(f"  - {p}\n")
            agent.emit("Use: approve patch <filename>\n\n")
        else:
            agent.emit("No pending patches.\n")
        return True

    if lowered.startswith("approve patch"):
        try:
            patch_file = lowered.split(" ", 2)[2]
            patch_path = os.path.join("pending_patches", patch_file)
            if not os.path.exists(patch_path):
                agent.emit(f"   Error: Patch file '{patch_file}' not found.\n")
                return True

            with open(patch_path, "r", encoding="utf-8") as f:
                patch_data = json.load(f)

            agent.emit(f"   [Applying Patch] {patch_file}...\n")
            agent.autonomous_mode = False
            call = {"function": {"name": "edit_source_file",
                                 "arguments": patch_data["args"]}}
            res = agent._execute_once(call, patch_data["trace_id"])

            if res.is_success:
                os.remove(patch_path)
                agent.emit(f"   [Patch Applied & Removed]: {res.value}\n")
            else:
                agent.emit(f"   [Patch Failed]: {res.value}\n")
        except Exception as e:
            agent.emit(f"   [Approve Patch Error]: {e}\n")
        return True

    return False


def run_blocking_loop(agent: "Agent", use_tui: bool = False) -> None:
    tui = None
    if use_tui:
        import tui as tui_module
        tui = tui_module.CursesTUI()
        agent.on_stream = tui.stream_handler
        agent.on_input = tui.input_handler
        sys.stdout = tui_module.StdoutRedirector(tui)

    try:
        while True:
            try:
                if tui:
                    user_input = tui.get_input_async(3600)
                    if user_input is None:
                        continue
                    user_input = user_input.strip()
                else:
                    user_input = input("User > ").strip()

                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    break
                if handle_special_commands(agent, user_input):
                    continue

                plan, tools_info = agent.generate_plan(user_input)
                learnings = retrieve_learnings(
                    user_input, query_emb=get_embedding(user_input))
                agent.run_agent_loop(user_input, plan, tools_info, learnings)
                agent.reflect_on_task(user_input, agent.history)
            except (KeyboardInterrupt, SystemExit):
                agent.emit("\nExiting...\n")
                sys.exit(0)
    finally:
        if tui:
            sys.stdout = sys.__stdout__
            tui.cleanup()


def run_scheduled_loop(agent: "Agent", idle_timeout: int, allow_self_edit: bool, use_tui: bool = False) -> None:
    from async_input import AsyncInputReader
    from goals import extract_goals_from_conversation, list_long_term_goals

    tui = None
    if use_tui:
        import tui as tui_module
        tui = tui_module.CursesTUI()
        agent.on_stream = tui.stream_handler
        agent.on_input = tui.input_handler
        sys.stdout = tui_module.StdoutRedirector(tui)

    try:
        reader = None
        if not tui:
            reader = AsyncInputReader()

        consecutive_autonomous = 0

        agent.emit(
            f"Autonomous Mode: ENABLED  (idle timeout: {idle_timeout}s, self-edit: {'ON' if allow_self_edit else 'OFF'})\n")

        pending = list_long_term_goals(status="pending")
        if pending:
            agent.emit(f"Pending long-term goals: {len(pending)}\n")
            for g in pending[:3]:
                agent.emit(f"  [#{g['id']}] {g['goal_text']}\n")
        agent.emit("\n")

        while True:
            try:
                if tui:
                    user_input = tui.get_input_async(idle_timeout)
                    if user_input is not None:
                        user_input = user_input.strip()
                else:
                    agent.emit("User > ", end="")
                    sys.stdout.flush()
                    user_input = reader.get_input(timeout=idle_timeout)

                if user_input is None:
                    consecutive_autonomous += 1
                    if consecutive_autonomous > MAX_AUTONOMOUS_CYCLES_IN_A_ROW:
                        agent.emit(
                            f"\n[Safety] Hit {MAX_AUTONOMOUS_CYCLES_IN_A_ROW} consecutive autonomous cycles. Pausing for user.\n")
                        consecutive_autonomous = 0
                        continue
                    agent.run_autonomous_cycle(allow_self_edit=allow_self_edit)
                    time.sleep(AUTONOMOUS_COOLDOWN_SECONDS)
                    continue

                consecutive_autonomous = 0
                agent.emit("\n")
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    break
                if handle_special_commands(agent, user_input):
                    continue

                plan, tools_info = agent.generate_plan(user_input)
                learnings = retrieve_learnings(
                    user_input, query_emb=get_embedding(user_input))
                agent.run_agent_loop(user_input, plan, tools_info, learnings)
                agent.reflect_on_task(user_input, agent.history)

                try:
                    new_goals = extract_goals_from_conversation(
                        user_input, agent.history)
                    for g in new_goals:
                        gid = add_long_term_goal(
                            g, priority=5, source="reflection", embedding=get_embedding(g))
                        agent.emit(
                            f"   [Goal Captured from conversation] #{gid}: {g}\n")
                except Exception:
                    pass

            except (KeyboardInterrupt, SystemExit):
                agent.emit("\nExiting...\n")
                sys.exit(0)
    finally:
        if tui:
            sys.stdout = sys.__stdout__
            tui.cleanup()


def main():
    signal.signal(signal.SIGINT, lambda s, f: (sys.stderr.write(
        "\n[Interrupt] Force shutting down...\n"), sys.exit(0)))
    init_db()

    parser = argparse.ArgumentParser(description="Ollama Dynamic Tool Agent")
    parser.add_argument("--session", type=str, help="Session ID to resume")
    parser.add_argument("--list-sessions", action="store_true")
    parser.add_argument("--yolo", action="store_true",
                        help="Skip HITL approval")
    parser.add_argument("--no-autonomous", action="store_true",
                        help="Disable scheduler; block on input (legacy behavior)")
    parser.add_argument("--idle-timeout", type=int, default=IDLE_TIMEOUT_SECONDS,
                        help=f"Seconds to wait for input before autonomous wake-up (default: {IDLE_TIMEOUT_SECONDS})")
    parser.add_argument("--autonomous-self-edit", action="store_true",
                        help="DANGER: allow autonomous cycles to edit source files without approval. Without this flag, only long-term goals are pursued autonomously.")
    parser.add_argument("--tui", action="store_true",
                        help="Enable Text User Interface (curses)")
    args = parser.parse_args()

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

    agent = Agent(session_id=args.session or str(
        uuid.uuid4())[:8], yolo_mode=args.yolo)

    if args.session:
        agent.load_state()
    else:
        agent.emit(
            f"=== Ollama Dynamic Agent Ready | Session: {agent.session_id} ===\n")

    agent.emit(
        f"Reasoning: '{REASONING_MODEL}' | Meta: '{META_PROMPT_MODEL}' | Tester: '{TESTER_MODEL}'\n")
    with REGISTRY_LOCK:
        agent.emit(f"Active Tools: {list(TOOL_REGISTRY.keys())}\n")
    agent.emit(f"LLM Backend: {'Ollama' if is_ollama() else 'llama.cpp'}\n")

    if args.no_autonomous:
        run_blocking_loop(agent, use_tui=args.tui)
    else:
        run_scheduled_loop(agent, idle_timeout=args.idle_timeout,
                           allow_self_edit=args.autonomous_self_edit, use_tui=args.tui)


if __name__ == "__main__":
    main()
