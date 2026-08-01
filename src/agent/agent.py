from src.config import (
    LOG_FILE_PATH, JSON_LOG_PATH, PATCH_DIR,
    REASONING_MODEL, META_PROMPT_MODEL, TESTER_MODEL, SUMMARY_MODEL, THINK_ENABLED, CUSTOM_TOOLS_DIR, IMPROVEMENT_GUIDE_PATH,
    REASONING_TEMP, META_PROMPT_TEMP, TESTER_TEMP  # <--- ADD THIS
)
from src.agent.context import build_aspect_context
from src.memory.storage import save_aspect_memory
from src.agent.prompts import BASE_SYSTEM_PROMPT
from src.memory.reflection import reflect_on_task, _extract_json_from_text
from src.agent.planner import generate_plan, generate_dynamic_system_prompt
from src.core.metrics import metrics
from src.core.logger import logger, JsonFormatter
from src.core.result import ToolResult, ResultStatus
from src.tools.sandbox import execute_tool_in_sandbox
from src.llm_backend import Message, chat, get_embedding
from src.memory.storage import (
    save_checkpoint, retrieve_learnings, retrieve_summaries, save_summary,
    load_latest_checkpoint, save_learning
)
from src.memory.condenser import condense_history
from src.tools.registry import (
    REGISTRY_LOCK, TOOL_REGISTRY, get_relevant_tools, is_builtin_tool, META_TOOLS,
)
from typing import Callable, Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import urllib.error
import logging
import time
import atexit
import uuid
import sys
import re
import os
import json
import inspect
import datetime
import ast


MAX_AGENT_ITERATIONS = 25
MAX_HEAL_RETRIES = 2

_log_fh = open(LOG_FILE_PATH, "a", encoding="utf-8", buffering=1)
atexit.register(_log_fh.close)

_handler = logging.FileHandler(JSON_LOG_PATH)
_handler.setFormatter(JsonFormatter())
logger.addHandler(_handler)


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
        text_buffer = ""
        suppress_output = False

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

                text_buffer += chunk_content
                content += chunk_content

                # State machine to filter out <think> tags from the TUI stream
                while text_buffer:
                    if suppress_output:
                        end_idx = text_buffer.find("</think>")
                        if end_idx != -1:
                            text_buffer = text_buffer[end_idx + 8:]
                            suppress_output = False
                        else:
                            break
                    else:
                        start_idx = text_buffer.find("<think>")
                        if start_idx != -1:
                            if start_idx > 0:
                                self.emit(text_buffer[:start_idx], end="")
                            text_buffer = text_buffer[start_idx + 7:]
                            suppress_output = True
                        else:
                            # Hold back the last 7 chars to avoid splitting a tag across chunks
                            if len(text_buffer) > 7:
                                emit_len = len(text_buffer) - 7
                                self.emit(text_buffer[:emit_len], end="")
                                text_buffer = text_buffer[emit_len:]
                            else:
                                break

        if not suppress_output and text_buffer:
            self.emit(text_buffer, end="")

        if in_thinking or in_content:
            self.emit(end)

        # Final cleanup of any unclosed tags in the returned content
        if content:
            content = re.sub(r'<think>.*?</think>', '',
                             content, flags=re.DOTALL).strip()
            if '<think>' in content:
                content = re.sub(r'<think>.*', '', content,
                                 flags=re.DOTALL).strip()
            if '</think>' in content:
                content = re.sub(r'</think>', '', content).strip()

        return content

    def generate_adversarial_test_cases(self, tool_name: str, python_code: str) -> list[dict]:
        self.emit(
            f"-> [Pass 1.5 - Adversarial Tester Agent] Generating edge-case verification tests for '{tool_name}'...\n")
        context_messages = build_aspect_context(
            self, "tester", tool_name, recent_n=0)

        tester_messages = [
            {"role": "system",
                "content": "You are an Adversarial QA Testing Agent. Inspect the provided Python function signature. Return ONLY a raw JSON array of test case objects. Each object must have a 'name' and 'args' key. The 'args' keys MUST EXACTLY match the parameter names of the function. Generate 3 distinct test cases: happy path, empty input, and special characters/edge case. Use REAL, plausible values for file paths or inputs (e.g., 'README.md', '', '/tmp/nonexistent.txt'). Do NOT invent paths under /test/ that don't exist."},
            *context_messages,  # Inject past test cases here!
            {"role": "user", "content": f"Target Function Code:\n{python_code}"},
        ]
        try:
            stream = chat(model=TESTER_MODEL, messages=tester_messages,
                          stream=True, temperature=TESTER_TEMP)
            content_buffer = self._print_stream_and_get_content(
                stream, header="[Adversarial Tester]\n", end="\n")
            cleaned = content_buffer.strip().replace(
                "```json", "").replace("```", "").strip()
            parsed_args = json.loads(cleaned)
            if isinstance(parsed_args, list):
                self.emit(f"   [Adversarial Tests Generated]: {parsed_args}\n")
                save_aspect_memory("tester", self.session_id, json.dumps(parsed_args), embedding=get_embedding(tool_name))
                return parsed_args
        except Exception as e:
            # Better JSON extraction with fallback
            try:
                # Try to find JSON array in response
                match = re.search(r'\[[\s\S]*\]', content_buffer)
                if match:
                    parsed_args = json.loads(match.group())
                    if isinstance(parsed_args, list) and parsed_args:
                        # Validate each test case has required keys
                        valid_cases = []
                        for tc in parsed_args:
                            if isinstance(tc, dict) and "args" in tc:
                                valid_cases.append(tc)
                        if valid_cases:
                            return valid_cases
            except (json.JSONDecodeError, Exception):
                pass
            # Fallback: extract function signature and generate basic tests
            try:
                tree = ast.parse(python_code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        sig_params = {
                            arg.arg: "" for arg in node.args.args
                            if arg.arg != 'kwargs' and arg.arg != 'self'
                        }
                        return [{"name": "default", "args": sig_params}]
            except Exception:
                pass
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
                          tools=tools, think=THINK_ENABLED, stream=True, temperature=REASONING_TEMP)
        except urllib.error.HTTPError as e:
            if e.code == 400 and THINK_ENABLED:
                self.emit(
                    "   [Warning] Model rejected think=true. Retrying without thinking mode...\n")
                stream = chat(model=REASONING_MODEL, messages=messages,
                              tools=tools, think=False, stream=True, temperature=REASONING_TEMP)
            else:
                raise

        in_thinking = False
        in_content = False
        assembled_content = ""
        assembled_thinking = ""
        assembled_tool_calls = []
        text_buffer = ""
        suppress_output = False

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

                text_buffer += chunk_content
                assembled_content += chunk_content

                # State machine to filter out <think> tags from the TUI stream
                while text_buffer:
                    if suppress_output:
                        end_idx = text_buffer.find("</think>")
                        if end_idx != -1:
                            text_buffer = text_buffer[end_idx + 8:]
                            suppress_output = False
                        else:
                            break
                    else:
                        start_idx = text_buffer.find("<think>")
                        if start_idx != -1:
                            if start_idx > 0:
                                self.emit(text_buffer[:start_idx], end="")
                            text_buffer = text_buffer[start_idx + 7:]
                            suppress_output = True
                        else:
                            if len(text_buffer) > 7:
                                emit_len = len(text_buffer) - 7
                                self.emit(text_buffer[:emit_len], end="")
                                text_buffer = text_buffer[emit_len:]
                            else:
                                break

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

        if not suppress_output and text_buffer:
            self.emit(text_buffer, end="")

        self.emit("\n")

        # Final cleanup of any unclosed tags before saving to history
        if assembled_content:
            assembled_content = re.sub(
                r'<think>.*?</think>', '', assembled_content, flags=re.DOTALL).strip()
            if '<think>' in assembled_content:
                assembled_content = re.sub(
                    r'<think>.*', '', assembled_content, flags=re.DOTALL).strip()
            if '</think>' in assembled_content:
                assembled_content = re.sub(
                    r'</think>', '', assembled_content).strip()
            if not assembled_content:
                assembled_content = None

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
            patch_dir = PATCH_DIR  # <-- USE CONFIG PATH
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
            # --- KEY FIX: Builtin tools execute IN-PROCESS ---
            if is_builtin_tool(resolved_name):
                self.emit(
                    f"-> [In-Process Execution] Tool: '{resolved_name}' [Trace: {trace_id}]...\n")

                if resolved_name in ("create_tool", "update_tool"):
                    target_tool = args.get("tool_name", "dynamic_func")
                    code = args.get("python_code", "")
                    test_cases = self.generate_adversarial_test_cases(
                        target_tool, code)
                    args["test_inputs"] = test_cases

                result_str = tool_func(**args)
                if isinstance(result_str, str) and "Successfully" in result_str:
                    status = ResultStatus.SUCCESS
                else:
                    status = ResultStatus.VALIDATION_FAILURE
                return ToolResult(status, result_str)

            # --- Custom tools still go through sandbox ---
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

            # --- KEY FIX: Don't try to heal builtin tools ---
            is_dynamic_tool = (resolved_name not in META_TOOLS
                               and not is_builtin_tool(resolved_name))

            if not is_dynamic_tool or result.is_success or result.status == ResultStatus.NOT_FOUND or heal_attempts >= max_heals:
                return resolved_name, result

            # ... rest of self-heal logic for custom tools only ...

            heal_attempts += 1
            self.emit(
                f"\n!!! [Self-Heal Triggered] Tool '{resolved_name}' crashed. Attempting automated patch ({heal_attempts}/{max_heals})...\n")
            metrics.record_heal(False)

            try:
                tool_path = os.path.join(
                    CUSTOM_TOOLS_DIR, f"{resolved_name}.py")
                with open(tool_path, 'r', encoding='utf-8') as f:
                    broken_code = f.read()
            except Exception:
                return resolved_name, result

            # Inside execute_single_tool_call, right before calling the LLM to heal:
            raw_func_name, args = self._get_call_details(current_call)

            heal_messages = [
                {"role": "system", "content": "You are an autonomous debugging agent. A tool just crashed. Analyze the traceback, the code, and the EXACT arguments that caused the crash. Return ONLY the complete, fixed Python code for the function. Do not include explanations."},
                {"role": "user",
                    "content": f"Tool Name: {resolved_name}\n\nArguments that caused the crash:\n{json.dumps(args, default=str)}\n\nTraceback/Error:\n{result.value}\n\nBroken Code:\n{broken_code}"}
            ]

            try:
                self.emit("   [Self-Heal] Generating fixed code:\n", end="")
                stream = chat(model=META_PROMPT_MODEL, messages=heal_messages,
                              stream=True, temperature=META_PROMPT_TEMP)
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
                "tool_name": resolved_name,
                "python_code": fixed_code,
                # Force the sandbox to test the exact args that caused the crash!
                "test_inputs": [{"name": "crash_case", "args": args}]
            }}}
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

    def run_agent_loop(self, user_prompt: str, plan: str, tools_info: str, learnings: list = None) -> None:
        self.emit(
            f"\n=== User Turn: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        self.emit(f"User > {user_prompt}\n\n")

        dynamic_instructions = generate_dynamic_system_prompt(
            self, user_prompt, plan, tools_info)
        plan_str = f"\n\nExecution Plan:\n{plan}" if plan else ""

        learnings_str = ""
        if learnings:
            learnings_str = "\n\nRelevant Past Learnings:\n" + \
                "\n".join(f"- {l}" for l in learnings)

        # Fetch accumulated summaries
        summaries = retrieve_summaries(self.session_id)
        summaries_str = ""
        if summaries:
            summaries_str = "\n\nAccumulated Progress Summaries:\n" + \
                "\n".join(f"- {s}" for s in summaries)

        combined_system_prompt = {
            "role": "system",
            "content": f"{BASE_SYSTEM_PROMPT}{summaries_str}{learnings_str}\n\nTask Specific Directives:\n{dynamic_instructions}{plan_str}",
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

            # Memory Condensation
            recent_history, summary = condense_history(
                self.history, model_name=SUMMARY_MODEL, log_stream_func=self.on_stream)

            if summary:
                save_summary(self.session_id, summary)
                # Update system prompt to include the newly saved summary
                summaries_str += f"\n- {summary}"
                combined_system_prompt["content"] = f"{BASE_SYSTEM_PROMPT}{summaries_str}{learnings_str}\n\nTask Specific Directives:\n{dynamic_instructions}{plan_str}"

                if recent_history and recent_history[0].get("role") == "system":
                    recent_history[0] = combined_system_prompt
                else:
                    recent_history.insert(0, combined_system_prompt)

                self.history = recent_history

            active_tools = get_relevant_tools(user_prompt, top_k=5)
            assistant_message = self.ask_model_stream(
                self.history, active_tools)

            # --- KEY FIX: Robust Fallback Parser for local models ---
            if not assistant_message.tool_calls and assistant_message.content:
                content = assistant_message.content
                # Strip <think> tags if the model outputted them as text
                clean_content = re.sub(
                    r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

                tool_name = None
                tool_args = {}

                # 1. Try to find a standard JSON tool call (e.g., {"name": "tool", "arguments": {}})
                match = re.search(
                    r'\{[\s\S]*?"name"[\s\S]*?"arguments"[\s\S]*?\}', clean_content)
                if match:
                    try:
                        parsed = json.loads(match.group(0))
                        if "name" in parsed:
                            tool_name = parsed["name"]
                            tool_args = parsed.get("arguments", {})
                            if isinstance(tool_args, str):
                                tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        pass

                # 2. Try to find a generic JSON tool call (e.g., {"tool": "list_directory", "args": {}})
                if not tool_name:
                    match = re.search(
                        r'\{[\s\S]*?"tool"[\s\S]*?\}', clean_content)
                    if match:
                        try:
                            parsed = json.loads(match.group(0))
                            if "tool" in parsed:
                                tool_name = parsed["tool"]
                                tool_args = parsed.get("args", {})
                        except json.JSONDecodeError:
                            pass

                # 3. Try to parse XML-like function calls (e.g., <function=tool_name>...<parameter=key>value</parameter>...</function>)
                if not tool_name:
                    match = re.search(
                        r'<(?:function|tool)=([a-zA-Z0-9_]+)>([\s\S]*?)</(?:function|tool)>', clean_content)
                    if match:
                        tool_name = match.group(1)
                        params_block = match.group(2)
                        # Extract parameters: <parameter=name>value</parameter> or <param=name>value</param>
                        param_matches = re.findall(
                            r'<(?:parameter|param)=([a-zA-Z0-9_]+)>([\s\S]*?)</(?:parameter|param)>', params_block)
                        for p_name, p_value in param_matches:
                            tool_args[p_name] = p_value.strip()

                # 4. Try to parse simple open tags (e.g., <list_directory>)
                if not tool_name:
                    # Find all potential XML tags
                    tags = re.findall(r'<([a-zA-Z0-9_]+)>', clean_content)
                    for tag in tags:
                        # Check if this tag is actually a registered tool
                        with REGISTRY_LOCK:
                            if tag in TOOL_REGISTRY:
                                tool_name = tag
                                break

                # 5. Try to parse Python code blocks (e.g., ```python\ntool_name(args)\n```)
                if not tool_name:
                    match = re.search(
                        r'```(?:python)?\s*([\s\S]*?)\s*```', clean_content)
                    code_to_parse = match.group(1) if match else clean_content
                    try:
                        tree = ast.parse(code_to_parse)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                                func = node.value.func
                                if isinstance(func, ast.Name):
                                    tool_name = func.id
                                    for kw in node.value.keywords:
                                        try:
                                            tool_args[kw.arg] = ast.literal_eval(
                                                kw.value)
                                        except:
                                            tool_args[kw.arg] = ""
                                    break
                    except SyntaxError:
                        pass
                    except Exception:
                        pass

                # If we found a tool, construct the tool_call object
                if tool_name:
                    self.emit(
                        f"   [Fallback Parser] Detected tool call '{tool_name}' in text response.\n")
                    assistant_message.tool_calls = [{
                        "id": "fallback_call",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": tool_args
                        }
                    }]
                    # Clear the text content so it doesn't confuse the history
                    assistant_message.content = None
            # ----------------------------------------------------------------

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

            failure_streak = {}
            for tool_name, result in results:
                if not result.is_success:
                    failure_streak[tool_name] = failure_streak.get(
                        tool_name, 0) + 1
                    if failure_streak[tool_name] >= 2:
                        self.emit(
                            f"\n[Safety] Tool '{tool_name}' failed {failure_streak[tool_name]} times. Forcing strategy change.\n")
                        self.history.append({
                            "role": "user",
                            "content": f"Tool '{tool_name}' has failed {failure_streak[tool_name]} consecutive times. Stop calling it. Try a different approach or explain to the user why the operation cannot be completed."
                        })
                else:
                    failure_streak[tool_name] = 0

            if iteration > 5 and not any(r.is_success for _, r in results):
                self.history.append({
                    "role": "user",
                    "content": "CRITICAL: Multiple tool calls have failed in recent iterations. Consider: (1) Check if you're calling a builtin tool that already exists, (2) Verify your arguments match the tool's expected parameters, (3) If truly stuck, explain the situation to the user instead of looping."
                })

            self.turn_count += 1
            save_checkpoint(self.session_id, self.turn_count, self.history)
        else:
            self.emit(
                f"[Safety] Reached max iterations ({MAX_AGENT_ITERATIONS}). Forcing stop.\n")
            self.history.append(
                {"role": "user", "content": "Forced stop: iteration budget exhausted."})

    def _is_goal_satisfied(self, goal_text: str) -> bool:
        summaries = retrieve_summaries(self.session_id)
        summaries_str = "\n".join(f"- {s}" for s in summaries) if summaries else "None"

        verification = [
            {"role": "system", "content": "You verify whether a stated goal has been accomplished based on the recent agent conversation AND the progress summaries. Return JSON: {\"satisfied\": true|false, \"reason\": \"...\"}."},
            {"role": "user",
                "content": f"Goal: {goal_text}\n\nProgress Summaries:\n{summaries_str}\n\nRecent conversation (last 8 turns):\n{json.dumps(self.history[-8:], default=str)}"}
        ]
        try:
            stream = chat(model=META_PROMPT_MODEL, messages=verification,
                          stream=True, temperature=META_PROMPT_TEMP)
            buf = self._print_stream_and_get_content(
                stream, header="[Goal Verification]\n", end="\n")
            data = _extract_json_from_text(buf)
            if data:
                return bool(data.get("satisfied", False))
        except Exception:
            pass
        return False

    def run_autonomous_cycle(self, allow_self_edit: bool = False) -> bool:
        from src.agent.autonomous import select_autonomous_task
        from src.memory.storage import mark_goal_attempted, mark_goal_completed, mark_goal_blocked, log_system_improvement

        self.autonomous_mode = True
        self.emit(
            "\n=== [Autonomous Mode] No user input. Selecting task... ===\n")

        state_text = " ".join([m.get("content", "")
                              for m in self.history[-4:] if m.get("content")])
        state_emb = get_embedding(state_text)

        task = select_autonomous_task(
            allow_self_edit=allow_self_edit, current_state_emb=state_emb, agent=self)

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
                plan, tools_info = generate_plan(
                    agent=self, user_prompt=prompt)
                learnings = retrieve_learnings(
                    gtext, query_emb=get_embedding(gtext))
                self.run_agent_loop(prompt, plan, tools_info, learnings)
                reflect_on_task(self, prompt, self.history, plan)

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
                plan, tools_info = generate_plan(agent=self, user_prompt=prompt)
                learnings = retrieve_learnings(
                    f"improve {fpath}", query_emb=get_embedding(f"improve {fpath}"))
                self.run_agent_loop(prompt, plan, tools_info, learnings)
                reflect_on_task(self, prompt, self.history, plan)
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
