import json
import os
import subprocess
import sys
import tempfile
from runtime.result import ToolResult, ResultStatus


def execute_tool_in_sandbox(
    tool_name: str,
    args: dict,
    timeout_seconds: int = 120,
    ephemeral: bool = True,
    work_dir: str = None,
) -> ToolResult:
    """Executes a dynamic tool in an isolated Python subprocess via STDIN IPC."""

    # Pass tool name safely via environment variable to prevent code injection
    runner_code = """
import json, sys, io, os, traceback
from contextlib import redirect_stdout

def trace_exceptions(exc_type, exc_value, tb):
    frames = []
    curr_tb = tb
    while curr_tb:
        frame = curr_tb.tb_frame
        local_vars = {
            k: str(v)[:150] for k, v in frame.f_locals.items() 
            if not k.startswith("__")
        }
        frames.append({
            "file": os.path.basename(frame.f_code.co_filename),
            "line": curr_tb.tb_lineno,
            "function": frame.f_code.co_name,
            "locals": local_vars
        })
        curr_tb = curr_tb.tb_next

    error_payload = {
        "error_type": exc_type.__name__,
        "message": str(exc_value),
        "execution_trace": frames,
        "formatted_tb": traceback.format_exc()
    }
    print(json.dumps({"error": error_payload}))
    sys.exit(1)

sys.excepthook = trace_exceptions

try:
    tool_name = os.environ.get("SANDBOX_TOOL_NAME")
    if not tool_name:
        raise ValueError("SANDBOX_TOOL_NAME environment variable not set.")

    import_buffer = io.StringIO()
    with redirect_stdout(import_buffer):
        module = __import__(f"custom_tools.{tool_name}", fromlist=[tool_name])
        func = getattr(module, tool_name)

    raw_input = sys.stdin.read()
    args = json.loads(raw_input) if raw_input.strip() else {}

    exec_buffer = io.StringIO()
    with redirect_stdout(exec_buffer):
        result = func(**args)

    tool_logs = import_buffer.getvalue().strip()
    if tool_logs:
        tool_logs += "\\n"
    tool_logs += exec_buffer.getvalue().strip()

    if isinstance(result, bool):
        result = "Success" if result else "Completed but returned False"

    out_payload = {"result": str(result)}
    if tool_logs:
        out_payload["logs"] = tool_logs

    print(json.dumps(out_payload))

except Exception as e:
    trace_exceptions(type(e), e, e.__traceback__)
"""

    payload = json.dumps(args)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["SANDBOX_TOOL_NAME"] = tool_name  # Safely pass the tool name

    cwd = os.getcwd()
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{cwd}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = cwd

    def _run_in_dir(target_dir: str) -> ToolResult:
        proc = subprocess.run(
            [sys.executable, "-c", runner_code],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
            cwd=target_dir,
        )

        if not proc.stdout.strip():
            stderr_msg = proc.stderr.strip() or "No output returned."
            return ToolResult(ResultStatus.RUNTIME_FAILURE, f"Execution Failure (Exit {proc.returncode}): {stderr_msg}")

        try:
            output = json.loads(proc.stdout.strip())
        except json.JSONDecodeError:
            return ToolResult(ResultStatus.RUNTIME_FAILURE, f"IPC Error: Unparseable JSON output.\nRaw Stdout: {proc.stdout}\nRaw Stderr: {proc.stderr}")

        if "error" in output:
            err = output["error"]
            if isinstance(err, dict):
                trace_lines = []
                for frame in err.get("execution_trace", []):
                    trace_lines.append(
                        f"  - Line {frame['line']} in `{frame['function']}` | Locals: {frame['locals']}")
                trace_str = "\n".join(trace_lines)
                return ToolResult(
                    status=ResultStatus.RUNTIME_FAILURE,
                    value=f"Runtime Failure [{err.get('error_type')}]: {err.get('message')}\nExecution Frame Traces:\n{trace_str}\nTraceback Details:\n{err.get('formatted_tb', '')}",
                    traceback=err.get('formatted_tb', ''),
                    execution_trace=err.get('execution_trace', [])
                )
            return ToolResult(ResultStatus.RUNTIME_FAILURE, f"Tool Error: {err}")

        result_str = output.get("result", "")
        if "logs" in output and output["logs"]:
            result_str += f"\n[Tool Execution Logs]:\n{output['logs']}"

        return ToolResult(ResultStatus.SUCCESS, result_str)

    try:
        if work_dir is not None:
            return _run_in_dir(work_dir)
        elif ephemeral:
            with tempfile.TemporaryDirectory() as temp_dir:
                return _run_in_dir(temp_dir)
        else:
            return _run_in_dir(cwd)
    except subprocess.TimeoutExpired:
        return ToolResult(ResultStatus.TIMEOUT, f"Error: Tool '{tool_name}' timed out after {timeout_seconds} seconds.")
    except Exception as e:
        return ToolResult(ResultStatus.RUNTIME_FAILURE, f"Sandbox Infrastructure Failure: {str(e)}")
