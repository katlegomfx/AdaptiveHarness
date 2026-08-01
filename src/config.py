# src/config.py
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Paths - Point to project root (parent of src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "agent_state.db")
BACKUP_DIR = os.path.join(DATA_DIR, ".safety_net_backups")
CUSTOM_TOOLS_DIR = os.path.join(DATA_DIR, "custom_tools")
LOG_DIR = os.path.join(DATA_DIR, "logs")
LOG_FILE_PATH = os.path.join(LOG_DIR, "agent_execution.log")
JSON_LOG_PATH = os.path.join(LOG_DIR, "agent.jsonl")
IMPROVEMENT_GUIDE_PATH = os.path.join(LOG_DIR, "system_improvement_guide.md")
PATCH_DIR = os.path.join(DATA_DIR, "pending_patches")
LLMS_DIR = os.path.join(BASE_DIR, "llms")

# Ensure directories exist
for d in (DATA_DIR, LOG_DIR, CUSTOM_TOOLS_DIR, BACKUP_DIR, PATCH_DIR, LLMS_DIR):
    os.makedirs(d, exist_ok=True)

# KEY FIX: Add DATA_DIR to sys.path so `import custom_tools` works in sandbox
if DATA_DIR not in sys.path:
    sys.path.insert(0, DATA_DIR)

# Models
REASONING_MODEL = os.environ.get("REASONING_MODEL", "ornith")
META_PROMPT_MODEL = os.environ.get("META_PROMPT_MODEL", "ornith")
TESTER_MODEL = os.environ.get("TESTER_MODEL", "ornith")
SUMMARY_MODEL = os.environ.get("SUMMARY_MODEL", "ornith")

# Temperatures
REASONING_TEMP = float(os.environ.get("REASONING_TEMP", "0.7"))
META_PROMPT_TEMP = float(os.environ.get("META_PROMPT_TEMP", "0.2"))
TESTER_TEMP = float(os.environ.get("TESTER_TEMP", "0.3"))
SUMMARY_TEMP = float(os.environ.get("SUMMARY_TEMP", "0.3"))
GOAL_TEMP = float(os.environ.get("GOAL_TEMP", "0.2"))

THINK_ENABLED = os.environ.get(
    "THINK_ENABLED", "false").strip().lower() in ("true", "1", "yes", "on")

IDLE_TIMEOUT_SECONDS = 240
AUTONOMOUS_COOLDOWN_SECONDS = 120
MAX_AUTONOMOUS_CYCLES_IN_A_ROW = 12
