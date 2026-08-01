from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any


class ResultStatus(Enum):
    SUCCESS = "success"
    RUNTIME_FAILURE = "runtime_failure"
    VALIDATION_FAILURE = "validation_failure"
    TIMEOUT = "timeout"
    NOT_FOUND = "not_found"


@dataclass
class ToolResult:
    status: ResultStatus
    value: str
    traceback: Optional[str] = None
    execution_trace: Optional[List[Dict[str, Any]]] = None

    @property
    def is_success(self) -> bool:
        return self.status == ResultStatus.SUCCESS
