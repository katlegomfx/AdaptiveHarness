# src/core/types.py
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    thinking: Optional[str] = None
