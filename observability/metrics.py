# observability/metrics.py
from collections import defaultdict
import time


class Metrics:
    def __init__(self):
        self.tool_calls = defaultdict(int)
        self.tool_failures = defaultdict(int)
        self.latency = defaultdict(list)
        self.tokens_in = 0
        self.tokens_out = 0
        self.heal_triggers = 0
        self.heal_successes = 0
        self.condensations = 0

    def record_tool(self, name: str, duration_s: float, success: bool):
        self.tool_calls[name] += 1
        if not success:
            self.tool_failures[name] += 1
        self.latency[name].append(duration_s)

    def record_heal(self, success: bool):
        self.heal_triggers += 1
        if success:
            self.heal_successes += 1

    def summary(self) -> dict:
        return {
            "tools": {
                k: {
                    "calls": v,
                    "failures": self.tool_failures[k],
                    "p50_lat": sorted(self.latency[k])[len(self.latency[k])//2] if self.latency[k] else 0
                } for k, v in self.tool_calls.items()
            },
            "tokens": {"in": self.tokens_in, "out": self.tokens_out},
            "heals": {"triggers": self.heal_triggers, "successes": self.heal_successes},
            "condensations": self.condensations
        }


metrics = Metrics()
