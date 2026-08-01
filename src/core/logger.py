import logging
import json


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "session_id": getattr(record, "session_id", None),
            "tool": getattr(record, "tool", None),
            "turn": getattr(record, "turn", None),
            "trace_id": getattr(record, "trace_id", None),
        }
        return json.dumps(log_record)


logger = logging.getLogger("environmentbots")
logger.setLevel(logging.INFO)
