# async_input.py
"""Non-blocking stdin reader. Lines are pushed into a queue by a daemon thread."""
import queue
import sys
import threading
from typing import Optional


class AsyncInputReader:
    def __init__(self) -> None:
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._stop = threading.Event()
        self._eof = False  # Add EOF flag
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                line = sys.stdin.readline()
            except (OSError, ValueError):
                self._q.put(None)
                self._eof = True
                break
            if line == "":          # EOF
                self._q.put(None)
                self._eof = True
                break
            self._q.put(line.rstrip("\n"))

    def is_eof(self) -> bool:
        return self._eof

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                line = sys.stdin.readline()
            except (OSError, ValueError):
                self._q.put(None)
                break
            if line == "":          # EOF
                self._q.put(None)
                break
            self._q.put(line.rstrip("\n"))

    def get_input(self, timeout: float) -> Optional[str]:
        """Return a line within `timeout` seconds, or None on timeout / EOF."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop.set()
