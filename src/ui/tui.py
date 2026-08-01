# tui.py
import curses
import queue
import time


class StdoutRedirector:
    """Redirects standard print statements to the TUI queue."""

    def __init__(self, tui):
        self.tui = tui

    def write(self, text):
        # IMPROVEMENT: Only drop completely empty strings, allow newlines/whitespace
        if text:
            self.tui.render_q.put(text)

    def flush(self):
        pass


class CursesTUI:
    def __init__(self):
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)
        curses.curs_set(0)  # Hide cursor initially
        self.render_q = queue.Queue()
        self.memory_q = queue.Queue()
        self._update_layout()

    def _update_layout(self):
        self.height, self.width = self.stdscr.getmaxyx()

        # Input window at the bottom (3 lines)
        self.input_height = 3
        # Memory window above input (7 lines)
        self.memory_height = 7
        # Chat window takes the rest
        self.chat_height = self.height - self.input_height - self.memory_height
        if self.chat_height < 1:
            self.chat_height = 1

        # Chat Window
        self.chat_win = curses.newwin(self.chat_height, self.width, 0, 0)
        self.chat_win.scrollok(True)
        self.chat_win.refresh()

        # Memory Window
        self.memory_win = curses.newwin(
            self.memory_height, self.width, self.chat_height, 0)
        self.memory_win.scrollok(True)
        self.memory_win.box()
        self.memory_win.addstr(0, 1, " Memory & Summaries ")
        self.memory_win.refresh()

        # Input Window
        self.input_win = curses.newwin(
            self.input_height, self.width, self.chat_height + self.memory_height, 0)
        self.input_win.box()
        self.input_win.refresh()

    def stream_handler(self, text: str):
        """Callback for agent.emit(). Routes text to appropriate window."""
        # Route summarizer thoughts and reflections to the memory window
        if "Memory Condenser" in text or "[Summary]" in text or "Learnings" in text or "Reflection" in text:
            self.memory_q.put(text)
        else:
            self.render_q.put(text)

    def input_handler(self, prompt: str) -> str:
        """Synchronous input for HITL approvals."""
        self.render_loop()
        self.input_win.clear()
        self.input_win.box()
        self.input_win.addstr(1, 1, prompt[:self.width-2])
        self.input_win.clrtoeol()
        self.input_win.refresh()

        curses.echo()
        curses.curs_set(1)
        try:
            # Read input starting after the prompt
            raw = self.input_win.getstr(1, len(prompt) + 1).decode('utf-8')
        except:
            raw = ""
        curses.curs_set(0)
        curses.noecho()

        self.input_win.clear()
        self.input_win.box()
        self.input_win.refresh()
        return raw

    def get_input_async(self, timeout: int) -> str:
        """Non-blocking input that polls for timeout, renders background logs, and captures typing."""
        self.render_loop()
        self.input_win.clear()
        self.input_win.box()
        self.input_win.addstr(1, 1, "User > ")
        self.input_win.clrtoeol()
        self.input_win.refresh()

        curses.curs_set(1)
        self.input_win.timeout(100)  # Poll every 100ms
        input_str = ""
        start_time = time.time()

        while time.time() - start_time < timeout:
            self.render_loop()  # Keep UI updating with agent logs

            ch = self.input_win.getch()
            if ch == -1:
                continue
            if ch == curses.KEY_RESIZE:
                self._update_layout()
                self.input_win.addstr(1, 1, "User > " + input_str)
            elif ch == curses.KEY_ENTER or ch == 10:  # Enter
                break
            elif ch == curses.KEY_BACKSPACE or ch == 127:
                if len(input_str) > 0:
                    input_str = input_str[:-1]
            elif 32 <= ch <= 126:  # Printable ASCII
                if len(input_str) < self.width - 10:
                    input_str += chr(ch)

            # Redraw input line
            self.input_win.move(1, 7)
            self.input_win.clrtoeol()
            self.input_win.addstr(input_str)
            self.input_win.refresh()

        curses.curs_set(0)
        self.input_win.timeout(-1)  # Reset to blocking

        if time.time() - start_time >= timeout and input_str == "":
            return None

        self.input_win.clear()
        self.input_win.box()
        self.input_win.refresh()
        return input_str

    def render_loop(self):
        """Drain the render queues and update windows."""
        # 1. Update Chat Window
        while True:
            try:
                text = self.render_q.get_nowait()
                self.chat_win.addstr(text)
            except queue.Empty:
                break
        self.chat_win.refresh()

        # 2. Update Memory Window
        while True:
            try:
                text = self.memory_q.get_nowait()
                # Scroll the memory window up by 1 line, keeping the box intact
                self.memory_win.scroll(1)
                self.memory_win.move(self.memory_height - 2, 1)
                self.memory_win.clrtoeol()
                self.memory_win.addstr(
                    self.memory_height - 2, 1, text[:self.width-2])
                self.memory_win.box()
                self.memory_win.addstr(0, 1, " Memory & Summaries ")
            except queue.Empty:
                break
        self.memory_win.refresh()

        # 3. Refresh Input Window
        self.input_win.box()
        self.input_win.refresh()

    def cleanup(self):
        curses.nocbreak()
        self.stdscr.keypad(False)
        curses.echo()
        curses.curs_set(1)
        curses.endwin()
