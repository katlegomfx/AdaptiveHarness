# src/ui/tui.py
import curses
import queue
import time


class StdoutRedirector:
    """Redirects standard print statements to the TUI queue."""

    def __init__(self, tui):
        self.tui = tui

    def write(self, text):
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

        # Initialize Colors
        curses.start_color()
        curses.use_default_colors()
        # 1: Cyan (Planning/Headers), 2: Green (Execution), 3: Yellow (Warnings/Fallback),
        # 4: Red (Errors/Safety), 5: Magenta (Memory/Reflection), 6: Blue (Tool Results)
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_RED, -1)
        curses.init_pair(5, curses.COLOR_MAGENTA, -1)
        curses.init_pair(6, curses.COLOR_BLUE, -1)
        curses.init_pair(7, curses.COLOR_WHITE,
                         curses.COLOR_BLUE)  # Input border

        self.render_q = queue.Queue()
        self.memory_q = queue.Queue()
        self.hitl_q = queue.Queue()
        self.hitl_response_q = queue.Queue()
        self._input_buffer = ""

        self._update_layout()

    def _update_layout(self):
        self.height, self.width = self.stdscr.getmaxyx()
        self.input_height = 3
        self.memory_height = 7
        self.chat_height = self.height - self.input_height - self.memory_height
        if self.chat_height < 1:
            self.chat_height = 1

        self.chat_win = curses.newwin(self.chat_height, self.width, 0, 0)
        self.chat_win.scrollok(True)
        self.chat_win.refresh()

        self.memory_win = curses.newwin(
            self.memory_height, self.width, self.chat_height, 0)
        self.memory_win.scrollok(True)
        self.memory_win.box()
        self.memory_win.addstr(0, 1, " Memory & Summaries ",
                               curses.color_pair(5) | curses.A_BOLD)
        self.memory_win.refresh()

        self.input_win = curses.newwin(
            self.input_height, self.width, self.chat_height + self.memory_height, 0)
        self.input_win.box()
        self.input_win.refresh()

    def stream_handler(self, text: str):
        """Callback for agent.emit(). Routes text to appropriate window safely."""
        if "Memory Condenser" in text or "[Summary]" in text or "Learnings" in text or "Reflection" in text or "Goal Extractor" in text:
            self.memory_q.put(text)
        else:
            self.render_q.put(text)

    def input_handler(self, prompt: str) -> str:
        """Synchronous input for HITL approvals (called from background thread)."""
        self.hitl_q.put(prompt)
        # Blocks background thread until main thread answers
        return self.hitl_response_q.get()

    def _get_color_for_line(self, line: str) -> int:
        """Determines the color code based on the content of the log line."""
        # Memory/Reflection phases (Magenta)
        if any(kw in line for kw in ["[Reflection]", "[Goal Extractor]", "[Memory Condenser", "[Summary]"]):
            return curses.color_pair(5)
        # Planning and Directives (Cyan)
        elif any(kw in line for kw in ["-> [Pass", "Task Profile", "[Plan]", "[Meta-Prompt Directives]", "Pass 1 Complete"]):
            return curses.color_pair(1) | curses.A_BOLD
        # Execution and Tools (Green)
        elif any(kw in line for kw in ["[In-Process Execution]", "[Sandbox Execution]", "Model requested tool", "[Adversarial Tester]"]):
            return curses.color_pair(2) | curses.A_BOLD
        # Tool Results (Blue)
        elif line.startswith("   Result ["):
            return curses.color_pair(6)
        # Warnings and Fallbacks (Yellow)
        elif any(kw in line for kw in ["[Fallback Parser]", "[Warning]", "Skipped", "Defaulting"]):
            return curses.color_pair(3)
        # Errors and Safety (Red)
        elif any(kw in line for kw in ["[Safety]", "Error", "Failed", "Runtime Failure", "Reverted", "CRITICAL ERROR"]):
            return curses.color_pair(4) | curses.A_BOLD
        # Default (White)
        return curses.color_pair(0)

    def _add_colored_text(self, win, text: str):
        """Splits text by newlines and applies the appropriate color to each line."""
        for line in text.splitlines(keepends=True):
            color = self._get_color_for_line(line)
            try:
                win.addstr(line, color)
            except curses.error:
                # Ignore error when writing to the bottom-right corner of the screen
                pass

    def render_loop(self):
        """Drains the render queues and updates windows. Called only by main thread."""
        # 1. Update Chat Window
        while True:
            try:
                text = self.render_q.get_nowait()
                self._add_colored_text(self.chat_win, text)
            except queue.Empty:
                break
        self.chat_win.refresh()

        # 2. Update Memory Window
        while True:
            try:
                text = self.memory_q.get_nowait()
                self.memory_win.scroll(1)
                self.memory_win.move(self.memory_height - 2, 1)
                self.memory_win.clrtoeol()

                # Memory window text is always Magenta
                color = self._get_color_for_line(text)
                try:
                    self.memory_win.addstr(
                        self.memory_height - 2, 1, text[:self.width-2], color)
                except curses.error:
                    pass

                self.memory_win.box()
                self.memory_win.addstr(
                    0, 1, " Memory & Summaries ", curses.color_pair(5) | curses.A_BOLD)
            except queue.Empty:
                break
        self.memory_win.refresh()

        # 3. Update Input Window
        self.input_win.box()
        self.input_win.addstr(
            0, 1, " Input ", curses.color_pair(7) | curses.A_BOLD)
        self.input_win.move(1, 1)
        self.input_win.clrtoeol()
        self.input_win.addstr(
            1, 1, "User > ", curses.color_pair(1) | curses.A_BOLD)
        self.input_win.addstr(self._input_buffer, curses.color_pair(0))
        self.input_win.refresh()

    def get_input_non_blocking(self) -> str | None:
        """Main loop function: checks for HITL, renders UI, and captures typing."""
        # 1. Check for HITL requests first
        try:
            hitl_prompt = self.hitl_q.get_nowait()
            self.render_loop()

            self.input_win.timeout(-1)  # Blocking input for HITL
            curses.echo()
            curses.curs_set(1)

            # Draw HITL prompt (usually red/yellow for attention)
            self.input_win.move(1, 1)
            self.input_win.clrtoeol()
            try:
                self.input_win.addstr(
                    1, 1, hitl_prompt[:self.width-2], curses.color_pair(3) | curses.A_BOLD)
            except curses.error:
                pass

            try:
                raw = self.input_win.getstr(
                    1, len(hitl_prompt) + 1).decode('utf-8')
            except:
                raw = ""

            curses.curs_set(0)
            curses.noecho()
            self.hitl_response_q.put(raw)
            self._input_buffer = ""
            return None
        except queue.Empty:
            pass

        # 2. Normal rendering and input polling
        self.render_loop()
        self.input_win.timeout(50)  # Poll every 50ms (non-blocking)
        ch = self.input_win.getch()

        if ch == -1:
            return None

        if ch == curses.KEY_RESIZE:
            self._update_layout()
        elif ch == curses.KEY_ENTER or ch == 10:  # Enter
            input_str = self._input_buffer
            self._input_buffer = ""
            return input_str
        elif ch == curses.KEY_BACKSPACE or ch == 127:
            if len(self._input_buffer) > 0:
                self._input_buffer = self._input_buffer[:-1]
        elif 32 <= ch <= 126:  # Printable ASCII
            if len(self._input_buffer) < self.width - 10:
                self._input_buffer += chr(ch)

        return None

    def cleanup(self):
        curses.nocbreak()
        self.stdscr.keypad(False)
        curses.echo()
        curses.curs_set(1)
        curses.endwin()
