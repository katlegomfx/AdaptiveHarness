# tui.py
import curses
import queue
import time


class StdoutRedirector:
    """Redirects standard print statements to the TUI queue."""

    def __init__(self, tui):
        self.tui = tui

    def write(self, text):
        if text.strip():
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
        self._update_layout()

    def _update_layout(self):
        self.height, self.width = self.stdscr.getmaxyx()
        self.chat_height = self.height - 4
        if self.chat_height < 1:
            self.chat_height = 1

        self.chat_win = curses.newwin(self.chat_height, self.width, 0, 0)
        self.chat_win.scrollok(True)
        self.chat_win.refresh()

        self.input_win = curses.newwin(4, self.width, self.chat_height, 0)
        self.input_win.box()
        self.input_win.refresh()

    def stream_handler(self, text: str):
        """Callback for agent.emit()"""
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
            raw = self.input_win.getstr(2, 1).decode('utf-8')
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
        """Drain the render queue and update the chat window."""
        while True:
            try:
                text = self.render_q.get_nowait()
                self.chat_win.addstr(text)
            except queue.Empty:
                break
        self.chat_win.refresh()
        self.input_win.box()
        self.input_win.refresh()

    def cleanup(self):
        curses.nocbreak()
        self.stdscr.keypad(False)
        curses.echo()
        curses.curs_set(1)
        curses.endwin()
