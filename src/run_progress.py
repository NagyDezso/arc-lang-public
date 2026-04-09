"""Console run progress: tasks, elapsed/ETA, and aggregated LLM token usage."""

from __future__ import annotations

import contextvars
import sys
import threading
import time

_CTX: contextvars.ContextVar[RunProgress | None] = contextvars.ContextVar(
    "arc_run_progress", default=None
)


def attach_run_progress(progress: RunProgress) -> contextvars.Token:
    return _CTX.set(progress)


def detach_run_progress(token: contextvars.Token) -> None:
    _CTX.reset(token)


def record_llm_usage(input_tokens: int = 0, output_tokens: int = 0) -> None:
    p = _CTX.get()
    if p is not None and (input_tokens or output_tokens):
        p.record_tokens(input_tokens, output_tokens)


def notify_task_finished() -> None:
    p = _CTX.get()
    if p is not None:
        p.task_finished()


def _fmt_duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # NaN
        return "—"
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 10_000:
        return f"{n / 1000:.1f}k"
    if n >= 1000:
        return f"{n / 1000:.2f}k"
    return str(n)


def _progress_bar(done: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + "?" * width + "]"
    filled = min(width, int(width * done / total))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


class RunProgress:
    """Thread-safe counters for a multi-task async run (printed on one status line)."""

    def __init__(self, total_tasks: int) -> None:
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.run_start = time.monotonic()
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()
        self._last_render = 0.0
        self._min_token_interval = 1.0

    def record_tokens(self, inp: int, out: int) -> None:
        now = time.monotonic()
        with self._lock:
            self.input_tokens += inp
            self.output_tokens += out
            should_render = (now - self._last_render) >= self._min_token_interval
            if should_render:
                self._last_render = now
                line = self._format_line_unlocked()
        if should_render:
            self._write_status_line(line)

    def task_finished(self) -> None:
        with self._lock:
            self.completed_tasks += 1
            self._last_render = time.monotonic()
            line = self._format_line_unlocked()
        self._write_status_line(line)

    def render_now(self) -> None:
        with self._lock:
            line = self._format_line_unlocked()
        self._write_status_line(line)

    def finish_line(self) -> None:
        """End the in-place status line so later prints start on a fresh row."""
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _format_line_unlocked(self) -> str:
        elapsed = time.monotonic() - self.run_start
        done = self.completed_tasks
        total = self.total_tasks
        bar = _progress_bar(done, total)
        if done > 0 and total > done:
            eta_s = (elapsed / done) * (total - done)
            eta_str = _fmt_duration(eta_s)
        elif done >= total and total > 0:
            eta_str = "0s"
        else:
            eta_str = "—"

        in_t = _fmt_tokens(self.input_tokens)
        out_t = _fmt_tokens(self.output_tokens)
        return (
            f"\r{bar} {done}/{total} tasks  "
            f"elapsed {_fmt_duration(elapsed)}  "
            f"ETA ~{eta_str}  "
            f"tokens in {in_t}  out {out_t}"
        )

    @staticmethod
    def _write_status_line(line: str) -> None:
        # Pad to clear remnants of a longer previous line
        pad = max(0, 120 - len(line))
        sys.stdout.write(line + " " * pad)
        sys.stdout.flush()
