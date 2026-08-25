"""A small, dependency-free live progress line.

Long phases (self-play until N positions, training for N steps) print a single
line that refreshes in place on a terminal, and degrades to an occasional full
log line when the output is redirected to a file or a pipe.

It cooperates with ``logging``: :func:`attach_logging` wraps the existing
handlers so a log record erases the bar, prints, and lets the bar redraw --
otherwise the two would overwrite each other.
"""

from __future__ import annotations

import logging
import shutil
import sys
import threading
import time
from typing import Optional, TextIO


def format_duration(seconds: float) -> str:
    """``93`` -> ``1m33s``, ``4000`` -> ``1h06m``."""
    if seconds != seconds or seconds < 0 or seconds == float("inf"):
        return "?"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def format_eta(done: float, total: float, elapsed: float) -> str:
    """Remaining time from a linear extrapolation of the rate so far."""
    if total <= 0 or done <= 0 or elapsed <= 0:
        return "?"
    if done >= total:
        return "0s"
    return format_duration(elapsed * (total - done) / done)


def bar(fraction: float, width: int = 18) -> str:
    fraction = min(1.0, max(0.0, fraction))
    filled = int(round(fraction * width))
    return "#" * filled + "-" * (width - filled)


class Progress:
    """One refreshing status line.

    ``enabled`` defaults to "the stream is a terminal". When it is off the same
    text is emitted through ``logging`` at most every ``log_interval`` seconds,
    so redirected runs stay readable without a flood of lines.
    """

    def __init__(self, stream: Optional[TextIO] = None,
                 enabled: Optional[bool] = None,
                 min_interval: float = 0.25,
                 log_interval: float = 30.0,
                 logger: Optional[logging.Logger] = None):
        self.stream = stream if stream is not None else sys.stderr
        if enabled is None:
            enabled = bool(getattr(self.stream, "isatty", lambda: False)())
        self.enabled = enabled
        self.min_interval = min_interval
        self.log_interval = log_interval
        self.logger = logger or logging.getLogger("mylc0.progress")
        self._lock = threading.RLock()
        self._current = ""
        self._drawn = False
        self._last_draw = 0.0
        self._last_log = 0.0

    # -- drawing -----------------------------------------------------------
    def set(self, text: str, force: bool = False) -> None:
        now = time.monotonic()
        with self._lock:
            self._current = text
            if self.enabled:
                if force or now - self._last_draw >= self.min_interval:
                    self._last_draw = now
                    self._draw()
            elif force or now - self._last_log >= self.log_interval:
                self._last_log = now
                self.logger.info("%s", text)

    def _draw(self) -> None:
        width = shutil.get_terminal_size((100, 25)).columns
        text = self._current
        if len(text) > width - 1:
            # Plain truncation: an ellipsis character would raise
            # UnicodeEncodeError on a console that is not UTF-8.
            text = text[:width - 1]
        self.stream.write("\r" + text.ljust(width - 1))
        self.stream.flush()
        self._drawn = True

    def clear(self) -> None:
        """Erase the line so something else can print cleanly."""
        with self._lock:
            if self.enabled and self._drawn:
                width = shutil.get_terminal_size((100, 25)).columns
                self.stream.write("\r" + " " * (width - 1) + "\r")
                self.stream.flush()
                self._drawn = False

    def redraw(self) -> None:
        with self._lock:
            if self.enabled and self._current and not self._drawn:
                self._draw()

    def close(self, final: Optional[str] = None) -> None:
        """Finish the line; ``final`` replaces it and is kept on screen."""
        with self._lock:
            if final is not None:
                self._current = final
            if self.enabled and self._current:
                self._draw()
                self.stream.write("\n")
                self.stream.flush()
                self._drawn = False
            elif final is not None:
                self.logger.info("%s", final)
            self._current = ""


class _BarAwareHandler(logging.Handler):
    """Wraps a handler so log records never land on top of the bar."""

    def __init__(self, inner: logging.Handler, progress: Progress):
        super().__init__(level=inner.level)
        self.inner = inner
        self.progress = progress

    def emit(self, record: logging.LogRecord) -> None:
        self.progress.clear()
        try:
            self.inner.emit(record)
        finally:
            self.progress.redraw()

    def setFormatter(self, fmt) -> None:  # noqa: N802 (logging API)
        self.inner.setFormatter(fmt)


def attach_logging(progress: Progress,
                   logger: Optional[logging.Logger] = None) -> None:
    """Make ``logging`` play nicely with a live bar on the same terminal."""
    if not progress.enabled:
        return
    logger = logger or logging.getLogger()
    for i, handler in enumerate(list(logger.handlers)):
        if isinstance(handler, _BarAwareHandler):
            continue
        logger.handlers[i] = _BarAwareHandler(handler, progress)
