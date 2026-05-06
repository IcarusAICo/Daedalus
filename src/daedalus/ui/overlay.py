"""Always-on-top status overlay shown while the executor is running.

The overlay tells the user "the agent is in control right now" and gives them
a one-click / one-hotkey abort.

Design constraints:

- Phase 0 wants minimum new dependencies. We use Tk from the standard library.
- Tk's grab on global hotkeys is window-scoped; truly global hotkeys would need
  ``pynput`` or ``keyboard`` (root on Linux). For Phase 0 the abort button on
  the overlay window plus ``Ctrl-C`` in the controlling terminal are enough.
- We must not crash the agent if no display is available (CI, SSH, headless).
  In that case ``make_overlay`` returns a :class:`NullOverlay` that just logs.

The overlay runs on a worker thread; the abort signal is delivered via a
``threading.Event`` shared with the :class:`ExecutionContext`.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from abc import ABC, abstractmethod
from typing import Protocol

log = logging.getLogger(__name__)


class Overlay(Protocol):
    abort_event: threading.Event

    def start(self) -> None: ...
    def update_status(self, text: str) -> None: ...
    def stop(self) -> None: ...


class _BaseOverlay(ABC):
    def __init__(self, abort_event: threading.Event | None = None) -> None:
        self.abort_event = abort_event or threading.Event()

    @abstractmethod
    def start(self) -> None: ...
    @abstractmethod
    def update_status(self, text: str) -> None: ...
    @abstractmethod
    def stop(self) -> None: ...


class NullOverlay(_BaseOverlay):
    """Fallback when no display is available. Prints status to the log."""

    def start(self) -> None:
        log.info("overlay disabled (headless); abort via Ctrl-C in the terminal")

    def update_status(self, text: str) -> None:
        log.info("agent: %s", text)

    def stop(self) -> None:
        pass


class TkOverlay(_BaseOverlay):
    """Tk-based status banner. Spawns its own thread and runs Tk's mainloop there."""

    def __init__(self, task_name: str, abort_event: threading.Event | None = None) -> None:
        super().__init__(abort_event)
        self._task_name = task_name
        self._status_text = "starting..."
        self._thread: threading.Thread | None = None
        self._root_ready = threading.Event()
        self._root = None  # type: ignore[var-annotated]
        self._status_var = None  # type: ignore[var-annotated]
        self._stop_requested = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="daedalus-overlay", daemon=True)
        self._thread.start()
        # Give Tk a moment to come up; never block forever.
        self._root_ready.wait(timeout=2.0)

    def update_status(self, text: str) -> None:
        self._status_text = text
        if self._root is not None and self._status_var is not None:
            with contextlib.suppress(Exception):
                self._root.after(0, self._status_var.set, text)

    def stop(self) -> None:
        self._stop_requested = True
        if self._root is not None:
            with contextlib.suppress(Exception):
                self._root.after(0, self._root.destroy)
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # --------------------------------------------------------------------

    def _on_abort(self) -> None:
        log.warning("overlay: user requested abort")
        self.abort_event.set()
        self.update_status("ABORT REQUESTED — finishing current step...")

    def _run(self) -> None:
        try:
            import tkinter as tk
        except Exception as exc:  # pragma: no cover - depends on system tk
            log.warning("Tk unavailable (%s); overlay disabled", exc)
            self._root_ready.set()
            return

        try:
            root = tk.Tk()
            root.title("Daedalus: AGENT ACTIVE")
            root.attributes("-topmost", True)
            with contextlib.suppress(tk.TclError):
                root.attributes("-alpha", 0.92)
            root.geometry("+20+20")
            root.configure(bg="#7a0010")

            tk.Label(
                root,
                text="AGENT ACTIVE",
                fg="white",
                bg="#7a0010",
                font=("TkDefaultFont", 14, "bold"),
                padx=12,
                pady=4,
            ).pack(fill="x")

            tk.Label(
                root,
                text=self._task_name,
                fg="#ffd0d0",
                bg="#7a0010",
                font=("TkDefaultFont", 10),
                padx=12,
            ).pack(fill="x")

            self._status_var = tk.StringVar(value=self._status_text)
            tk.Label(
                root,
                textvariable=self._status_var,
                fg="white",
                bg="#400008",
                font=("TkDefaultFont", 10),
                padx=12,
                pady=6,
                anchor="w",
                wraplength=360,
                justify="left",
            ).pack(fill="x")

            tk.Button(
                root,
                text=(
                    "ABORT  (Ctrl+Shift+Esc)"
                    if getattr(self, "_global_listener", None)
                    else "ABORT  (click here or Ctrl-C in terminal)"
                ),
                command=self._on_abort,
                bg="#ffe0e0",
                fg="#7a0010",
                activebackground="#ffb0b0",
                relief="raised",
                padx=10,
                pady=4,
            ).pack(fill="x", padx=8, pady=(4, 8))

            # Window-scoped hotkey
            root.bind_all("<Control-Shift-Escape>", lambda _e: self._on_abort())

            self._root = root
            self._root_ready.set()
            root.mainloop()
        except Exception as exc:  # pragma: no cover
            log.warning("overlay crashed: %s", exc)
            self._root_ready.set()


class GlobalHotkeyOverlay(TkOverlay):
    """TkOverlay enhanced with a system-wide hotkey via pynput (if available)."""

    def start(self) -> None:
        super().start()
        self._global_listener = None
        try:
            from pynput import keyboard

            def _on_activate():
                self._on_abort()

            hotkey = keyboard.HotKey(
                keyboard.HotKey.parse("<ctrl>+<shift>+<esc>"),
                _on_activate,
            )
            self._global_listener = keyboard.Listener(
                on_press=lambda key: hotkey.press(key),  # type: ignore[arg-type]
                on_release=lambda key: hotkey.release(key),  # type: ignore[arg-type]
            )
            self._global_listener.start()
            log.info("global hotkey Ctrl+Shift+Esc registered via pynput")
        except Exception as exc:
            log.debug("pynput unavailable (%s); global hotkey disabled", exc)

    def stop(self) -> None:
        listener = getattr(self, "_global_listener", None)
        if listener is not None:
            with contextlib.suppress(Exception):
                listener.stop()
            self._global_listener = None
        super().stop()


def make_overlay(
    task_name: str,
    *,
    enabled: bool = True,
    abort_event: threading.Event | None = None,
) -> _BaseOverlay:
    """Factory that picks the right overlay for the current environment."""
    if not enabled:
        return NullOverlay(abort_event=abort_event)
    # Probe for a display; if none, fall back silently.
    import os

    if os.name == "posix" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return NullOverlay(abort_event=abort_event)
    return GlobalHotkeyOverlay(task_name=task_name, abort_event=abort_event)
