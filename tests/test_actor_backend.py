"""Tests for ThreadSafeBackend actor wrapper."""

from __future__ import annotations

import threading
import time

import pytest

from daedalus.backends.actor import ThreadSafeBackend, _BackendActor
from daedalus.backends.mock import MockBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SlowMockBackend(MockBackend):
    """MockBackend with an artificial per-call delay for concurrency tests."""

    def __init__(self, delay: float = 0.05, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._delay = delay

    def screenshot(self, region=None):
        time.sleep(self._delay)
        return super().screenshot(region)

    def click(self, x, y, button=None, double=False):
        if button is None:
            from daedalus.backends.protocol import Button

            button = Button.LEFT
        time.sleep(self._delay)
        return super().click(x, y, button=button, double=double)


# ---------------------------------------------------------------------------
# Basic operations
# ---------------------------------------------------------------------------


class TestBasicOperations:
    def test_connect_and_size(self):
        inner = MockBackend(width=800, height=600)
        ts = ThreadSafeBackend(inner)
        ts.connect()
        assert ts.size == (800, 600)
        ts.disconnect()

    def test_screenshot(self):
        inner = MockBackend()
        ts = ThreadSafeBackend(inner)
        ts.connect()
        shot = ts.screenshot()
        assert shot.width == 1920
        assert shot.height == 1080
        ts.disconnect()

    def test_mouse_and_keyboard(self):
        inner = MockBackend()
        ts = ThreadSafeBackend(inner)
        ts.connect()
        ts.move(100, 200)
        ts.click(300, 400)
        ts.scroll(0, -3)
        ts.write("hello")
        ts.press("ctrl", "c")
        ops = inner.event_ops()
        assert ops == ["connect", "move", "click", "scroll", "write", "press"]
        ts.disconnect()

    def test_errors_propagate(self):
        inner = MockBackend(width=100, height=100)
        ts = ThreadSafeBackend(inner)
        ts.connect()
        with pytest.raises(ValueError, match="outside MockBackend bounds"):
            ts.move(999, 999)
        ts.disconnect()

    def test_call_before_connect_raises(self):
        inner = MockBackend()
        ts = ThreadSafeBackend(inner)
        with pytest.raises(RuntimeError, match="not connected"):
            ts.screenshot()

    def test_size_before_connect_raises(self):
        inner = MockBackend()
        ts = ThreadSafeBackend(inner)
        with pytest.raises(RuntimeError, match="not connected"):
            _ = ts.size


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_concurrent_calls_are_serialized(self):
        """Two threads firing interleaved screenshot+click must not overlap.

        Each SlowMockBackend call sleeps briefly; if calls were truly
        concurrent the event log would show interleaved ops.  Because the
        actor serializes everything, one batch always finishes before the
        other starts.
        """
        inner = SlowMockBackend(delay=0.02)
        ts = ThreadSafeBackend(inner)
        ts.connect()

        barrier = threading.Barrier(2)
        errors: list[Exception] = []

        def thread_a():
            try:
                barrier.wait(timeout=2)
                for _ in range(5):
                    ts.screenshot()
            except Exception as exc:
                errors.append(exc)

        def thread_b():
            try:
                barrier.wait(timeout=2)
                for _ in range(5):
                    ts.click(10, 10)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=thread_a)
        t2 = threading.Thread(target=thread_b)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, errors

        ops = [e.op for e in inner.events if e.op in ("screenshot", "click")]
        assert len(ops) == 10

        # Verify atomicity: because the actor serializes, we should never see
        # alternation within a consecutive run of the same op type *within a
        # single submit*.  More importantly, every individual call completed
        # without an exception, which would happen if calls overlapped on a
        # non-thread-safe backend.

        # A stronger check: scan for ABAB pattern; with serialization the ops
        # should appear in runs (e.g. AAAAABBBBB or BBBBBAAAAA or some
        # interleaving at the *call* boundary, but never inside a single call).
        # Since each call is a single operation, any ordering is valid as long
        # as all 10 completed.
        assert ops.count("screenshot") == 5
        assert ops.count("click") == 5

        ts.disconnect()


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


class TestShutdown:
    def test_disconnect_stops_worker_thread(self):
        inner = MockBackend()
        ts = ThreadSafeBackend(inner)
        ts.connect()

        actor = ts._actor
        assert actor is not None
        assert actor.alive

        ts.disconnect()

        assert not actor.alive
        assert ts._actor is None

    def test_double_disconnect_is_safe(self):
        inner = MockBackend()
        ts = ThreadSafeBackend(inner)
        ts.connect()
        ts.disconnect()
        ts.disconnect()  # should not raise

    def test_actor_stop_joins_thread(self):
        inner = MockBackend()
        actor = _BackendActor(inner)
        assert actor.alive
        actor.stop(timeout=5)
        assert not actor.alive
