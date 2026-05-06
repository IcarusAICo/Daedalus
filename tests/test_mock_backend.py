"""MockBackend records every call and validates bounds."""

from __future__ import annotations

import pytest

from daedalus.backends.mock import MockBackend
from daedalus.backends.protocol import Button, Rect


def test_screenshot_returns_correct_size():
    be = MockBackend(width=800, height=600)
    be.connect()
    shot = be.screenshot()
    assert shot.width == 800
    assert shot.height == 600
    assert shot.image.size == (800, 600)


def test_screenshot_with_region_crops():
    be = MockBackend()
    be.connect()
    shot = be.screenshot(Rect(x=0, y=0, width=10, height=20))
    assert shot.width == 10
    assert shot.height == 20


def test_click_records_event():
    be = MockBackend()
    be.connect()
    be.click(50, 60, button=Button.RIGHT)
    ops = be.event_ops()
    assert "click" in ops
    last = be.events[-1]
    assert last.args == {"x": 50, "y": 60, "button": "right", "double": False}


def test_press_records_keys_in_order():
    be = MockBackend()
    be.connect()
    be.press("ctrl", "shift", "t")
    assert be.events[-1].args == {"keys": ["ctrl", "shift", "t"]}


def test_disconnected_calls_raise():
    be = MockBackend()
    with pytest.raises(RuntimeError):
        be.click(1, 2)


def test_out_of_bounds_click_rejected():
    be = MockBackend(width=100, height=100)
    be.connect()
    with pytest.raises(ValueError):
        be.click(150, 50)


def test_event_log_resets_on_reset():
    be = MockBackend()
    be.connect()
    be.click(1, 2)
    assert len(be.events) >= 1
    be.reset()
    assert be.events == []
