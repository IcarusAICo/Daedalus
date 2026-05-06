"""Thread-safe wrapper that serializes all backend calls through a single worker.

Wrap any :class:`RemoteDesktop` backend in :class:`ThreadSafeBackend` to guarantee
that only one call executes at a time, regardless of how many threads share the
backend handle.  This is useful when a backend (e.g. the Twisted-based VNC client)
is not thread-safe internally.
"""

from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from typing import Any

from daedalus.backends.protocol import Button, Rect, RemoteDesktop, Screenshot

_SENTINEL = object()


class _BackendActor:
    """Daemon thread that drains a work queue and executes calls one-at-a-time."""

    def __init__(self, backend: RemoteDesktop) -> None:
        self._backend = backend
        self._queue: queue.Queue[
            tuple[str, tuple[Any, ...], dict[str, Any], Future[Any]] | object
        ] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                self._queue.task_done()
                return
            method_name, args, kwargs, future = item  # type: ignore[misc]
            try:
                result = getattr(self._backend, method_name)(*args, **kwargs)
                future.set_result(result)
            except BaseException as exc:
                future.set_exception(exc)
            finally:
                self._queue.task_done()

    def submit(
        self,
        method_name: str,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Future[Any]:
        fut: Future[Any] = Future()
        self._queue.put((method_name, args, kwargs or {}, fut))
        return fut

    def stop(self, timeout: float | None = 10.0) -> None:
        self._queue.put(_SENTINEL)
        self._thread.join(timeout=timeout)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()


class ThreadSafeBackend:
    """Wraps any :class:`RemoteDesktop` backend with actor-style serialization.

    Every method call is forwarded to a single worker thread, so callers from
    arbitrary threads never race with each other.

    Parameters
    ----------
    backend:
        The concrete backend to wrap.
    call_timeout_s:
        Maximum seconds to wait for any single call to complete.
        ``None`` means wait forever.
    """

    def __init__(
        self,
        backend: RemoteDesktop,
        call_timeout_s: float | None = 60.0,
    ) -> None:
        self._backend = backend
        self._call_timeout_s = call_timeout_s
        self._actor: _BackendActor | None = None
        self._cached_size: tuple[int, int] | None = None

    # -- Lifecycle -----------------------------------------------------------------

    def connect(self) -> None:
        self._actor = _BackendActor(self._backend)
        self._actor.submit("connect").result(timeout=self._call_timeout_s)
        self._cached_size = self._backend.size

    def disconnect(self) -> None:
        if self._actor is None:
            return
        try:
            self._actor.submit("disconnect").result(timeout=self._call_timeout_s)
        finally:
            self._actor.stop()
            self._actor = None
            self._cached_size = None

    @property
    def size(self) -> tuple[int, int]:
        if self._cached_size is None:
            raise RuntimeError("ThreadSafeBackend.size: not connected")
        return self._cached_size

    @property
    def is_connected(self) -> bool:
        return self._backend.is_connected

    # -- Proxied operations --------------------------------------------------------

    def screenshot(self, region: Rect | None = None) -> Screenshot:
        return self._call("screenshot", region=region)

    def move(self, x: int, y: int) -> None:
        self._call("move", x=x, y=y)

    def click(
        self,
        x: int,
        y: int,
        button: Button = Button.LEFT,
        double: bool = False,
    ) -> None:
        self._call("click", x=x, y=y, button=button, double=double)

    def write(self, text: str) -> None:
        self._call("write", text=text)

    def press(self, *keys: str) -> None:
        self._call("press", *keys)

    def scroll(self, dx: int, dy: int) -> None:
        self._call("scroll", dx=dx, dy=dy)

    # -- Internal ------------------------------------------------------------------

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if self._actor is None:
            raise RuntimeError(f"ThreadSafeBackend.{method}: not connected")
        return self._actor.submit(method, args, kwargs).result(
            timeout=self._call_timeout_s,
        )
