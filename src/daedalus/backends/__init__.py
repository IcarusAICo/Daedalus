"""Remote-desktop backends.

The :class:`RemoteDesktop` protocol is the only thing skills depend on. Concrete
backends are loaded by name via :func:`make_backend`.
"""

from daedalus.backends.actor import ThreadSafeBackend
from daedalus.backends.protocol import (
    Button,
    Point,
    Rect,
    RemoteDesktop,
    Screenshot,
)

__all__ = [
    "Button",
    "Point",
    "Rect",
    "RemoteDesktop",
    "Screenshot",
    "ThreadSafeBackend",
    "make_backend",
]


def make_backend(
    kind: str, *, thread_safe: bool = False, **kwargs: object
) -> RemoteDesktop:
    """Factory. ``kind`` is one of: ``"mock"``, ``"vnc"``.

    When *thread_safe* is ``True`` the backend is wrapped in
    :class:`ThreadSafeBackend` so all calls are serialized through a single
    worker thread.
    """
    if kind == "mock":
        from daedalus.backends.mock import MockBackend

        backend: RemoteDesktop = MockBackend(**kwargs)  # type: ignore[arg-type]
    elif kind == "vnc":
        from daedalus.backends.vnc import VNCBackend

        backend = VNCBackend(**kwargs)  # type: ignore[arg-type]
    else:
        raise ValueError(f"unknown backend kind {kind!r}")

    if thread_safe:
        backend = ThreadSafeBackend(backend)
    return backend
