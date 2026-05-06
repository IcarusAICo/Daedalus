"""Structured trace recording.

Every task run emits a JSONL stream of events plus PNG screenshots and gets a
row in the SQLite index. The Learner (Phase 2) reads these.
"""

from daedalus.tracing.recorder import TraceEvent, TraceRecorder, list_traces

__all__ = ["TraceEvent", "TraceRecorder", "list_traces"]
