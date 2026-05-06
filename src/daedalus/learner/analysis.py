"""Heuristic trace analysis. Produces objective findings from JSONL traces.

These findings are usable on their own (the CLI will print them) and are also
fed to the LLM-backed :class:`daedalus.learner.Learner` to ground its proposals.

We deliberately keep the heuristics simple and deterministic so the same
input always yields the same output. Statistics-driven findings:

- per-skill timings (calls, mean ms, p95 ms)
- per-skill failures (count, distinct error types, sample message)
- recurring n-gram sub-sequences in the skill_finished stream
- screenshot count, total event count
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SkillTiming:
    skill_id: str
    calls: int
    mean_ms: float
    p95_ms: float
    total_ms: float


@dataclass
class SkillFailure:
    skill_id: str
    failures: int
    error_types: list[str]
    sample_message: str


@dataclass
class Ngram:
    skills: tuple[str, ...]
    occurrences: int
    in_traces: int  # number of distinct traces this n-gram appeared in


@dataclass
class TraceSummary:
    task_id: str
    name: str
    status: str
    started: str
    finished: str | None
    step_count: int
    event_count: int
    screenshot_count: int
    total_duration_ms: float
    timings: dict[str, SkillTiming]
    failures: dict[str, SkillFailure]
    skill_sequence: list[str]


@dataclass
class HeuristicFindings:
    traces_analyzed: int
    overall_status_counts: Counter[str]
    timings: dict[str, SkillTiming] = field(default_factory=dict)
    failures: dict[str, SkillFailure] = field(default_factory=dict)
    repeated_subsequences: list[Ngram] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _read_events(events_path: Path) -> list[dict[str, Any]]:
    if not events_path.exists():
        return []
    out: list[dict[str, Any]] = []
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_trace(task_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read meta.json and events.jsonl for a task. Returns (meta, events)."""
    meta_path = task_dir / "meta.json"
    events_path = task_dir / "events.jsonl"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return meta, _read_events(events_path)


# ---------------------------------------------------------------------------
# Per-trace analysis
# ---------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    k = (len(s) - 1) * pct / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def analyze_trace(task_dir: Path) -> TraceSummary:
    meta, events = load_trace(task_dir)
    timings_buckets: dict[str, list[float]] = defaultdict(list)
    failures: dict[str, SkillFailure] = {}
    skill_sequence: list[str] = []
    screenshot_count = 0
    total_dur = 0.0

    for evt in events:
        kind = evt.get("kind")
        data = evt.get("data") or {}
        if kind == "skill_finished":
            sid = data.get("skill_id", "?")
            ms = float(data.get("duration_ms") or 0.0)
            timings_buckets[sid].append(ms)
            skill_sequence.append(sid)
            total_dur += ms
        elif kind == "skill_error":
            sid = data.get("skill_id", "?")
            etype = data.get("error_type") or "Unknown"
            msg = data.get("message") or ""
            f = failures.get(sid)
            if f is None:
                failures[sid] = SkillFailure(
                    skill_id=sid,
                    failures=1,
                    error_types=[etype],
                    sample_message=msg[:200],
                )
            else:
                f.failures += 1
                if etype not in f.error_types:
                    f.error_types.append(etype)
        elif kind == "screenshot":
            screenshot_count += 1

    timings: dict[str, SkillTiming] = {}
    for sid, vals in timings_buckets.items():
        timings[sid] = SkillTiming(
            skill_id=sid,
            calls=len(vals),
            mean_ms=statistics.fmean(vals),
            p95_ms=_percentile(vals, 95.0),
            total_ms=sum(vals),
        )

    return TraceSummary(
        task_id=meta.get("task_id") or task_dir.name,
        name=meta.get("task_name") or "?",
        status=meta.get("status") or "unknown",
        started=meta.get("started") or "",
        finished=meta.get("finished"),
        step_count=len(skill_sequence),
        event_count=len(events),
        screenshot_count=screenshot_count,
        total_duration_ms=total_dur,
        timings=timings,
        failures=failures,
        skill_sequence=skill_sequence,
    )


# ---------------------------------------------------------------------------
# Cross-trace n-gram mining
# ---------------------------------------------------------------------------


def _ngrams(seq: list[str], n: int) -> Iterable[tuple[str, ...]]:
    if n <= 0 or n > len(seq):
        return
    for i in range(len(seq) - n + 1):
        yield tuple(seq[i : i + n])


def find_repeated_subsequences(
    summaries: list[TraceSummary],
    *,
    min_n: int = 2,
    max_n: int = 5,
    min_occurrences: int = 2,
) -> list[Ngram]:
    """Find n-grams that recur across multiple traces or multiple times in
    a single trace. Returns hits sorted by (in_traces desc, occurrences desc,
    longer-first).
    """
    counts: Counter[tuple[str, ...]] = Counter()
    in_traces: defaultdict[tuple[str, ...], set[str]] = defaultdict(set)
    for s in summaries:
        for n in range(min_n, max_n + 1):
            for ng in _ngrams(s.skill_sequence, n):
                counts[ng] += 1
                in_traces[ng].add(s.task_id)
    out: list[Ngram] = []
    for ng, occ in counts.items():
        if occ < min_occurrences:
            continue
        out.append(Ngram(skills=ng, occurrences=occ, in_traces=len(in_traces[ng])))
    out.sort(key=lambda x: (-x.in_traces, -x.occurrences, -len(x.skills)))
    # Drop strict subgrams of a larger covered ngram if they have the same support.
    pruned: list[Ngram] = []
    for ng in out:
        absorbed = False
        for bigger in pruned:
            if (
                len(bigger.skills) > len(ng.skills)
                and bigger.occurrences == ng.occurrences
                and bigger.in_traces == ng.in_traces
                and _is_contiguous_subsequence(ng.skills, bigger.skills)
            ):
                absorbed = True
                break
        if not absorbed:
            pruned.append(ng)
    return pruned


def _is_contiguous_subsequence(small: tuple[str, ...], big: tuple[str, ...]) -> bool:
    if len(small) > len(big):
        return False
    return any(
        big[i : i + len(small)] == small for i in range(len(big) - len(small) + 1)
    )


# ---------------------------------------------------------------------------
# Top-level analyzer
# ---------------------------------------------------------------------------


def analyze_traces(task_dirs: list[Path]) -> HeuristicFindings:
    summaries = [analyze_trace(d) for d in task_dirs]
    status_counts: Counter[str] = Counter(s.status for s in summaries)

    # Aggregate timings across traces
    agg_times: dict[str, list[float]] = defaultdict(list)
    for s in summaries:
        for t in s.timings.values():
            agg_times[t.skill_id].extend([t.mean_ms] * t.calls)
    timings: dict[str, SkillTiming] = {}
    for sid, vals in agg_times.items():
        timings[sid] = SkillTiming(
            skill_id=sid,
            calls=len(vals),
            mean_ms=statistics.fmean(vals),
            p95_ms=_percentile(vals, 95.0),
            total_ms=sum(vals),
        )

    # Aggregate failures
    failures: dict[str, SkillFailure] = {}
    for s in summaries:
        for sid, f in s.failures.items():
            existing = failures.get(sid)
            if existing is None:
                failures[sid] = SkillFailure(
                    skill_id=sid,
                    failures=f.failures,
                    error_types=list(f.error_types),
                    sample_message=f.sample_message,
                )
            else:
                existing.failures += f.failures
                for et in f.error_types:
                    if et not in existing.error_types:
                        existing.error_types.append(et)

    repeats = find_repeated_subsequences(summaries)

    notes: list[str] = []
    if status_counts.get("failed"):
        notes.append(
            f"{status_counts['failed']} trace(s) failed; consider tightening preconditions."
        )
    if any(t.p95_ms > 1000 for t in timings.values()):
        slow = sorted(timings.values(), key=lambda x: -x.p95_ms)[:3]
        notes.append(
            "slow steps (>1s p95): " + ", ".join(f"{t.skill_id}({t.p95_ms:.0f}ms)" for t in slow)
        )
    if repeats:
        biggest = repeats[0]
        notes.append(
            f"recurring sequence {' -> '.join(biggest.skills)} occurs {biggest.occurrences}x"
            f" across {biggest.in_traces} trace(s); candidate for a compound skill."
        )

    return HeuristicFindings(
        traces_analyzed=len(summaries),
        overall_status_counts=status_counts,
        timings=timings,
        failures=failures,
        repeated_subsequences=repeats,
        notes=notes,
    )
