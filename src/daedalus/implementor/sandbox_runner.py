"""Subprocess sandbox for loading and testing synthesized skills.

Invoked by the Implementor via:
    subprocess.run([sys.executable, "-I", "-S", "-B", "-m", "daedalus.implementor.sandbox_runner"], ...)

Reads JSON from stdin, writes JSON to stdout. Exit code 0 = success.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path


def _apply_resource_limits() -> None:
    """Best-effort CPU limit on Linux."""
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    except Exception:
        pass


def main() -> None:
    _apply_resource_limits()

    raw = sys.stdin.read()
    request = json.loads(raw)
    skill_dir = Path(request["skill_dir"])
    skill_id = request["skill_id"]
    fixtures = request.get("fixtures", [])

    result: dict = {"ok": True, "test_results": [], "error": None}

    try:
        from daedalus.core.registry import Registry, use_registry
        from daedalus.library.loader import load_skill

        sandbox_registry = Registry()
        with use_registry(sandbox_registry):
            loaded_id = load_skill(skill_dir, registry=sandbox_registry)

        if loaded_id != skill_id:
            result["ok"] = False
            result["error"] = f"loaded id {loaded_id!r} != expected {skill_id!r}"
            print(json.dumps(result))
            return

        skill_cls = sandbox_registry.get(loaded_id).cls

        from daedalus.backends.mock import MockBackend
        from daedalus.core.context import ExecutionContext, TaskState
        from daedalus.tracing.recorder import TraceRecorder

        for fixture in fixtures:
            fx_result = {"name": fixture.get("name", "?"), "ok": True, "message": "ok"}
            try:
                inputs = fixture["inputs"]
                expected_output = fixture.get("expected_output", {})
                expected_events = fixture.get("expected_events", [])
                ignore_keys = set(fixture.get("ignore_output_keys", []))

                with tempfile.TemporaryDirectory() as tmp:
                    tmpdir = Path(tmp)
                    backend = MockBackend()
                    backend.connect()
                    tracer = TraceRecorder(
                        traces_root=tmpdir, db_path=tmpdir / "tasks.db",
                        task_name="sandbox"
                    )
                    state = TaskState(tmpdir / "tasks.db", tracer.task_id)
                    ctx = ExecutionContext(
                        task_id=tracer.task_id, backend=backend,
                        task_state=state, tracer=tracer,
                    )
                    inp = skill_cls.Inputs.model_validate(inputs)
                    out = skill_cls().run(inp, ctx)
                    out_dict = out.model_dump(mode="json") if hasattr(out, "model_dump") else dict(out)
                    tracer.finish("success")

                actual = {k: v for k, v in out_dict.items() if k not in ignore_keys}
                expected = {k: v for k, v in expected_output.items() if k not in ignore_keys}
                if actual != expected:
                    fx_result["ok"] = False
                    fx_result["message"] = f"output {actual} != expected {expected}"
                else:
                    search_start = 0
                    for want in expected_events:
                        op = want["op"]
                        args = want.get("args", {})
                        matched = False
                        for j, e in enumerate(backend.events[search_start:], start=search_start):
                            if e.op != op:
                                continue
                            if all(e.args.get(k) == v for k, v in args.items()):
                                matched = True
                                search_start = j + 1
                                break
                        if not matched:
                            fx_result["ok"] = False
                            fx_result["message"] = f"missing expected event {want}"
                            break
            except Exception as exc:
                fx_result["ok"] = False
                fx_result["message"] = f"{type(exc).__name__}: {exc}"

            result["test_results"].append(fx_result)
            if not fx_result["ok"]:
                result["ok"] = False

    except Exception as exc:
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(result))


if __name__ == "__main__":
    main()
