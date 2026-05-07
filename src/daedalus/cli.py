"""Top-level command-line interface.

::

    daedalus run --program PATH [--backend mock|vnc] [--host ...] [--yes]
    daedalus skills list
    daedalus skills test [SKILL_ID]
    daedalus traces list
    daedalus traces show TASK_ID
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

from daedalus.backends import make_backend
from daedalus.core.context import ExecutionContext, TaskState, compute_coordinate_scale
from daedalus.core.errors import DaedalusError, ProgramValidationError
from daedalus.core.registry import get_registry
from daedalus.evaluator.criteria import SuccessCriteria
from daedalus.executor.dsl import (
    AnyProgram,
    Program,
    PythonProgram,
    load_program,
    parse_program,
    validate_program_against_registry,
)
from daedalus.executor.program_executor import PythonProgramExecutor
from daedalus.executor.runner import SequentialExecutor
from daedalus.implementor import ImplementorRequest, SyntheticSkillImplementor
from daedalus.library import Librarian, load_library
from daedalus.llm.gateway import LLMConfig, LLMGateway, make_gateway
from daedalus.memory import AgentMemory
from daedalus.planner import Planner
from daedalus.learner import Learner
from daedalus.tracing.recorder import TraceRecorder, list_traces
from daedalus.ui.confirm import ConfirmDecision, ConfirmResult, confirm_program, confirm_skills
from daedalus.ui.overlay import make_overlay

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    invoke_without_command=True,
    help="daedalus – computer control agent",
)
skills_app = typer.Typer(no_args_is_help=True, help="Skill library commands.")
traces_app = typer.Typer(no_args_is_help=True, help="Inspect recorded traces.")
app.add_typer(skills_app, name="skills")
app.add_typer(traces_app, name="traces")

console = Console()


@app.callback()
def _main_callback(ctx: typer.Context) -> None:
    """Launch the interactive shell when no subcommand is given (legacy fallback)."""
    if ctx.invoked_subcommand is None:
        # When invoked directly via `daedalus.cli:app`, show help.
        # The interactive UI is now handled by main.py -> Node.js frontend.
        ctx.get_help()
        raise typer.Exit(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if verbose:
        # Keep noisy third-party loggers at INFO to avoid dumping huge
        # base64 blobs and full HTTP bodies to the terminal.
        for noisy in ("LiteLLM", "httpcore", "botocore", "urllib3", "PIL", "asyncio"):
            logging.getLogger(noisy).setLevel(logging.INFO)


def _resolve_skills_dir(explicit: Path | None) -> Path:
    if explicit:
        return explicit.resolve()
    env = os.environ.get("DAEDALUS_SKILLS_DIR")
    if env:
        return Path(env).resolve()
    # Convention: skills/ at the repo root (the cwd in development).
    return (Path.cwd() / "skills").resolve()


def _resolve_traces_dir(explicit: Path | None) -> Path:
    if explicit:
        return explicit.resolve()
    return (Path.cwd() / "traces").resolve()


def _update_run_meta(run_dir: Path, **updates: Any) -> None:
    """Update run-level meta.json with the given fields."""
    meta_path = run_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta.update(updates)
        meta_path.write_text(json.dumps(meta, indent=2))
    except Exception:
        pass


def _resolve_db(explicit: Path | None) -> Path:
    if explicit:
        return explicit.resolve()
    return (Path.cwd() / "tasks.db").resolve()


_LAUNCHCTL_LABEL = "com.daedalus.screenrecord"


class _ScreenRecorder:
    """Records the screen on a remote macOS host via launchctl (to get GUI session
    access for Screen Recording permissions), then pulls the file back via scp.
    Falls back to x11grab for local displays."""

    def __init__(
        self,
        output_path: Path,
        backend: Any,
        *,
        fps: int = 30,
        host: str = "127.0.0.1",
        port: int = 5900,
    ) -> None:
        self._output_path = output_path
        self._backend = backend
        self._fps = fps
        self._host = host
        self._port = port
        self._proc: subprocess.Popen | None = None
        self._remote_path: str | None = None
        self._ssh_host: str | None = None
        self._plist_path: str = "/tmp/daedalus_record.plist"

    def _ssh(self, cmd: str, *, timeout: int = 10) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
             self._ssh_host, cmd],
            timeout=timeout, capture_output=True, stdin=subprocess.DEVNULL,
        )

    def start(self) -> None:
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        log = logging.getLogger(__name__)

        # Local display: use x11grab directly.
        display = os.environ.get("DISPLAY")
        if display:
            cmd = [
                "ffmpeg", "-y",
                "-f", "x11grab",
                "-framerate", str(self._fps),
                "-i", display,
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "28",
                str(self._output_path),
            ]
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.PIPE,
                )
            except FileNotFoundError:
                log.warning("ffmpeg not found — screen recording disabled")
            return

        # Remote macOS host: use launchctl to run ffmpeg in the GUI session
        # so it has Screen Recording permission via avfoundation.
        self._ssh_host = self._host
        self._remote_path = f"/tmp/daedalus_recording_{os.getpid()}.mp4"

        plist_xml = f"""\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" \
"http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>{_LAUNCHCTL_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/opt/homebrew/bin/ffmpeg</string>
    <string>-y</string>
    <string>-f</string>
    <string>avfoundation</string>
    <string>-framerate</string>
    <string>{self._fps}</string>
    <string>-capture_cursor</string>
    <string>1</string>
    <string>-i</string>
    <string>1:none</string>
    <string>-c:v</string>
    <string>libx264</string>
    <string>-preset</string>
    <string>ultrafast</string>
    <string>-crf</string>
    <string>23</string>
    <string>-pix_fmt</string>
    <string>yuv420p</string>
    <string>{self._remote_path}</string>
  </array>
  <key>StandardOutPath</key>
  <string>/tmp/daedalus_ffmpeg.log</string>
  <key>StandardErrorPath</key>
  <string>/tmp/daedalus_ffmpeg.log</string>
  <key>RunAtLoad</key>
  <true/>
</dict>
</plist>"""

        try:
            # Write plist, unload any previous instance, then load to start recording.
            write_cmd = (
                f"cat > {self._plist_path} << 'PLISTEOF'\n{plist_xml}\nPLISTEOF"
            )
            self._ssh(write_cmd)
            self._ssh(f"launchctl unload {self._plist_path} 2>/dev/null; true")
            self._ssh(f"rm -f {self._remote_path}")
            result = self._ssh(f"launchctl load {self._plist_path}")
            if result.returncode == 0:
                log.info("screen recording started on %s via launchctl: %s",
                         self._ssh_host, self._remote_path)
                self._proc = True  # type: ignore[assignment]  # sentinel
            else:
                log.warning("launchctl load failed: %s",
                            result.stderr.decode(errors="replace"))
        except FileNotFoundError:
            log.warning("ssh not found — screen recording disabled")
        except Exception as exc:
            log.warning("failed to start remote recording: %s", exc)

    def stop(self) -> Path | None:
        import time
        log = logging.getLogger(__name__)

        if self._proc is None:
            return None

        if self._ssh_host and self._remote_path:
            # Stop the launchctl job (sends SIGTERM to ffmpeg, which finalizes the file).
            try:
                self._ssh(f"launchctl unload {self._plist_path} 2>/dev/null; true")
            except Exception:
                pass
            time.sleep(2)

            # Pull the recording back via scp.
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            output = self._output_path.with_suffix(".mp4")
            scp_cmd = [
                "scp", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
                f"{self._ssh_host}:{self._remote_path}",
                str(output),
            ]
            try:
                result = subprocess.run(scp_cmd, timeout=120, capture_output=True)
                if result.returncode == 0:
                    log.info("recording pulled to %s", output)
                    self._output_path = output
                else:
                    log.warning("scp failed: %s", result.stderr.decode(errors="replace"))
            except Exception as exc:
                log.warning("failed to pull recording: %s", exc)

            # Clean up remote files.
            try:
                self._ssh(f"rm -f {self._remote_path} {self._plist_path} /tmp/daedalus_ffmpeg.log")
            except Exception:
                pass
        else:
            # Local x11grab: send 'q' to ffmpeg.
            try:
                if self._proc.stdin:
                    self._proc.stdin.write(b"q")
                    self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()

        if self._output_path.exists() and self._output_path.stat().st_size > 0:
            return self._output_path
        return None


def _ensure_library_loaded(skills_dir: Path) -> int:
    registry = get_registry()
    if len(registry) == 0:
        try:
            ids = load_library(skills_dir)
            return len(ids)
        except Exception as exc:
            console.print(f"[red]failed to load skill library at {skills_dir}: {exc}[/red]")
            raise typer.Exit(2) from exc
    return len(registry)


def _load_llm_config(config_path: Path | None) -> LLMConfig | None:
    """Load LLM role mapping from a YAML config. Returns None if no config."""
    if config_path is None or not config_path.exists():
        return None
    raw = yaml.safe_load(config_path.read_text()) or {}
    llm = raw.get("llm") or {}
    roles = (llm.get("roles") or {}) if isinstance(llm, dict) else {}
    if not roles:
        return None
    aws_region = llm.get("aws_region") if isinstance(llm, dict) else None
    kwargs: dict[str, object] = {"roles": roles}
    if aws_region:
        kwargs["aws_region"] = aws_region
    if isinstance(llm, dict):
        if "request_timeout_s" in llm:
            kwargs["request_timeout_s"] = float(llm["request_timeout_s"])
        if "max_retries" in llm:
            kwargs["max_retries"] = int(llm["max_retries"])
        if "creative_temperature" in llm:
            kwargs["creative_temperature"] = float(llm["creative_temperature"])
        if "analytical_temperature" in llm:
            kwargs["analytical_temperature"] = float(llm["analytical_temperature"])
    return LLMConfig(**kwargs)  # type: ignore[arg-type]


def _load_host_os(config_path: Path | None) -> str:
    """Read backend.host_os from config. Defaults to 'unknown'."""
    if config_path is None or not config_path.exists():
        return "unknown"
    raw = yaml.safe_load(config_path.read_text()) or {}
    backend = raw.get("backend") or {}
    return str(backend.get("host_os", "unknown")) if isinstance(backend, dict) else "unknown"


def _render_success_criteria(c: Console, criteria: SuccessCriteria) -> None:
    """Display success criteria for user review."""
    from rich.panel import Panel
    from rich.text import Text

    lines = [Text.from_markup(f"[bold]{criteria.goal_summary}[/bold]")]
    mode = "ALL must pass" if criteria.must_pass_all else "ANY passing is sufficient"
    lines.append(Text.from_markup(f"[dim]Mode: {mode}[/dim]"))
    c.print(Panel(Text("\n").join(lines), title="Success Criteria", border_style="green"))

    from rich.table import Table

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("kind", style="cyan", no_wrap=True)
    table.add_column("description", overflow="fold")
    table.add_column("check", overflow="fold", style="dim")
    for i, cr in enumerate(criteria.criteria):
        if cr.kind == "visual":
            check = cr.visual_claim or ""
        elif cr.kind == "trace":
            check = cr.trace_pattern or ""
        elif cr.kind == "state":
            check = f"{cr.state_key}: {cr.state_condition}"
        else:
            check = ""
        table.add_row(str(i), cr.kind, cr.description, check)
    c.print(table)


def _confirm_criteria(
    c: Console,
    criteria: SuccessCriteria,
    *,
    auto_yes: bool = False,
) -> ConfirmResult:
    """Show criteria and ask for approval. Returns a ConfirmResult."""
    _render_success_criteria(c, criteria)
    if auto_yes:
        c.print("[yellow]success criteria auto-approved (--yes)[/yellow]")
        return ConfirmResult(decision=ConfirmDecision.APPROVE)
    if not sys.stdin.isatty():
        c.print("[bold red]No TTY; auto-approving success criteria.[/bold red]")
        return ConfirmResult(decision=ConfirmDecision.APPROVE)
    c.print(
        "Type [bold green]approve[/bold green] to accept these success criteria, "
        "[bold yellow]deny[/bold yellow] to reject with comments, "
        "anything else to skip criteria evaluation."
    )
    answer = input(">> ").strip().lower()
    if answer == "approve":
        return ConfirmResult(decision=ConfirmDecision.APPROVE)
    if answer == "deny":
        comments = input("Comments >> ").strip()
        return ConfirmResult(decision=ConfirmDecision.DENY_WITH_COMMENTS, comments=comments)
    return ConfirmResult(decision=ConfirmDecision.REJECT)


def _render_verdict(c: Console, result) -> None:  # type: ignore[no-untyped-def]
    """Print the GoalVerdict after execution."""
    from rich.panel import Panel
    from rich.text import Text

    verdict = result.goal_verdict
    if verdict is None:
        return

    status_style = "bold green" if verdict.achieved else "bold red"
    status_text = "GOAL ACHIEVED" if verdict.achieved else "GOAL NOT ACHIEVED"

    lines = [Text.from_markup(f"[{status_style}]{status_text}[/{status_style}]")]
    lines.append(Text(verdict.summary, style="dim"))

    for r in verdict.results:
        icon = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        lines.append(
            Text.from_markup(f"  {icon} [{r.criterion.kind}] {r.criterion.description}")
        )
        if r.explanation:
            lines.append(Text(f"       {r.explanation}", style="dim"))

    c.print(Panel(Text("\n").join(lines), title="Goal Evaluation", border_style=("green" if verdict.achieved else "red")))


def _close_backend(be) -> None:  # type: ignore[no-untyped-def]
    """Shut down the backend (client disconnect + reactor stop)."""
    with contextlib.suppress(Exception):
        be.close()


def _run_test_mode(
    *,
    bridge,
    config: Path | None,
    backend_kind: str,
    host: str,
    port: int,
    password_env: str,
    username_env: str,
    sk_dir: Path,
    tr_dir: Path,
    db_path: Path,
    run_dir: Path,
    console: Console,
) -> None:
    """Interactive skill REPL: connect to backend, wait for skill execution requests."""
    import threading
    from daedalus.core.store import RunStore

    registry = get_registry()

    # Connect to the backend
    if backend_kind == "vnc":
        max_res: tuple[int, int] | None = None
        effective_username_env = username_env
        if config and config.exists():
            raw_cfg = yaml.safe_load(config.read_text()) or {}
            vnc_cfg = (raw_cfg.get("backend") or {}).get("vnc") or {}
            if not effective_username_env:
                effective_username_env = vnc_cfg.get("username_env")
            mr_w = vnc_cfg.get("max_width")
            mr_h = vnc_cfg.get("max_height")
            if mr_w and mr_h:
                max_res = (int(mr_w), int(mr_h))
        password = os.environ.get(password_env) if password_env else None
        username = os.environ.get(effective_username_env) if effective_username_env else None
        be = make_backend(
            "vnc", host=host, port=port, password=password,
            username=username, max_resolution=max_res,
        )
    else:
        be = make_backend("mock")
    be.connect()

    # Set up an execution context — no_persist so test runs leave no trace files
    tracer = TraceRecorder(
        traces_root=run_dir,
        db_path=db_path,
        task_name="test-session",
        task_id="test",
        no_persist=True,
    )
    tracer.start()
    task_state = TaskState(db_path, tracer.task_id)
    store = RunStore(db_path, tracer.task_id)
    screen_w = be.size[0] if hasattr(be, "size") else 1920
    ctx = ExecutionContext(
        task_id=tracer.task_id,
        backend=be,
        task_state=task_state,
        tracer=tracer,
        store=store,
        abort_event=threading.Event(),
        coordinate_scale=compute_coordinate_scale(screen_w),
    )

    if bridge:
        bridge.emit_phase("executor", "running", "test mode — waiting for skill commands")
        bridge.emit_event("test_mode_ready", {
            "skills": [e.id for e in registry],
        })

        # Register request handler for execute_skill
        def _handle_execute_skill(params: dict) -> dict:
            skill_id = params.get("skill_id", "")
            inputs = params.get("inputs", {})

            entry = None
            for e in registry:
                if e.id == skill_id:
                    entry = e
                    break
            if entry is None:
                return {"success": False, "error": f"unknown skill: {skill_id}"}

            try:
                inp = entry.cls.Inputs.model_validate(inputs)
                instance = entry.cls()
                out = instance.run(inp, ctx)
                out_dict = out.model_dump(mode="json") if hasattr(out, "model_dump") else {}

                # Take a screenshot after execution
                try:
                    png_bytes = be.screenshot()
                    if png_bytes:
                        tracer.attach_screenshot(png_bytes, screen_w, be.size[1] if hasattr(be, "size") else 1080)
                except Exception:
                    pass

                return {"success": True, "outputs": out_dict}
            except Exception as exc:
                return {"success": False, "error": f"{type(exc).__name__}: {exc}"}

        bridge.register_handler("execute_skill", _handle_execute_skill)

        # Send initial screenshot
        try:
            png_bytes = be.screenshot()
            if png_bytes and hasattr(bridge, "emit_event"):
                import base64
                bridge.emit_event("screenshot", {
                    "width": screen_w,
                    "height": be.size[1] if hasattr(be, "size") else 1080,
                    "base64": base64.b64encode(png_bytes).decode(),
                })
        except Exception:
            pass

        # Block until the bridge is closed (frontend disconnects)
        try:
            bridge.wait_until_closed()
        except (KeyboardInterrupt, EOFError):
            pass
    else:
        # Non-frontend mode: simple terminal REPL
        console.print("[bold cyan]Test mode — interactive skill REPL[/bold cyan]")
        console.print("[dim]Available skills:[/dim]")
        for entry in sorted(registry, key=lambda e: e.id):
            console.print(f"  [cyan]{entry.id}[/cyan]")
        console.print("\n[dim]Type: skill_id {{\"input_key\": \"value\"}}  or 'quit' to exit[/dim]\n")

        while True:
            try:
                line = console.input("[green]skill>[/green] ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not line or line in ("quit", "exit", "q"):
                break

            # Parse: skill_id {json_inputs}
            parts = line.split(None, 1)
            skill_id = parts[0]
            raw_inputs = parts[1] if len(parts) > 1 else "{}"

            entry = None
            for e in registry:
                if e.id == skill_id:
                    entry = e
                    break
            if entry is None:
                console.print(f"[red]unknown skill: {skill_id}[/red]")
                continue

            try:
                inputs = json.loads(raw_inputs)
            except json.JSONDecodeError as exc:
                console.print(f"[red]invalid JSON inputs: {exc}[/red]")
                continue

            try:
                inp = entry.cls.Inputs.model_validate(inputs)
                instance = entry.cls()
                out = instance.run(inp, ctx)
                out_dict = out.model_dump(mode="json") if hasattr(out, "model_dump") else {}
                console.print(f"[green]OK:[/green] {json.dumps(out_dict, indent=2)}")
            except Exception as exc:
                console.print(f"[red]{type(exc).__name__}: {exc}[/red]")

    tracer.finish("success")
    _close_backend(be)


# ---------------------------------------------------------------------------
# `daedalus run`
# ---------------------------------------------------------------------------


@app.command("run")
def cmd_run(
    program: Optional[Path] = typer.Option(None, "--program", "-p", help="YAML program file."),
    goal: Optional[str] = typer.Option(None, "--goal", "-g", help="Free-form goal; invokes the planner, then confirms, then runs."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config with llm.roles."),
    backend: str = typer.Option("mock", "--backend", "-b", help="mock | vnc"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(5900, "--port"),
    password_env: Optional[str] = typer.Option("DAEDALUS_VNC_PASSWORD", "--password-env"),
    username_env: Optional[str] = typer.Option(None, "--username-env", help="Env var holding the VNC/ARD username (needed for macOS)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
    explain_only: bool = typer.Option(False, "--explain-only", help="Print the plan and exit."),
    no_overlay: bool = typer.Option(False, "--no-overlay", help="Disable the Tk overlay window."),
    skills_dir: Optional[Path] = typer.Option(None, "--skills-dir"),
    traces_dir: Optional[Path] = typer.Option(None, "--traces-dir"),
    tasks_db: Optional[Path] = typer.Option(None, "--tasks-db"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    no_strategy: bool = typer.Option(False, "--no-strategy", help="Skip the strategy phase (no proactive skill synthesis)."),
    mode: str = typer.Option("learn", "--mode", "-m", help="Agent mode: learn (explore+plan+learn), explore (explorer solves directly), plan (skip explorer, go to planner), test (interactive skill REPL)."),
    explore_steps: int = typer.Option(20, "--explore-steps", help="Max explorer iterations."),
    max_retries: int = typer.Option(3, "--max-retries", "-r", help="Max learner retry loops on failure."),
    learn_on_succeed: bool = typer.Option(False, "--learn-on-succeed", help="Run learner analysis even on success."),
    record: bool = typer.Option(False, "--record", help="Record the screen via ffmpeg during execution (saves to trace dir)."),
    record_fps: int = typer.Option(30, "--record-fps", help="Frames per second for screen recording (default 30)."),
    reuse_exploration: Optional[str] = typer.Option(None, "--reuse-exploration", help="Trace directory name to reuse exploration from (skips explorer, proceeds to planning)."),
    frontend_mode: bool = typer.Option(False, "--frontend-mode", help="Emit JSON-RPC events to stdout for the TypeScript UI."),
) -> None:
    """Run a hand-written or planner-emitted program."""
    _setup_logging(verbose)

    # --- Frontend bridge setup ---
    bridge = None
    if frontend_mode:
        from daedalus.ui.frontend_bridge import FrontendBridge
        bridge = FrontendBridge()
        abort_event = __import__("threading").Event()

        def _on_frontend_command(method: str, params: dict) -> None:
            if method == "abort":
                abort_event.set()

        bridge.start(on_command=_on_frontend_command)
        # Suppress Rich console output in frontend mode
        import io
        console.file = io.StringIO()

        # Redirect all logging to a file instead of stderr to avoid noise
        _log_path = Path(traces_dir or "traces") / "frontend.log"
        _log_path.parent.mkdir(parents=True, exist_ok=True)
        _file_handler = logging.FileHandler(str(_log_path), mode="w")
        _file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root_logger = logging.getLogger()
        # Remove all existing handlers (stderr ones)
        for h in root_logger.handlers[:]:
            root_logger.removeHandler(h)
        root_logger.addHandler(_file_handler)
        # Also quiet litellm's verbose output
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        logging.getLogger("litellm").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("twisted").setLevel(logging.WARNING)

    sk_dir = _resolve_skills_dir(skills_dir)
    tr_dir = _resolve_traces_dir(traces_dir)
    db_path = _resolve_db(tasks_db)

    # Create a single run directory for all artifacts of this session.
    run_id = f"t_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir = tr_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _run_started = datetime.now(timezone.utc).isoformat()
    _run_meta = {
        "run_id": run_id,
        "goal": goal,
        "mode": mode,
        "status": "running",
        "started": _run_started,
        "finished": None,
        "attempts": 0,
    }
    (run_dir / "meta.json").write_text(json.dumps(_run_meta, indent=2))

    if bridge:
        bridge.set_trace_dir(run_dir)

    # Load agent-level defaults from config file if present
    if config and config.exists():
        _raw = yaml.safe_load(config.read_text()) or {}
        _agent_cfg = _raw.get("agent") or {}
        if max_retries == 3 and "max_retries" in _agent_cfg:
            max_retries = int(_agent_cfg["max_retries"])
        if explore_steps == 20 and "explore_steps" in _agent_cfg:
            explore_steps = int(_agent_cfg["explore_steps"])
        if mode == "learn" and _agent_cfg.get("no_explore"):
            mode = "plan"
        if not no_strategy and _agent_cfg.get("no_strategy"):
            no_strategy = True
        if not verbose and _agent_cfg.get("verbose"):
            verbose = True
            _setup_logging(True)
        if not record and _agent_cfg.get("record"):
            record = True
        if record_fps == 30 and "record_fps" in _agent_cfg:
            record_fps = int(_agent_cfg["record_fps"])

        # Load context management config
        from daedalus.llm.context import ContextConfig, set_context_config
        _ctx_cfg = _raw.get("context") or {}
        set_context_config(ContextConfig.from_dict(_ctx_cfg))

    _ensure_library_loaded(sk_dir)

    # --- Test mode: interactive skill REPL ---
    if mode == "test":
        _run_test_mode(
            bridge=bridge,
            config=config,
            backend_kind=backend,
            host=host,
            port=port,
            password_env=password_env,
            username_env=username_env,
            sk_dir=sk_dir,
            tr_dir=tr_dir,
            db_path=db_path,
            run_dir=run_dir,
            console=console,
        )
        return

    criteria: SuccessCriteria | None = None

    if goal is not None:
        llm_cfg = _load_llm_config(config)
        if llm_cfg is None:
            console.print("[red]--goal requires --config with llm.roles[/red]")
            raise typer.Exit(2)
        gateway = make_gateway(llm_cfg)
        if gateway is None:
            console.print("[red]could not build gateway from config[/red]")
            raise typer.Exit(2)
        librarian = Librarian()
        librarian.reindex()
        host_os = _load_host_os(config)

        # Determine the logical screen size for the planner. Prefer
        # vnc.max_width/max_height (the downscaled resolution), fall back to
        # executor.default_screen_*, then error out -- no more guessing.
        plan_w: int | None = None
        plan_h: int | None = None
        if config and config.exists():
            raw_cfg = yaml.safe_load(config.read_text()) or {}
            vnc_cfg = (raw_cfg.get("backend") or {}).get("vnc") or {}
            exec_cfg = raw_cfg.get("executor") or {}
            plan_w = int(vnc_cfg.get("max_width") or exec_cfg.get("default_screen_width") or 0) or None
            plan_h = int(vnc_cfg.get("max_height") or exec_cfg.get("default_screen_height") or 0) or None
        if plan_w is None or plan_h is None:
            console.print(
                "[red]cannot determine screen resolution. Set backend.vnc.max_width/max_height "
                "or executor.default_screen_width/default_screen_height in the config.[/red]"
            )
            raise typer.Exit(2)

        planner = Planner(gateway=gateway, librarian=librarian, host_os=host_os, screen_size=(plan_w, plan_h))

        # Load agent memory for cross-run context.
        memory_db = db_path.parent / "memory.db"
        agent_memory = AgentMemory(memory_db)
        memory_context: str | None = None
        try:
            facts = agent_memory.recall(goal, limit=5)
            if facts:
                memory_context = "\n".join(f"- [{f.category}] {f.content}" for f in facts)
        except Exception:
            pass

        def _context_usage(used: int, max_tokens: int) -> None:
            if bridge:
                bridge.emit_context_usage(used, max_tokens)

        # Step 0: Strategy phase — subtask decomposition only (no skill synthesis).
        # Skipped when explorer is active since exploration provides richer context.
        if mode in ("learn", "explore") or reuse_exploration:
            console.print("[dim]strategy phase skipped (explorer is active)[/dim]")
            if bridge:
                bridge.emit_phase("strategy", "skipped")
        elif no_strategy:
            console.print("[dim]strategy phase skipped (--no-strategy)[/dim]")
            if bridge:
                bridge.emit_phase("strategy", "skipped")
        else:
            try:
                if bridge:
                    bridge.emit_phase("strategy", "running")
                console.print("[cyan]analyzing goal strategy...[/cyan]")
                strategy = planner.plan_strategy(goal)
                if strategy.needs_new_skills:
                    console.print(
                        f"[dim]strategy notes {len(strategy.composite_skills)} "
                        f"potential composite skill(s) (will be created by the learner if needed)[/dim]"
                    )
                    for cs in strategy.composite_skills:
                        console.print(f"  [dim]{cs.proposed_id}: {cs.description}[/dim]")
                if strategy.notes:
                    console.print(f"[dim]strategy: {strategy.notes}[/dim]")
                if bridge:
                    bridge.emit_phase("strategy", "complete", strategy.notes)
            except Exception as exc:
                console.print(f"[dim]strategy phase skipped: {exc}[/dim]")
                if bridge:
                    bridge.emit_phase("strategy", "failed", str(exc))

        # Step 0.5: Exploration phase — freeform environment discovery.
        if reuse_exploration:
            # Reuse exploration from a previous trace
            _reuse_dir = tr_dir / reuse_exploration / "explorer"
            _reuse_obs_path = _reuse_dir / "observations.md"
            if _reuse_obs_path.exists():
                _reuse_obs = _reuse_obs_path.read_text()
                console.print(f"[green]reusing exploration from {reuse_exploration}[/green]")
                obs_section = "\n\n## Explorer Observations\n" + _reuse_obs
                memory_context = (memory_context or "") + obs_section

                # Copy reused observations into current run's explorer dir
                _cur_explorer_dir = run_dir / "explorer"
                _cur_explorer_dir.mkdir(parents=True, exist_ok=True)
                (_cur_explorer_dir / "observations.md").write_text(_reuse_obs)
                (_cur_explorer_dir / "reused_from.txt").write_text(reuse_exploration)

                if bridge:
                    bridge.emit_phase("explorer", "complete", f"reused from {reuse_exploration}")
            else:
                console.print(f"[red]no explorer observations found in {reuse_exploration}[/red]")
                if bridge:
                    bridge.emit_phase("explorer", "failed", f"no observations in {reuse_exploration}")
                raise typer.Exit(2)
        elif mode == "plan":
            console.print("[dim]exploration phase skipped (plan mode)[/dim]")
            if bridge:
                bridge.emit_phase("explorer", "skipped")
        else:
            try:
                if bridge:
                    bridge.emit_phase("explorer", "running")
                console.print("[cyan]starting exploration phase...[/cyan]")
                # Build backend early for exploration.
                if backend == "mock":
                    explore_be = make_backend("mock")
                elif backend == "vnc":
                    _effective_username_env = username_env
                    _max_res: tuple[int, int] | None = None
                    if config and config.exists():
                        _raw_cfg = yaml.safe_load(config.read_text()) or {}
                        _vnc_cfg = (_raw_cfg.get("backend") or {}).get("vnc") or {}
                        if not _effective_username_env:
                            _effective_username_env = _vnc_cfg.get("username_env")
                        _mr_w = _vnc_cfg.get("max_width")
                        _mr_h = _vnc_cfg.get("max_height")
                        if _mr_w and _mr_h:
                            _max_res = (int(_mr_w), int(_mr_h))
                    _password = os.environ.get(password_env) if password_env else None
                    _username = os.environ.get(_effective_username_env) if _effective_username_env else None
                    explore_be = make_backend(
                        "vnc", host=host, port=port, password=_password,
                        username=_username, max_resolution=_max_res,
                    )
                else:
                    console.print(f"[red]unknown backend {backend!r}[/red]")
                    raise typer.Exit(2)

                from daedalus.explorer import Explorer

                if bridge:
                    bridge.emit_event("status_update", {"text": f"Connecting to {host}:{port}..."})
                explore_be.connect()
                if bridge:
                    bridge.emit_event("status_update", {"text": "Connected, starting exploration..."})

                implementor = SyntheticSkillImplementor(
                    gateway=gateway,
                    skills_dir=sk_dir,
                )
                explorer = Explorer(
                    gateway=gateway,
                    librarian=librarian,
                    implementor=implementor,
                    skills_dir=sk_dir,
                    traces_root=run_dir,
                    tasks_db=db_path,
                    verbose=verbose,
                    max_iterations=explore_steps,
                )

                def _explorer_progress(current: int, total: int, desc: str) -> None:
                    if bridge:
                        bridge.emit_explorer_progress(current, total)
                        bridge.emit_event("status_update", {"text": f"Explorer: {desc}"})
                        bridge.emit_thinking_clear()

                def _stream_thinking(token: str) -> None:
                    if bridge:
                        bridge.emit_thinking(token)

                _tool_call_counter = [0]
                def _tool_callback(name: str, tool_id: str, args: dict, result: str | None, image_path: str | None = None) -> None:
                    if bridge:
                        if result is None:
                            _tool_call_counter[0] += 1
                            bridge.emit_event("skill_started", {
                                "step_idx": _tool_call_counter[0],
                                "skill_id": name,
                                "inputs": args,
                            })
                        else:
                            data: dict = {
                                "step_idx": _tool_call_counter[0],
                                "skill_id": name,
                                "outputs": result[:200] if result else "",
                                "duration_ms": 0,
                            }
                            if image_path:
                                data["image_path"] = image_path
                            bridge.emit_event("skill_finished", data)

                explore_result = explorer.explore(
                    goal, backend=explore_be,
                    abort_event=abort_event if bridge else None,
                    progress_callback=_explorer_progress,
                    stream_callback=_stream_thinking if bridge else None,
                    tool_callback=_tool_callback if bridge else None,
                    context_usage_callback=_context_usage if bridge else None,
                    solve_mode=(mode == "explore"),
                )

                import contextlib as _ctxlib
                with _ctxlib.suppress(Exception):
                    explore_be.disconnect()

                console.print(
                    f"[green]exploration complete[/green]: "
                    f"{explore_result.tool_calls_count} tool calls, "
                    f"{len(explore_result.new_skills)} new skill(s)"
                )
                if bridge:
                    bridge.emit_phase(
                        "explorer", "complete",
                        f"{explore_result.tool_calls_count} tool calls, {len(explore_result.new_skills)} new skill(s)",
                    )
                if explore_result.new_skills:
                    for sid in explore_result.new_skills:
                        console.print(f"  [dim]new skill: {sid}[/dim]")
                    librarian.reindex()

                # Merge observations into memory context for the planner.
                obs_section = "\n\n## Explorer Observations\n" + explore_result.observations
                memory_context = (memory_context or "") + obs_section

                # Save explorer output to traces directory
                try:
                    explorer_dir = run_dir / "explorer"
                    explorer_dir.mkdir(parents=True, exist_ok=True)
                    (explorer_dir / "observations.md").write_text(explore_result.observations)
                    if explore_result.new_skills:
                        (explorer_dir / "new_skills.txt").write_text(
                            "\n".join(explore_result.new_skills)
                        )
                except Exception:
                    pass
            except Exception as exc:
                console.print(f"[yellow]exploration phase failed: {exc}[/yellow]")
                if bridge:
                    bridge.emit_phase("explorer", "failed", str(exc))

        # In "explore" mode the explorer is the sole actor — skip planning/execution.
        # Exception: when reusing exploration, always proceed to planning.
        if mode == "explore" and not reuse_exploration:
            console.print("[green]explore mode complete — task handled by explorer.[/green]")
            _update_run_meta(run_dir, status="success", finished=datetime.now(timezone.utc).isoformat())
            if bridge:
                bridge.emit_phase("planner", "skipped")
                bridge.emit_phase("executor", "skipped")
                bridge.emit_phase("evaluator", "skipped")
                bridge.emit_phase("learner", "skipped")
                bridge.emit_event("finished", {"success": True, "mode": "explore"})
            return

        # Step 1: Generate and approve success criteria.
        try:
            console.print("[cyan]generating success criteria...[/cyan]")
            if bridge:
                bridge.emit_event("status_update", {"text": "generating success criteria..."})
            criteria = planner.plan_success_criteria(goal)
            if bridge:
                # Log criteria to trace and frontend
                cr_data = criteria.model_dump(mode="json") if hasattr(criteria, "model_dump") else {}
                bridge.emit_event("criteria_generated", cr_data)
                if yes:
                    bridge.emit_event("status_update", {"text": "success criteria auto-approved (yolo mode)"})
                else:
                    result = bridge.confirm_criteria(cr_data)
                    if result.get("decision") == "approve":
                        pass  # keep criteria
                    elif result.get("decision") == "deny":
                        criteria = planner.plan_success_criteria(goal)
                    else:
                        criteria = None
            else:
                while True:
                    cr_result = _confirm_criteria(console, criteria, auto_yes=yes)
                    if cr_result.decision == ConfirmDecision.APPROVE:
                        break
                    elif cr_result.decision == ConfirmDecision.DENY_WITH_COMMENTS:
                        console.print("[cyan]regenerating success criteria with your feedback...[/cyan]")
                        criteria = planner.plan_success_criteria(goal)
                    else:
                        console.print("[yellow]skipping criteria evaluation[/yellow]")
                        criteria = None
                        break
        except Exception as exc:
            console.print(f"[yellow]could not generate success criteria: {exc}[/yellow]")
            criteria = None

        # Save success criteria to the run directory
        try:
            plan_dir = run_dir / "plan"
            plan_dir.mkdir(parents=True, exist_ok=True)
            if criteria is not None:
                cr_save = criteria.model_dump(mode="json") if hasattr(criteria, "model_dump") else {}
                (plan_dir / "success_criteria.json").write_text(json.dumps(cr_save, indent=2))
            if goal:
                (run_dir / "goal.txt").write_text(goal)
        except Exception:
            pass

        # Step 2: Plan (if planner reports missing skills, inform the user and re-plan).
        if bridge:
            bridge.emit_phase("planner", "running")
        learner_feedback_ctx: str | None = None
        max_plan_cycles = 3
        for cycle in range(max_plan_cycles):
            try:
                plan_result = planner.plan(
                    goal,
                    memory_context=memory_context,
                    learner_feedback=learner_feedback_ctx,
                )
            except Exception as exc:
                console.print(f"[red]planning failed: {exc}[/red]")
                raise typer.Exit(1) from exc

            if plan_result.program is not None:
                if plan_result.missing_skills:
                    console.print(
                        f"[yellow]planner noted {len(plan_result.missing_skills)} missing skill(s) "
                        f"but produced a plan anyway (skills may be created by the learner after execution):[/yellow]"
                    )
                    for ms in plan_result.missing_skills:
                        console.print(f"  [dim]{ms.proposed_id}: {ms.description}[/dim]")
                break

            if plan_result.missing_skills:
                console.print(
                    f"[yellow]cycle {cycle + 1}: planner cannot produce a plan without "
                    f"{len(plan_result.missing_skills)} missing skill(s):[/yellow]"
                )
                for ms in plan_result.missing_skills:
                    console.print(f"  [dim]{ms.proposed_id}: {ms.description}[/dim]")
                console.print("[dim]re-planning, asking planner to work with available skills only...[/dim]")
                learner_feedback_ctx = (
                    "The following skills are NOT available. You must produce a plan "
                    "using ONLY the available skills: "
                    + ", ".join(ms.proposed_id for ms in plan_result.missing_skills)
                )
                continue

            console.print("[yellow]planner could not produce a program and reported no missing skills[/yellow]")
            if plan_result.notes:
                console.print(f"[dim]{plan_result.notes}[/dim]")
            raise typer.Exit(1)
        else:
            console.print(f"[red]could not produce a valid plan after {max_plan_cycles} cycles[/red]")
            raise typer.Exit(1)

        prog = plan_result.program
        # Save plan to the run trace directory
        try:
            plan_dir = run_dir / "plan"
            plan_dir.mkdir(parents=True, exist_ok=True)
            plan_data = prog.model_dump(mode="json", exclude_none=True)
            if goal:
                plan_data.setdefault("metadata", {})["goal"] = goal
            (plan_dir / "plan.yaml").write_text(
                yaml.dump(plan_data, default_flow_style=False, sort_keys=False)
            )
            code = getattr(prog, "code", None)
            if code:
                (plan_dir / "plan.py").write_text(code)
        except Exception:
            pass
        if bridge:
            prog_data = prog.model_dump(mode="json", exclude_none=True) if hasattr(prog, "model_dump") else {}
            bridge.emit_event("program_planned", {"program": prog_data})
            bridge.emit_phase("planner", "complete", f"{prog.step_count} steps")
            # Log the plan code for debugging
            code = getattr(prog, "code", None)
            if code:
                bridge.emit_thinking(f"\n--- Generated Plan ({prog.step_count} steps) ---\n{code}\n---\n")
    elif program is not None:
        if not program.exists():
            console.print(f"[red]program file not found: {program}[/red]")
            raise typer.Exit(2)
        try:
            prog = load_program(program)
            validate_program_against_registry(prog)
        except ProgramValidationError as exc:
            console.print(f"[red]program failed validation: {exc}[/red]")
            raise typer.Exit(2) from exc
    else:
        console.print("[red]provide either --program or --goal[/red]")
        raise typer.Exit(2)

    # Step 3: Confirm the plan (with deny-with-comments loop).
    if bridge:
        # In frontend mode, send confirm request
        prog_data = prog.model_dump(mode="json", exclude_none=True) if hasattr(prog, "model_dump") else {}
        if yes:
            pass  # yolo mode: auto-approve program
        else:
            result = bridge.confirm_program(prog_data)
            if result.get("decision") != "approve":
                if result.get("decision") == "deny" and goal is not None and result.get("comments"):
                    try:
                        plan_result = planner.plan(goal, extra_context=result["comments"], memory_context=memory_context)
                    except Exception as exc:
                        bridge.emit_event("error", {"message": f"re-planning failed: {exc}"}, level="error")
                        raise typer.Exit(1) from exc
                    if plan_result.program is None:
                        bridge.emit_event("error", {"message": "planner could not produce a revised plan"}, level="error")
                        raise typer.Exit(1)
                    prog = plan_result.program
                else:
                    bridge.emit_event("aborted", {"reason": "user cancelled"})
                    raise typer.Exit(1)
    else:
        while True:
            cr = confirm_program(prog, console=console, auto_yes=yes, explain_only=explain_only)
            if explain_only:
                return
            if cr.decision == ConfirmDecision.APPROVE:
                break
            elif cr.decision == ConfirmDecision.DENY_WITH_COMMENTS and goal is not None:
                console.print("[cyan]re-planning with your comments...[/cyan]")
                try:
                    plan_result = planner.plan(goal, extra_context=cr.comments, memory_context=memory_context)
                except Exception as exc:
                    console.print(f"[red]re-planning failed: {exc}[/red]")
                    raise typer.Exit(1) from exc
                if plan_result.program is None:
                    console.print("[red]planner could not produce a revised plan[/red]")
                    raise typer.Exit(1)
                prog = plan_result.program
                continue
            else:
                console.print("[yellow]aborted by user; not executing.[/yellow]")
                raise typer.Exit(1)

    # Build backend
    try:
        if backend == "mock":
            be = make_backend("mock")
        elif backend == "vnc":
            effective_username_env = username_env
            max_res: tuple[int, int] | None = None
            if config and config.exists():
                raw_cfg = yaml.safe_load(config.read_text()) or {}
                vnc_cfg = (raw_cfg.get("backend") or {}).get("vnc") or {}
                if not effective_username_env:
                    effective_username_env = vnc_cfg.get("username_env")
                mr_w = vnc_cfg.get("max_width")
                mr_h = vnc_cfg.get("max_height")
                if mr_w and mr_h:
                    max_res = (int(mr_w), int(mr_h))
            password = os.environ.get(password_env) if password_env else None
            username = os.environ.get(effective_username_env) if effective_username_env else None
            be = make_backend(
                "vnc", host=host, port=port, password=password,
                username=username, max_resolution=max_res,
            )
        else:
            console.print(f"[red]unknown backend {backend!r}[/red]")
            raise typer.Exit(2)
    except Exception as exc:
        console.print(f"[red]backend init failed: {exc}[/red]")
        raise typer.Exit(2) from exc

    # Build LLM gateway for skills that need it (e.g. assert_screen_contains)
    llm_cfg = _load_llm_config(config)
    skill_gateway = make_gateway(llm_cfg) if llm_cfg else None

    # Step 4: Execute with learner retry loop.
    recorder: _ScreenRecorder | None = None
    for attempt in range(1 + max_retries):
        if bridge:
            bridge.emit_event("attempt_started", {"attempt": attempt})
            bridge.emit_phase("executor", "running")

        overlay = make_overlay(prog.name, enabled=(not no_overlay and not frontend_mode))
        overlay.start()

        # In frontend mode, use the bridge's abort event and status callback
        _abort_event = abort_event if frontend_mode else overlay.abort_event
        _abort_event.clear()
        _status_cb = bridge.update_status if bridge else overlay.update_status

        def _exec_event_callback(kind: str, data: dict) -> None:
            if not bridge:
                return
            if kind == "run_trace_dir":
                trace_dir_str = data.get("trace_dir")
                if trace_dir_str:
                    bridge.set_run_trace_dir(Path(trace_dir_str))
            bridge.emit_event(kind, data)

        if isinstance(prog, PythonProgram):
            executor_obj: SequentialExecutor | PythonProgramExecutor = PythonProgramExecutor(
                backend=be,
                llm=skill_gateway,
                traces_root=run_dir / "execution",
                tasks_db=db_path,
                abort_event=_abort_event,
                status_callback=_status_cb,
                event_callback=_exec_event_callback if bridge else None,
            )
        else:
            executor_obj = SequentialExecutor(
                backend=be,
                llm=skill_gateway,
                traces_root=run_dir / "execution",
                tasks_db=db_path,
                abort_event=_abort_event,
                status_callback=_status_cb,
                event_callback=_exec_event_callback if bridge else None,
            )

        try:
            # Start screen recording if requested.
            if record:
                rec_tmp = run_dir / "execution" / f"_recording_attempt_{attempt}.mp4"
                rec_tmp.parent.mkdir(parents=True, exist_ok=True)
                recorder = _ScreenRecorder(rec_tmp, backend=be, fps=record_fps, host=host, port=port)
                recorder.start()

            logging.getLogger(__name__).info(
                "executor starting: program=%s, timeout=%.0fs, step_timeout=%.0fs, attempt=%d",
                prog.name, executor_obj._program_timeout_s, executor_obj._step_timeout_s, attempt + 1,
            )
            result = executor_obj.run(
                prog,
                program_ref=str(program) if program else "<goal>",
                success_criteria=criteria,
                task_id=f"attempt_{attempt}",
            )
            logging.getLogger(__name__).info(
                "executor finished: status=%s, steps=%d, duration=%.1fs, task_id=%s",
                result.status, len(result.steps), result.duration_s, result.task_id,
            )
            if result.error_message:
                logging.getLogger(__name__).error("executor error detail: %s", result.error_message)
        except Exception as exc:
            logging.getLogger(__name__).exception("executor raised unexpected exception")
            console.print(f"[red]unexpected error during execution: {exc}[/red]")
            overlay.stop()
            if recorder:
                recorder.stop()
                recorder = None
            _close_backend(be)
            raise typer.Exit(1) from exc
        finally:
            overlay.stop()

        # Stop recording and move to trace directory.
        recording_path: Path | None = None
        if recorder:
            recording_path = recorder.stop()
            recorder = None
            if recording_path and recording_path.exists():
                final_rec = run_dir / "execution" / result.task_id / "recording.mp4"
                final_rec.parent.mkdir(parents=True, exist_ok=True)
                recording_path.rename(final_rec)
                recording_path = final_rec

        console.print(
            f"[{'green' if result.status in ('success', 'goal_achieved') else 'red'}]"
            f"attempt {attempt + 1}[/] task=[cyan]{result.task_id}[/cyan] "
            f"status=[cyan]{result.status}[/cyan] "
            f"steps=[cyan]{len(result.steps)}[/cyan] "
            f"duration={result.duration_s:.2f}s"
        )
        if bridge:
            bridge.emit_event("task_started", {"task_id": result.task_id})
            exec_status = "complete" if result.status in ("success", "goal_achieved") else "failed"
            summary = f"{result.status} in {result.duration_s:.1f}s"
            if result.error_message:
                summary += f" — {result.error_message}"
            bridge.emit_phase("executor", exec_status, summary)
        console.print(f"trace dir: {run_dir}")
        if recording_path:
            console.print(f"[dim]recording: {recording_path}[/dim]")
        _render_verdict(console, result)

        # Emit evaluator result to bridge
        if bridge and hasattr(result, "goal_verdict") and result.goal_verdict is not None:
            bridge.emit_phase("evaluator", "complete")
            verdict_data = result.goal_verdict.model_dump(mode="json") if hasattr(result.goal_verdict, "model_dump") else {
                "achieved": result.goal_verdict.achieved,
                "summary": result.goal_verdict.summary,
            }
            bridge.emit_event("goal_evaluation", verdict_data)

        # Write run facts to persistent memory.
        if goal:
            try:
                success = result.status in ("success", "goal_achieved")
                fact_cat = "strategy" if success else "failure_mode"
                fact_content = (
                    f"Goal: {goal!r} -> {'SUCCESS' if success else 'FAILED'} "
                    f"({len(result.steps)} steps, {result.duration_s:.1f}s)"
                )
                agent_memory.add_fact(fact_cat, fact_content, result.task_id)
                for step_res in result.steps:
                    agent_memory.add_skill_outcome(
                        skill_id=step_res.skill_id,
                        task_id=result.task_id,
                        success=step_res.status == "success",
                        notes=step_res.error or "",
                    )
            except Exception:
                pass

        is_success = result.status in ("success", "goal_achieved")
        is_aborted = result.status == "aborted"

        # Update run-level metadata
        _update_run_meta(run_dir, status=result.status, attempts=attempt + 1)

        if is_aborted:
            console.print("[yellow]aborted by user[/yellow]")
            _close_backend(be)
            raise typer.Exit(1)

        if is_success:
            if learn_on_succeed and goal is not None:
                console.print("[cyan]analyzing successful trace for optimizations...[/cyan]")
                if bridge:
                    bridge.emit_phase("learner", "running", "analyzing successful trace")
                    bridge.emit_thinking_clear()

                def _success_learner_stream(text: str) -> None:
                    if bridge:
                        bridge.emit_thinking(text + "\n")

                _success_tool_counter = [0]
                def _success_tool_callback(name: str, tool_id: str, args: dict, result: str | None, image_path: str | None = None) -> None:
                    if bridge:
                        if result is None:
                            _success_tool_counter[0] += 1
                            bridge.emit_event("skill_started", {
                                "step_idx": _success_tool_counter[0],
                                "skill_id": name,
                                "inputs": args,
                            })
                        else:
                            data: dict = {
                                "step_idx": _success_tool_counter[0],
                                "skill_id": name,
                                "outputs": result[:200] if result else "",
                                "duration_ms": 0,
                            }
                            if image_path:
                                data["image_path"] = image_path
                            bridge.emit_event("skill_finished", data)

                try:
                    learner = Learner(gateway=gateway)
                    feedback = learner.analyze_success(
                        run_dir / "execution" / result.task_id,
                        program=prog,
                        stream_callback=_success_learner_stream if bridge else None,
                        tool_callback=_success_tool_callback if bridge else None,
                        explorer_context=memory_context,
                        context_usage_callback=_context_usage if bridge else None,
                    )
                    console.print(f"[bold]Learner:[/bold] {feedback.summary}")
                    # Save success learner feedback to trace directory
                    try:
                        fb_out = {
                            "status": "success",
                            "summary": feedback.summary,
                            "suggestions": [{"category": s.category, "description": s.description, "affected_step_idx": s.affected_step_idx} for s in feedback.suggestions] if feedback.suggestions else [],
                            "skill_amendments": [{"skill_id": a.skill_id, "issue": a.issue_description, "change": a.proposed_change} for a in feedback.skill_amendments] if hasattr(feedback, "skill_amendments") and feedback.skill_amendments else [],
                            "new_skill_candidates": [{"proposed_id": c.proposed_id, "description": c.description} for c in feedback.new_skill_candidates] if hasattr(feedback, "new_skill_candidates") and feedback.new_skill_candidates else [],
                        }
                        learner_dir = run_dir / "learner"
                        learner_dir.mkdir(parents=True, exist_ok=True)
                        (learner_dir / "feedback_success.json").write_text(
                            json.dumps(fb_out, indent=2)
                        )
                    except Exception:
                        pass
                    if feedback.suggestions:
                        console.print("[bold cyan]Optimization suggestions:[/bold cyan]")
                        for s in feedback.suggestions:
                            step_ref = f" (step {s.affected_step_idx})" if s.affected_step_idx is not None else ""
                            console.print(f"  - [{s.category}]{step_ref} {s.description}")
                    if feedback.skill_amendments:
                        console.print("[bold magenta]Skill amendments proposed:[/bold magenta]")
                        for a in feedback.skill_amendments:
                            console.print(f"  - [bold]{a.skill_id}[/bold]: {a.issue_description}")
                            console.print(f"    Change: {a.proposed_change}")
                            if a.evidence:
                                console.print(f"    Evidence: {a.evidence}")
                    if feedback.new_skill_candidates:
                        approved = confirm_skills(feedback.new_skill_candidates, console=console, auto_yes=yes)
                        if approved:
                            impl = SyntheticSkillImplementor(gateway=gateway, skills_dir=sk_dir)
                            for cand in approved:
                                try:
                                    ir = impl.synthesize(cand.as_implementor_request())
                                    if ir.ok:
                                        impl.publish(ir.bundle)
                                        console.print(f"    [green]published {cand.proposed_id}[/green]")
                                except Exception as exc:
                                    console.print(f"    [yellow]synthesis error: {exc}[/yellow]")
                    if bridge:
                        bridge.emit_phase("learner", "complete", feedback.summary)
                except Exception as exc:
                    console.print(f"[yellow]learner analysis failed: {exc}[/yellow]")
                    if bridge:
                        bridge.emit_phase("learner", "failed", str(exc))
            break

        # -- FAILURE PATH: enter the learner loop --
        if goal is None:
            console.print("[red]program execution failed[/red]")
            _close_backend(be)
            raise typer.Exit(1)

        if attempt >= max_retries:
            console.print(f"[red]failed after {max_retries + 1} attempt(s); giving up[/red]")
            _close_backend(be)
            raise typer.Exit(1)

        console.print(f"[yellow]attempt {attempt + 1} failed; entering learner loop...[/yellow]")
        if bridge:
            bridge.emit_phase("learner", "running")
            bridge.emit_thinking_clear()

        def _learner_stream(text: str) -> None:
            if bridge:
                bridge.emit_thinking(text + "\n")

        _learner_tool_counter = [0]
        def _learner_tool_callback(name: str, tool_id: str, args: dict, result: str | None, image_path: str | None = None) -> None:
            if bridge:
                if result is None:
                    _learner_tool_counter[0] += 1
                    bridge.emit_event("skill_started", {
                        "step_idx": _learner_tool_counter[0],
                        "skill_id": name,
                        "inputs": args,
                    })
                else:
                    data: dict = {
                        "step_idx": _learner_tool_counter[0],
                        "skill_id": name,
                        "outputs": result[:200] if result else "",
                        "duration_ms": 0,
                    }
                    if image_path:
                        data["image_path"] = image_path
                    bridge.emit_event("skill_finished", data)

        try:
            learner = Learner(gateway=gateway)
            feedback = learner.analyze_failure(
                run_dir / "execution" / result.task_id,
                program=prog,
                stream_callback=_learner_stream if bridge else None,
                tool_callback=_learner_tool_callback if bridge else None,
                explorer_context=memory_context,
                context_usage_callback=_context_usage if bridge else None,
            )
        except Exception as exc:
            console.print(f"[red]learner analysis failed: {exc}[/red]")
            _close_backend(be)
            raise typer.Exit(1) from exc

        console.print(f"[bold]Learner diagnosis:[/bold] {feedback.summary}")
        # Save learner feedback to the trace directory
        try:
            fb_out = {
                "summary": feedback.summary,
                "failure_point": feedback.failure_point,
                "suggestions": [{"category": s.category, "description": s.description, "affected_step_idx": s.affected_step_idx} for s in feedback.suggestions],
                "new_skill_candidates": [{"proposed_id": c.proposed_id, "description": c.description} for c in feedback.new_skill_candidates] if hasattr(feedback, "new_skill_candidates") else [],
                "revised_plan_hints": getattr(feedback, "revised_plan_hints", None),
                "skill_amendments": [{"skill_id": a.skill_id, "issue": a.issue_description, "change": a.proposed_change} for a in feedback.skill_amendments] if hasattr(feedback, "skill_amendments") and feedback.skill_amendments else [],
            }
            learner_dir = run_dir / "learner"
            learner_dir.mkdir(parents=True, exist_ok=True)
            (learner_dir / f"feedback_attempt_{attempt}.json").write_text(
                json.dumps(fb_out, indent=2)
            )
        except Exception:
            pass
        if bridge:
            fb_data = {
                "summary": feedback.summary,
                "failure_point": feedback.failure_point,
                "suggestions": [{"category": s.category, "description": s.description, "affected_step_idx": s.affected_step_idx} for s in feedback.suggestions],
                "new_skill_candidates": [{"proposed_id": c.proposed_id, "description": c.description} for c in feedback.new_skill_candidates] if hasattr(feedback, "new_skill_candidates") else [],
            }
            bridge.emit_event("learner_feedback", fb_data)
            bridge.emit_phase("learner", "complete", feedback.summary)
        if feedback.failure_point:
            console.print(f"[bold red]Failure point:[/bold red] {feedback.failure_point}")
        if feedback.suggestions:
            console.print("[bold cyan]Suggestions:[/bold cyan]")
            for s in feedback.suggestions:
                step_ref = f" (step {s.affected_step_idx})" if s.affected_step_idx is not None else ""
                console.print(f"  - [{s.category}]{step_ref} {s.description}")

        if feedback.skill_amendments:
            console.print("[bold magenta]Skill amendments proposed:[/bold magenta]")
            for a in feedback.skill_amendments:
                console.print(f"  - [bold]{a.skill_id}[/bold]: {a.issue_description}")
                console.print(f"    Change: {a.proposed_change}")
                if a.evidence:
                    console.print(f"    Evidence: {a.evidence}")

        # Handle learner-proposed skills with user approval.
        if feedback.new_skill_candidates:
            if bridge:
                skills_data = [{"proposed_id": c.proposed_id, "description": c.description} for c in feedback.new_skill_candidates]
                if yes:
                    approved = feedback.new_skill_candidates
                else:
                    result = bridge.confirm_skills(skills_data)
                    if result.get("decision") == "approve":
                        approved = feedback.new_skill_candidates
                    else:
                        approved = []
            else:
                approved = confirm_skills(feedback.new_skill_candidates, console=console, auto_yes=yes)
            if approved:
                impl = SyntheticSkillImplementor(gateway=gateway, skills_dir=sk_dir)
                for cand in approved:
                    try:
                        ir = impl.synthesize(cand.as_implementor_request())
                        if ir.ok:
                            impl.publish(ir.bundle)
                            console.print(f"    [green]published {cand.proposed_id}[/green]")
                    except Exception as exc:
                        console.print(f"    [yellow]synthesis error: {exc}[/yellow]")
                librarian.reindex()

        # Build learner feedback context for the planner.
        learner_feedback_parts = [feedback.revised_plan_hints]
        for s in feedback.suggestions:
            learner_feedback_parts.append(f"- [{s.category}] {s.description}")
        learner_feedback_ctx = "\n".join(learner_feedback_parts)

        # Re-plan with learner feedback.
        console.print("[cyan]re-planning with learner feedback...[/cyan]")
        if bridge:
            bridge.emit_phase("planner", "running", "re-planning with learner feedback")
        try:
            plan_result = planner.plan(
                goal,
                memory_context=memory_context,
                learner_feedback=learner_feedback_ctx,
            )
        except Exception as exc:
            console.print(f"[red]re-planning failed: {exc}[/red]")
            _close_backend(be)
            raise typer.Exit(1) from exc

        if plan_result.program is None:
            console.print("[red]planner could not produce a revised plan[/red]")
            _close_backend(be)
            raise typer.Exit(1)
        prog = plan_result.program
        # Save revised plan to traces
        try:
            plan_dir = run_dir / "plan"
            plan_dir.mkdir(parents=True, exist_ok=True)
            plan_data = prog.model_dump(mode="json", exclude_none=True)
            if goal:
                plan_data.setdefault("metadata", {})["goal"] = goal
            (plan_dir / f"plan_attempt_{attempt + 1}.yaml").write_text(
                yaml.dump(plan_data, default_flow_style=False, sort_keys=False)
            )
            code = getattr(prog, "code", None)
            if code:
                (plan_dir / f"plan_attempt_{attempt + 1}.py").write_text(code)
        except Exception:
            pass

        if bridge:
            prog_data = prog.model_dump(mode="json", exclude_none=True) if hasattr(prog, "model_dump") else {}
            bridge.emit_event("program_planned", {"program": prog_data})
            bridge.emit_phase("planner", "complete", f"{prog.step_count} steps (revised)")
            # Log the revised plan code for debugging
            code = getattr(prog, "code", None)
            if code:
                bridge.emit_thinking(f"\n--- Revised Plan ({prog.step_count} steps) ---\n{code}\n---\n")
        # Confirm the new plan (with deny-with-comments).
        if bridge:
            prog_data = prog.model_dump(mode="json", exclude_none=True) if hasattr(prog, "model_dump") else {}
            if yes:
                pass  # yolo mode: auto-approve revised program
            else:
                result = bridge.confirm_program(prog_data)
                if result.get("decision") == "approve":
                    pass  # continue to next attempt
                elif result.get("decision") == "deny" and result.get("comments"):
                    try:
                        plan_result = planner.plan(
                            goal,
                            extra_context=result["comments"],
                            memory_context=memory_context,
                            learner_feedback=learner_feedback_ctx,
                        )
                    except Exception as exc:
                        bridge.emit_event("error", {"message": f"re-planning failed: {exc}"}, level="error")
                        _close_backend(be)
                        raise typer.Exit(1) from exc
                    if plan_result.program is None:
                        bridge.emit_event("error", {"message": "planner could not produce a revised plan"}, level="error")
                        _close_backend(be)
                        raise typer.Exit(1)
                    prog = plan_result.program
                else:
                    bridge.emit_event("aborted", {"reason": "user cancelled re-plan"})
                    _close_backend(be)
                    raise typer.Exit(1)
        else:
            while True:
                cr = confirm_program(prog, console=console, auto_yes=yes)
                if cr.decision == ConfirmDecision.APPROVE:
                    break
                elif cr.decision == ConfirmDecision.DENY_WITH_COMMENTS:
                    console.print("[cyan]re-planning with your comments...[/cyan]")
                    try:
                        plan_result = planner.plan(
                            goal,
                            extra_context=cr.comments,
                            memory_context=memory_context,
                            learner_feedback=learner_feedback_ctx,
                        )
                    except Exception as exc:
                        console.print(f"[red]re-planning failed: {exc}[/red]")
                        _close_backend(be)
                        raise typer.Exit(1) from exc
                    if plan_result.program is None:
                        console.print("[red]planner could not produce a plan[/red]")
                        _close_backend(be)
                        raise typer.Exit(1)
                    prog = plan_result.program
                    continue
                else:
                    console.print("[yellow]aborted by user; not retrying.[/yellow]")
                    _close_backend(be)
                    raise typer.Exit(1)

    _update_run_meta(run_dir, finished=datetime.now(timezone.utc).isoformat())
    _close_backend(be)


# ---------------------------------------------------------------------------
# `daedalus skills list / test`
# ---------------------------------------------------------------------------


@skills_app.command("list")
def cmd_skills_list(
    skills_dir: Optional[Path] = typer.Option(None, "--skills-dir"),
) -> None:
    sk_dir = _resolve_skills_dir(skills_dir)
    _ensure_library_loaded(sk_dir)
    registry = get_registry()
    table = Table(show_header=True, header_style="bold")
    table.add_column("id", style="cyan")
    table.add_column("version")
    table.add_column("hash", style="dim", width=8)
    table.add_column("kind")
    table.add_column("side effects", style="yellow")
    table.add_column("description", overflow="fold", style="dim")
    for entry in sorted(registry, key=lambda e: e.id):
        spec = entry.cls.SPEC
        table.add_row(
            entry.id,
            entry.version.raw,
            entry.content_hash[:8] if entry.content_hash else "—",
            spec.kind,
            ", ".join(spec.side_effects) or "—",
            spec.description.strip().split("\n")[0],
        )
    console.print(table)


@skills_app.command("test")
def cmd_skills_test(
    skill: Optional[str] = typer.Argument(None, help="Skill id; default = all skills"),
    skills_dir: Optional[Path] = typer.Option(None, "--skills-dir"),
) -> None:
    """Replay each skill's tests/*.json fixtures against the MockBackend."""
    sk_dir = _resolve_skills_dir(skills_dir)
    _ensure_library_loaded(sk_dir)
    registry = get_registry()

    targets = [registry.get(skill)] if skill else list(registry)
    failures = 0
    for entry in targets:
        skill_dir = sk_dir / entry.id
        tests_dir = skill_dir / "tests"
        if not tests_dir.is_dir():
            console.print(f"[dim]{entry.id}: no tests/ dir[/dim]")
            continue
        for fixture_path in sorted(tests_dir.glob("*.json")):
            ok, msg = _run_skill_fixture(entry.cls, fixture_path)
            tag = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
            console.print(f"{tag} {entry.id}::{fixture_path.stem}  {msg}")
            if not ok:
                failures += 1
    if failures:
        console.print(f"[red]{failures} failure(s)[/red]")
        raise typer.Exit(1)


@skills_app.command("sync")
def cmd_skills_sync(
    skills_dir: Optional[Path] = typer.Option(None, "--skills-dir"),
) -> None:
    """Regenerate all spec.yaml files from the Python SPEC (Python wins)."""
    import yaml as _yaml
    sk_dir = _resolve_skills_dir(skills_dir)
    _ensure_library_loaded(sk_dir)
    registry = get_registry()
    count = 0
    for entry in sorted(registry, key=lambda e: e.id):
        skill_dir = sk_dir / entry.id
        if not skill_dir.is_dir():
            continue
        generated = entry.cls.SPEC.to_yaml_dict()
        (skill_dir / "spec.yaml").write_text(
            _yaml.safe_dump(generated, default_flow_style=False, sort_keys=False, allow_unicode=True)
        )
        count += 1
    console.print(f"[green]synced {count} spec.yaml file(s)[/green]")


def _run_skill_fixture(skill_cls, fixture_path: Path) -> tuple[bool, str]:  # type: ignore[no-untyped-def]
    """Execute a skill against the MockBackend using a fixture, return (ok, message)."""
    fixture = json.loads(fixture_path.read_text())
    inputs = fixture["inputs"]
    expected_output = fixture.get("expected_output", {})
    expected_events = fixture.get("expected_events", [])
    ignore_keys = set(fixture.get("ignore_output_keys", []))

    from daedalus.backends.mock import MockBackend

    backend = MockBackend()
    backend.connect()

    # Minimal in-memory tracer that swallows events. Skills that depend on
    # the trace recorder still need a real TraceRecorder; we use a temp dir.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        tracer = TraceRecorder(traces_root=tmpdir, db_path=tmpdir / "tasks.db", task_name="fixture-test")
        ts = TaskState(tmpdir / "tasks.db", tracer.task_id)
        ctx = ExecutionContext(
            task_id=tracer.task_id,
            backend=backend,
            task_state=ts,
            tracer=tracer,
            coordinate_scale=compute_coordinate_scale(backend.size[0]) if hasattr(backend, "size") else 1.0,
        )
        try:
            inp = skill_cls.Inputs.model_validate(inputs)
            instance = skill_cls()
            out = instance.run(inp, ctx)
            out_dict = out.model_dump(mode="json") if hasattr(out, "model_dump") else dict(out)
        except Exception as exc:
            tracer.finish("failed")
            return False, f"raised {type(exc).__name__}: {exc}"
        tracer.finish("success")

    # Compare outputs (after stripping ignored keys)
    cmp_actual = {k: v for k, v in out_dict.items() if k not in ignore_keys}
    cmp_expected = {k: v for k, v in expected_output.items() if k not in ignore_keys}
    if cmp_actual != cmp_expected:
        return False, f"output mismatch: expected {cmp_expected}, got {cmp_actual}"

    # Compare events. Each expected_event matches a backend event by op + (subset of args)
    actual_events = [{"op": e.op, "args": e.args} for e in backend.events]
    search_start = 0
    for i, want in enumerate(expected_events):
        want_op = want["op"]
        want_args = want.get("args", {})
        matched = False
        for j, actual in enumerate(actual_events[search_start:], start=search_start):
            if actual["op"] != want_op:
                continue
            if all(actual["args"].get(k) == v for k, v in want_args.items()):
                matched = True
                search_start = j + 1
                break
        if not matched:
            return False, f"expected event[{i}] {want} not found in events from index {search_start}"
    return True, "ok"


# ---------------------------------------------------------------------------
# `daedalus traces ...`
# ---------------------------------------------------------------------------


@traces_app.command("list")
def cmd_traces_list(
    traces_dir: Optional[Path] = typer.Option(None, "--traces-dir"),
    tasks_db: Optional[Path] = typer.Option(None, "--tasks-db"),
    limit: int = typer.Option(50, "--limit"),
) -> None:
    tr_dir = _resolve_traces_dir(traces_dir)
    if not tr_dir.is_dir():
        console.print("[dim]no traces yet[/dim]")
        return

    # List run directories (t_* pattern) sorted by name (newest first)
    run_dirs = sorted(
        (d for d in tr_dir.iterdir() if d.is_dir() and d.name.startswith("t_")),
        key=lambda d: d.name,
        reverse=True,
    )[:limit]

    if not run_dirs:
        console.print("[dim]no traces yet[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    for col in ("run_id", "goal", "mode", "status", "started"):
        table.add_column(col)
    for d in run_dirs:
        meta_path = d / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}
        else:
            meta = {}
        goal_text = (meta.get("goal") or "")[:60]
        table.add_row(
            d.name,
            goal_text,
            meta.get("mode", "?"),
            meta.get("status", "?"),
            meta.get("started", "?"),
        )
    console.print(table)


@traces_app.command("show")
def cmd_traces_show(
    task_id: str = typer.Argument(..., help="Task id (e.g. t_20260502_010203_abcd1234)"),
    traces_dir: Optional[Path] = typer.Option(None, "--traces-dir"),
    full: bool = typer.Option(False, "--full", help="Print every event verbatim."),
) -> None:
    tr_dir = _resolve_traces_dir(traces_dir)
    task_dir = tr_dir / task_id
    if not task_dir.is_dir():
        console.print(f"[red]no trace dir at {task_dir}[/red]")
        raise typer.Exit(1)
    meta_path = task_dir / "meta.json"
    if meta_path.exists():
        console.print("[bold]Run metadata:[/bold]")
        console.print(yaml.safe_dump(json.loads(meta_path.read_text()), sort_keys=False))

    # Show goal
    goal_path = task_dir / "goal.txt"
    if goal_path.exists():
        console.print(f"[bold]Goal:[/bold] {goal_path.read_text().strip()}")

    # Show explorer info
    explorer_dir = task_dir / "explorer"
    if explorer_dir.is_dir():
        console.print("\n[bold cyan]Explorer:[/bold cyan]")
        obs_path = explorer_dir / "observations.md"
        if obs_path.exists():
            console.print(f"  observations: {obs_path}")
        events_path = explorer_dir / "events.jsonl"
        if events_path.exists():
            n_events = sum(1 for _ in events_path.open("r", encoding="utf-8") if _.strip())
            console.print(f"  events: {n_events}")
        screens_dir = explorer_dir / "screens"
        if screens_dir.is_dir():
            n = len(list(screens_dir.glob("*.png")))
            console.print(f"  screenshots: {n}")

    # Show plan info
    plan_dir = task_dir / "plan"
    if plan_dir.is_dir():
        console.print("\n[bold cyan]Plan:[/bold cyan]")
        for f in sorted(plan_dir.iterdir()):
            console.print(f"  {f.name}")

    # Show execution attempts
    exec_dir = task_dir / "execution"
    if exec_dir.is_dir():
        attempts = sorted(d for d in exec_dir.iterdir() if d.is_dir() and d.name.startswith("attempt_"))
        for attempt_dir in attempts:
            console.print(f"\n[bold cyan]Execution {attempt_dir.name}:[/bold cyan]")
            attempt_meta = attempt_dir / "meta.json"
            if attempt_meta.exists():
                console.print(yaml.safe_dump(json.loads(attempt_meta.read_text()), sort_keys=False))
            events_path = attempt_dir / "events.jsonl"
            if events_path.exists():
                for i, line in enumerate(events_path.open("r", encoding="utf-8")):
                    line = line.rstrip()
                    if not line:
                        continue
                    evt = json.loads(line)
                    if full:
                        console.print(line)
                    else:
                        console.print(
                            f"  [cyan]{i:>3}[/cyan] {evt['ts']} [bold]{evt['kind']}[/bold] "
                            f"{json.dumps(evt.get('data', {}), default=str)[:160]}"
                        )
            screens_dir = attempt_dir / "screens"
            if screens_dir.is_dir():
                n = len(list(screens_dir.glob("*.png")))
                console.print(f"  screenshots: {n}")
            rec = attempt_dir / "recording.mp4"
            if rec.exists():
                console.print(f"  recording: {rec}")

    # Show learner info
    learner_dir = task_dir / "learner"
    if learner_dir.is_dir():
        console.print("\n[bold cyan]Learner:[/bold cyan]")
        for f in sorted(learner_dir.iterdir()):
            console.print(f"  {f.name}")

    # Fallback: if this is an old-format trace (has events.jsonl at root)
    events_path = task_dir / "events.jsonl"
    if events_path.exists():
        console.print("\n[bold]Events (legacy format):[/bold]")
        for i, line in enumerate(events_path.open("r", encoding="utf-8")):
            line = line.rstrip()
            if not line:
                continue
            evt = json.loads(line)
            if full:
                console.print(line)
            else:
                console.print(
                    f"[cyan]{i:>3}[/cyan] {evt['ts']} [bold]{evt['kind']}[/bold] "
                    f"{json.dumps(evt.get('data', {}), default=str)[:160]}"
                )
        screens_dir = task_dir / "screens"
        if screens_dir.is_dir():
            n = len(list(screens_dir.glob("*.png")))
            console.print(f"[dim]screenshots: {n} ({screens_dir})[/dim]")


# ---------------------------------------------------------------------------
# `daedalus verify-litellm` (delegates to the bash IOC scanner)
# ---------------------------------------------------------------------------


@app.command("plan")
def cmd_plan(
    goal: str = typer.Argument(..., help="Free-form description of what to do."),
    config: Path | None = typer.Option(None, "--config", "-c", help="YAML config with llm.roles."),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write the program to this YAML."),
    skills_dir: Path | None = typer.Option(None, "--skills-dir"),
    extra: str | None = typer.Option(None, "--extra", help="Extra context for the planner."),
    screen_width: int = typer.Option(None, "--screen-width", help="Screen width in pixels (required if no --config)."),
    screen_height: int = typer.Option(None, "--screen-height", help="Screen height in pixels (required if no --config)."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Ask the LLM planner to draft a program for ``goal``."""
    _setup_logging(verbose)
    sk_dir = _resolve_skills_dir(skills_dir)
    _ensure_library_loaded(sk_dir)
    llm_cfg = _load_llm_config(config)
    if llm_cfg is None:
        console.print(
            "[red]no LLM config found. Pass --config path/to/config.yaml with an llm.roles mapping.[/red]"
        )
        raise typer.Exit(2)
    gateway = make_gateway(llm_cfg)
    if gateway is None:
        console.print("[red]could not build gateway from config[/red]")
        raise typer.Exit(2)

    librarian = Librarian()
    librarian.reindex()
    host_os = _load_host_os(config)

    # Resolve screen size: CLI flags win, then config, then error.
    sw, sh = screen_width, screen_height
    if (sw is None or sh is None) and config and config.exists():
        raw_cfg = yaml.safe_load(config.read_text()) or {}
        vnc_cfg = (raw_cfg.get("backend") or {}).get("vnc") or {}
        exec_cfg = raw_cfg.get("executor") or {}
        sw = sw or int(vnc_cfg.get("max_width") or exec_cfg.get("default_screen_width") or 0) or None
        sh = sh or int(vnc_cfg.get("max_height") or exec_cfg.get("default_screen_height") or 0) or None
    if sw is None or sh is None:
        console.print(
            "[red]screen dimensions required. Pass --screen-width/--screen-height "
            "or provide them in --config.[/red]"
        )
        raise typer.Exit(2)

    planner = Planner(
        gateway=gateway,
        librarian=librarian,
        screen_size=(sw, sh),
        host_os=host_os,
    )
    try:
        result = planner.plan(goal, extra_context=extra)
    except Exception as exc:
        console.print(f"[red]planning failed: {exc}[/red]")
        raise typer.Exit(1) from exc

    if result.program is not None:
        from daedalus.ui.confirm import render_program

        render_program(console, result.program)
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(yaml.safe_dump(result.program.model_dump(exclude_none=True), sort_keys=False))
            console.print(f"[green]wrote program to {out}[/green]")
    else:
        console.print("[yellow]planner could not produce a program with the current skill library.[/yellow]")
    if result.missing_skills:
        console.print("[bold]Missing skills the planner asked for:[/bold]")
        for ms in result.missing_skills:
            console.print(f"  - [cyan]{ms.proposed_id}[/cyan]: {ms.description}")
            console.print(f"    rationale: [dim]{ms.rationale}[/dim]")
    if result.notes:
        console.print(f"[dim]notes: {result.notes}[/dim]")


@app.command("teach")
def cmd_teach(
    task_ids: list[str] = typer.Argument(None, help="Task ids to analyse. Default: latest 5."),
    config: Path | None = typer.Option(None, "--config", "-c", help="YAML config with llm.roles."),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write the report JSON here."),
    traces_dir: Path | None = typer.Option(None, "--traces-dir"),
    tasks_db: Path | None = typer.Option(None, "--tasks-db"),
    heuristics_only: bool = typer.Option(False, "--heuristics-only", help="Skip the LLM step."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Analyse recent traces and (optionally) ask the Learner LLM for proposals."""
    _setup_logging(verbose)
    tr_dir = _resolve_traces_dir(traces_dir)
    db_path = _resolve_db(tasks_db)

    if not task_ids:
        recent = list_traces(db_path, limit=5)
        if not recent:
            console.print("[yellow]no traces found yet; run a program first[/yellow]")
            raise typer.Exit(0)
        task_ids = [r["task_id"] for r in recent]

    task_dirs = [tr_dir / tid for tid in task_ids]
    missing = [d for d in task_dirs if not d.is_dir()]
    if missing:
        console.print(f"[red]missing trace dirs: {missing}[/red]")
        raise typer.Exit(2)

    # Collect execution attempt directories from the new nested structure
    execution_dirs: list[Path] = []
    for td in task_dirs:
        exec_dir = td / "execution"
        if exec_dir.is_dir():
            for attempt_dir in sorted(exec_dir.iterdir()):
                if attempt_dir.is_dir() and attempt_dir.name.startswith("attempt_"):
                    execution_dirs.append(attempt_dir)
        elif (td / "events.jsonl").exists():
            execution_dirs.append(td)

    from daedalus.learner.analysis import analyze_traces

    findings = analyze_traces(execution_dirs if execution_dirs else task_dirs)
    console.print(f"[bold]traces analysed:[/bold] {findings.traces_analyzed}")
    if findings.notes:
        console.print("[bold]heuristic notes:[/bold]")
        for n in findings.notes:
            console.print(f"  - {n}")

    if heuristics_only:
        if out is not None:
            from dataclasses import asdict

            out.write_text(json.dumps(_findings_to_jsonable(findings), indent=2))
            console.print(f"[green]wrote heuristics to {out}[/green]")
        return

    llm_cfg = _load_llm_config(config)
    if llm_cfg is None:
        console.print(
            "[red]no LLM config found. Pass --config path/to/config.yaml or use --heuristics-only.[/red]"
        )
        raise typer.Exit(2)
    gateway = make_gateway(llm_cfg)
    if gateway is None:
        console.print("[red]could not build gateway from config[/red]")
        raise typer.Exit(2)
    learner = Learner(gateway=gateway)
    try:
        report = learner.learn_from_findings(findings)
    except Exception as exc:
        console.print(f"[red]learner LLM call failed: {exc}[/red]")
        raise typer.Exit(1) from exc

    console.print(f"\n[bold]Learner summary:[/bold] {report.summary}")
    if report.efficiency_wins:
        console.print("\n[bold cyan]Efficiency wins:[/bold cyan]")
        for w in report.efficiency_wins:
            console.print(f"  - {w.description}\n    -> {w.recommendation}")
    if report.new_skill_candidates:
        console.print("\n[bold cyan]New skill candidates:[/bold cyan]")
        for c in report.new_skill_candidates:
            console.print(f"  - [cyan]{c.proposed_id}[/cyan]: {c.description} ({c.occurrences}x)")
    if report.failure_proposals:
        console.print("\n[bold cyan]Failure proposals:[/bold cyan]")
        for f in report.failure_proposals:
            console.print(f"  - {f.affected_skill}: {f.proposal}")

    if out is not None:
        out.write_text(report.model_dump_json(indent=2))
        console.print(f"\n[green]wrote report to {out}[/green]")


def _findings_to_jsonable(findings) -> dict:  # type: ignore[no-untyped-def]
    from dataclasses import asdict

    return {
        "traces_analyzed": findings.traces_analyzed,
        "status_counts": dict(findings.overall_status_counts),
        "timings": {sid: asdict(t) for sid, t in findings.timings.items()},
        "failures": {sid: asdict(f) for sid, f in findings.failures.items()},
        "repeated_subsequences": [
            {"skills": list(ng.skills), "occurrences": ng.occurrences, "in_traces": ng.in_traces}
            for ng in findings.repeated_subsequences
        ],
        "notes": findings.notes,
    }


@app.command("implement")
def cmd_implement(
    spec: Path = typer.Argument(..., help="Path to JSON file describing the missing skill."),
    config: Path | None = typer.Option(None, "--config", "-c"),
    skills_dir: Path | None = typer.Option(None, "--skills-dir"),
    publish: bool = typer.Option(
        False,
        "--publish/--no-publish",
        help="If true and synthesis succeeds, install the skill into skills_dir.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Synthesize a new skill from a JSON spec, lint it, run its fixtures, optionally publish."""
    _setup_logging(verbose)
    sk_dir = _resolve_skills_dir(skills_dir)
    _ensure_library_loaded(sk_dir)

    llm_cfg = _load_llm_config(config)
    if llm_cfg is None:
        console.print(
            "[red]no LLM config found. Pass --config path/to/config.yaml.[/red]"
        )
        raise typer.Exit(2)
    gateway = make_gateway(llm_cfg)
    if gateway is None:
        console.print("[red]could not build gateway from config[/red]")
        raise typer.Exit(2)

    try:
        req_data = json.loads(spec.read_text())
        req = ImplementorRequest.model_validate(req_data)
    except Exception as exc:
        console.print(f"[red]bad spec file: {exc}[/red]")
        raise typer.Exit(2) from exc

    impl = SyntheticSkillImplementor(gateway=gateway, skills_dir=sk_dir)
    try:
        result = impl.synthesize(req)
    except Exception as exc:
        console.print(f"[red]synthesis failed: {exc}[/red]")
        raise typer.Exit(1) from exc

    if not result.ok:
        console.print(f"[red]synthesis did not produce a clean skill[/red]")
        if result.violations:
            console.print("[bold]safety violations:[/bold]")
            for v in result.violations:
                console.print(f"  line {v.lineno}: {v.rule}: {v.detail}")
        if result.test_failures:
            console.print("[bold]test failures:[/bold]")
            for f in result.test_failures:
                console.print(f"  - {f}")
        raise typer.Exit(1)

    bundle = result.bundle
    console.print(f"[green]green build:[/green] sandbox at {bundle.sandbox_dir}")
    if publish:
        sid = impl.publish(bundle)
        console.print(f"[green]published[/green] skill [cyan]{sid}[/cyan] to {sk_dir / sid}")
    else:
        console.print("[yellow]not publishing (use --publish to install).[/yellow]")


@app.command("shell")
def cmd_shell(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file to load on start."),
) -> None:
    """Interactive Daedalus terminal (REPL)."""
    from daedalus.repl.repl import DaedalusREPL

    repl = DaedalusREPL(config=config)
    repl.run()


@app.command("verify-litellm")
def cmd_verify_litellm() -> None:
    """Run scripts/verify_litellm.sh to check for the March 2026 IOC artifacts."""
    import subprocess

    script = Path(__file__).parent.parent.parent / "scripts" / "verify_litellm.sh"
    if not script.exists():
        # When installed as a wheel scripts/ may not ship; fall back to a hint.
        console.print(
            "[yellow]scripts/verify_litellm.sh not found. Clone the repo and run it directly.[/yellow]"
        )
        raise typer.Exit(2)
    rc = subprocess.call(["bash", str(script)])
    raise typer.Exit(rc)


@app.command("archive")
def cmd_archive(
    skills_dir: Optional[Path] = typer.Option(None, "--skills-dir"),
) -> None:
    """Archive generated traces and learned skills, resetting the agent for testing."""
    import datetime
    import shutil

    sk_dir = _resolve_skills_dir(skills_dir)
    project_root = sk_dir.parent

    core_skills = {
        "assert_screen_contains", "click_all", "click_element", "mouse",
        "locate_element", "locate_elements", "monitor_text_region",
        "populate_store_from_analysis", "store_query", "tick_counter",
        "type_shortcut", "type_text", "type_text_secret", "view_screen",
        "vision_query", "wait",
    }

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = project_root / "backup" / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    (backup_dir / "skills").mkdir(exist_ok=True)
    (backup_dir / "traces").mkdir(exist_ok=True)
    (backup_dir / "memory").mkdir(exist_ok=True)
    (backup_dir / "sandbox").mkdir(exist_ok=True)

    learned_count = 0
    if sk_dir.exists():
        for skill_path in sorted(sk_dir.iterdir()):
            if not skill_path.is_dir():
                continue
            if skill_path.name not in core_skills:
                shutil.copytree(skill_path, backup_dir / "skills" / skill_path.name)
                shutil.rmtree(skill_path)
                learned_count += 1
    console.print(f"  archived [cyan]{learned_count}[/cyan] learned skill(s)")

    traces_dir = project_root / "traces"
    trace_count = 0
    if traces_dir.exists():
        for td in sorted(traces_dir.iterdir()):
            if td.is_dir():
                shutil.move(str(td), str(backup_dir / "traces" / td.name))
                trace_count += 1
    console.print(f"  archived [cyan]{trace_count}[/cyan] trace(s)")

    for db_name in ("tasks.db", "memory.db"):
        db_path = project_root / db_name
        if db_path.exists():
            shutil.copy2(db_path, backup_dir / "memory" / db_name)
            db_path.unlink()
            console.print(f"  archived database: [cyan]{db_name}[/cyan]")

    sandbox_dir = project_root / ".daedalus" / "implementor_sandbox"
    if sandbox_dir.exists() and any(sandbox_dir.iterdir()):
        shutil.copytree(sandbox_dir, backup_dir / "sandbox" / "implementor_sandbox")
        shutil.rmtree(sandbox_dir)
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        console.print("  archived implementor sandbox")

    console.print(f"\n[green]Backup complete:[/green] {backup_dir}")
    console.print("[green]Agent is now reset to core skills only.[/green]")


@app.command("restore")
def cmd_restore(
    backup_path: Path = typer.Argument(..., help="Path to backup directory (e.g. backup/20260504_120000)"),
    skills_dir: Optional[Path] = typer.Option(None, "--skills-dir"),
) -> None:
    """Restore learned skills, traces, and databases from a backup."""
    import shutil

    sk_dir = _resolve_skills_dir(skills_dir)
    project_root = sk_dir.parent

    if not backup_path.exists():
        console.print(f"[red]backup directory not found: {backup_path}[/red]")
        backups = project_root / "backup"
        if backups.exists():
            console.print("\nAvailable backups:")
            for d in sorted(backups.iterdir(), reverse=True):
                if d.is_dir():
                    console.print(f"  {d}")
        raise typer.Exit(1)

    skills_backup = backup_path / "skills"
    if skills_backup.exists():
        skill_count = 0
        for skill_path in sorted(skills_backup.iterdir()):
            if not skill_path.is_dir():
                continue
            target = sk_dir / skill_path.name
            if target.exists():
                console.print(f"  skipping skill {skill_path.name} (already exists)")
            else:
                shutil.copytree(skill_path, target)
                skill_count += 1
        console.print(f"  restored [cyan]{skill_count}[/cyan] skill(s)")

    traces_backup = backup_path / "traces"
    traces_dir = project_root / "traces"
    if traces_backup.exists():
        traces_dir.mkdir(exist_ok=True)
        trace_count = 0
        for td in sorted(traces_backup.iterdir()):
            if not td.is_dir():
                continue
            target = traces_dir / td.name
            if target.exists():
                continue
            shutil.copytree(td, target)
            trace_count += 1
        console.print(f"  restored [cyan]{trace_count}[/cyan] trace(s)")

    memory_backup = backup_path / "memory"
    if memory_backup.exists():
        for db_name in ("tasks.db", "memory.db"):
            src = memory_backup / db_name
            if src.exists():
                dest = project_root / db_name
                if dest.exists():
                    console.print(f"  skipping {db_name} (already exists)")
                else:
                    shutil.copy2(src, dest)
                    console.print(f"  restored database: [cyan]{db_name}[/cyan]")

    sandbox_backup = backup_path / "sandbox" / "implementor_sandbox"
    if sandbox_backup.exists():
        target = project_root / ".daedalus" / "implementor_sandbox"
        if target.exists() and any(target.iterdir()):
            console.print("  skipping sandbox (already has content)")
        else:
            target.mkdir(parents=True, exist_ok=True)
            for item in sandbox_backup.iterdir():
                if item.is_dir():
                    shutil.copytree(item, target / item.name)
            console.print("  restored implementor sandbox")

    console.print(f"\n[green]Restore complete from:[/green] {backup_path}")


if __name__ == "__main__":
    app()
