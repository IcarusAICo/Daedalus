"""Main REPL loop for the Daedalus interactive terminal."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console

from daedalus.repl.banner import print_banner
from daedalus.repl import commands as cmd


class DaedalusREPL:
    """Interactive terminal for Daedalus."""

    def __init__(self, config: Path | None = None) -> None:
        self._console = Console()
        self._session: dict[str, Any] = {
            "max_retries": 3,
            "auto_yes": False,
            "backend": "vnc",
            "no_overlay": False,
            "learn_on_succeed": False,
        }
        self._config_path = config
        if config:
            self._session["config_path"] = str(config)
        self._skills_dir = self._resolve_skills_dir()
        self._traces_dir = self._skills_dir.parent / "traces"
        self._memory_db = self._skills_dir.parent / "memory.db"
        self._history_path = Path.home() / ".daedalus" / "history"
        self._history_path.parent.mkdir(parents=True, exist_ok=True)

    def _resolve_skills_dir(self) -> Path:
        candidates = [Path("skills"), Path(__file__).parent.parent.parent.parent / "skills"]
        for c in candidates:
            if c.exists():
                return c.resolve()
        return Path("skills").resolve()

    def run(self) -> None:
        """Main entry point: start the REPL loop."""
        print_banner(self._console)

        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.history import FileHistory
            from prompt_toolkit.completion import WordCompleter

            completions = self._build_completions()
            completer = WordCompleter(completions, ignore_case=True)
            session: PromptSession = PromptSession(
                history=FileHistory(str(self._history_path)),
                completer=completer,
            )
            use_prompt_toolkit = True
        except ImportError:
            use_prompt_toolkit = False
            session = None  # type: ignore[assignment]

        while True:
            try:
                if use_prompt_toolkit:
                    user_input = session.prompt("daedalus> ").strip()
                else:
                    try:
                        user_input = input("daedalus> ").strip()
                    except EOFError:
                        break
            except KeyboardInterrupt:
                self._console.print("\n[dim]Use /quit to exit.[/dim]")
                continue

            if not user_input:
                continue

            if user_input.startswith("/"):
                should_quit = self._handle_command(user_input)
                if should_quit:
                    break
            else:
                self._handle_goal(user_input)

    def _build_completions(self) -> list[str]:
        """Build list of completions for prompt_toolkit."""
        slash_commands = [
            "/help", "/skills", "/traces", "/config", "/set",
            "/archive", "/restore", "/memory", "/goal", "/run", "/quit",
        ]
        skill_names: list[str] = []
        try:
            from daedalus.library import load_library
            from daedalus.core.registry import get_registry
            load_library(self._skills_dir)
            for entry in get_registry():
                skill_names.append(entry.id)
        except Exception:
            pass
        return slash_commands + skill_names

    def _handle_command(self, raw: str) -> bool:
        """Dispatch a slash command. Returns True if REPL should exit."""
        parts = raw.split(None, 1)
        command = parts[0].lower()
        args_str = parts[1] if len(parts) > 1 else ""
        args = args_str.split() if args_str else []

        kwargs = {
            "console": self._console,
            "args": args,
            "session": self._session,
            "skills_dir": self._skills_dir,
            "traces_dir": self._traces_dir,
            "memory_db": self._memory_db,
        }

        if command == "/quit" or command == "/exit":
            self._console.print("[dim]goodbye[/dim]")
            return True
        elif command == "/help":
            cmd.cmd_help(**kwargs)
        elif command == "/skills":
            cmd.cmd_skills(**kwargs)
        elif command == "/traces":
            cmd.cmd_traces(**kwargs)
        elif command == "/set":
            cmd.cmd_set(**kwargs)
        elif command == "/config":
            cmd.cmd_config(**kwargs)
        elif command == "/archive":
            cmd.cmd_archive(**kwargs)
        elif command == "/restore":
            cmd.cmd_restore(**kwargs)
        elif command == "/memory":
            cmd.cmd_memory(**kwargs)
        elif command == "/goal":
            if args_str:
                self._handle_goal(args_str)
            else:
                self._console.print("[yellow]Usage: /goal <text>[/yellow]")
        elif command == "/run":
            if args:
                self._handle_run_program(args[0])
            else:
                self._console.print("[yellow]Usage: /run <program.yaml>[/yellow]")
        else:
            self._console.print(f"[yellow]Unknown command: {command}. Type /help for usage.[/yellow]")

        return False

    def _handle_goal(self, goal: str) -> None:
        """Plan and execute a goal through the full lifecycle."""
        config_path = self._session.get("config_path")
        if not config_path:
            self._console.print(
                "[red]No config loaded. Use /config <path> first "
                "(e.g. /config config.mac.yaml)[/red]"
            )
            return

        config = Path(config_path)
        if not config.exists():
            self._console.print(f"[red]Config file not found: {config}[/red]")
            return

        self._console.print(f"[cyan]Goal:[/cyan] {goal}")
        self._console.print("[cyan]planning...[/cyan]")

        try:
            raw_cfg = yaml.safe_load(config.read_text()) or {}
        except Exception as exc:
            self._console.print(f"[red]Failed to parse config: {exc}[/red]")
            return

        try:
            from daedalus.library import Librarian, load_library
            from daedalus.llm.gateway import LLMConfig, make_gateway
            from daedalus.planner import Planner
            from daedalus.memory import AgentMemory
            from daedalus.backends import make_backend
            from daedalus.executor.dsl import PythonProgram
            from daedalus.executor.program_executor import PythonProgramExecutor
            from daedalus.executor.runner import SequentialExecutor
            from daedalus.ui.confirm import ConfirmDecision, confirm_program

            load_library(self._skills_dir)
            librarian = Librarian()
            librarian.reindex()

            llm_roles = (raw_cfg.get("llm") or {}).get("roles")
            if not llm_roles:
                self._console.print("[red]Config missing llm.roles[/red]")
                return

            llm_cfg = LLMConfig(
                roles=llm_roles,
                aws_region=(raw_cfg.get("llm") or {}).get("aws_region"),
                request_timeout_s=(raw_cfg.get("llm") or {}).get("request_timeout_s", 120),
                max_retries=(raw_cfg.get("llm") or {}).get("max_retries", 2),
            )
            gateway = make_gateway(llm_cfg)
            if gateway is None:
                self._console.print("[red]Could not build LLM gateway[/red]")
                return

            vnc_cfg = (raw_cfg.get("backend") or {}).get("vnc") or {}
            exec_cfg = raw_cfg.get("executor") or {}
            plan_w = int(vnc_cfg.get("max_width") or exec_cfg.get("default_screen_width") or 0) or None
            plan_h = int(vnc_cfg.get("max_height") or exec_cfg.get("default_screen_height") or 0) or None
            if not plan_w or not plan_h:
                self._console.print("[red]Cannot determine screen size from config[/red]")
                return

            host_os = (raw_cfg.get("backend") or {}).get("host_os", "unknown")
            planner = Planner(
                gateway=gateway, librarian=librarian,
                host_os=host_os, screen_size=(plan_w, plan_h),
            )

            memory_context = None
            if self._memory_db.exists():
                try:
                    mem = AgentMemory(self._memory_db)
                    facts = mem.recall(goal, limit=5)
                    if facts:
                        memory_context = "\n".join(f"- [{f.category}] {f.content}" for f in facts)
                except Exception:
                    pass

            plan_result = planner.plan(goal, memory_context=memory_context)
            if plan_result.program is None:
                self._console.print("[red]Planner could not produce a program.[/red]")
                if plan_result.notes:
                    self._console.print(f"[dim]{plan_result.notes}[/dim]")
                return

            prog = plan_result.program

            cr = confirm_program(prog, console=self._console, auto_yes=self._session.get("auto_yes", False))
            if cr.decision != ConfirmDecision.APPROVE:
                if cr.decision == ConfirmDecision.DENY_WITH_COMMENTS:
                    self._console.print("[cyan]Re-planning with feedback...[/cyan]")
                    plan_result = planner.plan(goal, extra_context=cr.comments, memory_context=memory_context)
                    if plan_result.program is None:
                        self._console.print("[red]Re-planning failed.[/red]")
                        return
                    prog = plan_result.program
                    cr = confirm_program(prog, console=self._console)
                    if cr.decision != ConfirmDecision.APPROVE:
                        self._console.print("[yellow]Cancelled.[/yellow]")
                        return
                else:
                    self._console.print("[yellow]Cancelled.[/yellow]")
                    return

            self._console.print("[cyan]executing...[/cyan]")

            backend_kind = self._session.get("backend", "vnc")
            if backend_kind == "vnc":
                host = vnc_cfg.get("host", "127.0.0.1")
                port = int(vnc_cfg.get("port", 5900))
                pw_env = vnc_cfg.get("password_env")
                password = os.environ.get(pw_env) if pw_env else None
                user_env = vnc_cfg.get("username_env")
                username = os.environ.get(user_env) if user_env else None
                max_res = None
                mw, mh = vnc_cfg.get("max_width"), vnc_cfg.get("max_height")
                if mw and mh:
                    max_res = (int(mw), int(mh))
                be = make_backend("vnc", host=host, port=port, password=password,
                                  username=username, max_resolution=max_res)
            else:
                be = make_backend("mock")

            abort_event = threading.Event()
            db_path = self._skills_dir.parent / "tasks.db"

            if isinstance(prog, PythonProgram):
                executor = PythonProgramExecutor(
                    backend=be, llm=gateway,
                    traces_root=self._traces_dir, tasks_db=db_path,
                    abort_event=abort_event,
                )
            else:
                executor = SequentialExecutor(
                    backend=be, llm=gateway,
                    traces_root=self._traces_dir, tasks_db=db_path,
                    abort_event=abort_event,
                )

            result = executor.run(prog, program_ref="<repl-goal>")

            status_style = "green" if result.status in ("success", "goal_achieved") else "red"
            self._console.print(
                f"[{status_style}]Result:[/{status_style}] {result.status} "
                f"({len(result.steps)} steps, {result.duration_s:.1f}s) "
                f"task={result.task_id}"
            )

        except KeyboardInterrupt:
            self._console.print("\n[yellow]Execution interrupted.[/yellow]")
        except Exception as exc:
            self._console.print(f"[red]Error: {exc}[/red]")

    def _handle_run_program(self, program_path: str) -> None:
        """Run a YAML program file directly."""
        path = Path(program_path)
        if not path.exists():
            self._console.print(f"[red]Program file not found: {path}[/red]")
            return

        try:
            from daedalus.executor.dsl import load_program, validate_program_against_registry
            from daedalus.library import load_library

            load_library(self._skills_dir)
            prog = load_program(path)
            validate_program_against_registry(prog)
            self._console.print(f"[green]Loaded program:[/green] {prog.name} ({prog.step_count} steps)")

            from daedalus.ui.confirm import ConfirmDecision, confirm_program
            cr = confirm_program(prog, console=self._console)
            if cr.decision != ConfirmDecision.APPROVE:
                self._console.print("[yellow]Cancelled.[/yellow]")
                return

            from daedalus.backends import make_backend
            from daedalus.executor.runner import SequentialExecutor

            be = make_backend("mock")
            db_path = self._skills_dir.parent / "tasks.db"
            executor = SequentialExecutor(
                backend=be, traces_root=self._traces_dir, tasks_db=db_path,
            )
            result = executor.run(prog, program_ref=str(path))

            status_style = "green" if result.status in ("success", "goal_achieved") else "red"
            self._console.print(
                f"[{status_style}]Result:[/{status_style}] {result.status} "
                f"({len(result.steps)} steps, {result.duration_s:.1f}s)"
            )
        except Exception as exc:
            self._console.print(f"[red]Error: {exc}[/red]")
