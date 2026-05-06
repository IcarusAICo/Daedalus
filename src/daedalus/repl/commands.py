"""Slash command handlers for the Daedalus REPL."""

from __future__ import annotations

import datetime
import shutil
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from daedalus.core.registry import get_registry


def cmd_help(console: Console, **_kwargs: Any) -> None:
    """Show available commands."""
    table = Table(show_header=True, header_style="bold", title="Commands")
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description")
    commands = [
        ("/help", "Show this help message"),
        ("/skills [list|info <id>]", "List skills or show info about one"),
        ("/traces [list|show <id>]", "List traces or show details"),
        ("/config <path>", "Load a config file"),
        ("/set <key> <value>", "Set a session parameter (max-retries, backend, etc.)"),
        ("/archive", "Archive learned skills and traces for a fresh start"),
        ("/restore <path>", "Restore from a backup directory"),
        ("/memory recall <query>", "Search agent memory"),
        ("/memory forget", "Clear agent memory"),
        ("/goal <text>", "Explicitly run a goal (same as typing it directly)"),
        ("/run <program.yaml>", "Run a YAML/Python program file"),
        ("/quit", "Exit the REPL"),
    ]
    for cmd, desc in commands:
        table.add_row(cmd, desc)
    console.print(table)
    console.print("\n[dim]Or just type a natural language goal to plan and execute it.[/dim]")


def cmd_skills(console: Console, args: list[str], skills_dir: Path, **_kwargs: Any) -> None:
    """Handle /skills subcommands."""
    from daedalus.library import load_library

    load_library(skills_dir)
    registry = get_registry()

    if not args or args[0] == "list":
        table = Table(show_header=True, header_style="bold")
        table.add_column("id", style="cyan")
        table.add_column("version")
        table.add_column("kind")
        table.add_column("tags", style="dim")
        table.add_column("description", overflow="fold", style="dim")
        for entry in sorted(registry, key=lambda e: e.id):
            spec = entry.cls.SPEC
            tags = ", ".join(spec.tags) if spec.tags else "—"
            desc = spec.description.strip().split("\n")[0]
            table.add_row(entry.id, entry.version.raw, spec.kind, tags, desc)
        console.print(table)
    elif args[0] == "info" and len(args) > 1:
        skill_id = args[1]
        try:
            entry = registry.get(skill_id)
        except Exception:
            console.print(f"[red]skill not found: {skill_id}[/red]")
            return
        spec = entry.cls.SPEC
        console.print(f"[bold cyan]{entry.id}[/bold cyan] v{entry.version.raw} ({spec.kind})")
        console.print(f"  [bold]Description:[/bold] {spec.description}")
        console.print(f"  [bold]Side effects:[/bold] {', '.join(spec.side_effects) or 'none'}")
        console.print(f"  [bold]Tags:[/bold] {', '.join(spec.tags) or 'none'}")
        if spec.preconditions:
            console.print(f"  [bold]Preconditions:[/bold] {spec.preconditions}")
        inputs_schema = entry.cls.Inputs.model_json_schema()
        console.print(f"  [bold]Inputs:[/bold] {inputs_schema.get('properties', {})}")
        outputs_schema = entry.cls.Outputs.model_json_schema()
        console.print(f"  [bold]Outputs:[/bold] {outputs_schema.get('properties', {})}")
    else:
        console.print("[yellow]Usage: /skills [list|info <id>][/yellow]")


def cmd_traces(console: Console, args: list[str], traces_dir: Path, **_kwargs: Any) -> None:
    """Handle /traces subcommands."""
    from daedalus.tracing.recorder import list_traces

    tasks_db = traces_dir.parent / "tasks.db"

    if not args or args[0] == "list":
        if not tasks_db.exists():
            console.print("[dim]No traces recorded yet.[/dim]")
            return
        traces = list_traces(tasks_db)
        if not traces:
            console.print("[dim]No traces found.[/dim]")
            return
        table = Table(show_header=True, header_style="bold")
        table.add_column("task_id", style="cyan", no_wrap=True)
        table.add_column("name")
        table.add_column("status")
        table.add_column("started")
        table.add_column("events", justify="right")
        for t in traces[-20:]:
            status_style = "green" if t.get("status") == "success" else "red"
            table.add_row(
                t.get("task_id", "?"),
                t.get("name", "?"),
                f"[{status_style}]{t.get('status', '?')}[/{status_style}]",
                t.get("started", "?")[:19],
                str(t.get("num_events", 0)),
            )
        console.print(table)
    elif args[0] == "show" and len(args) > 1:
        task_id = args[1]
        task_dir = traces_dir / task_id
        if not task_dir.exists():
            console.print(f"[red]trace not found: {task_id}[/red]")
            return
        import json
        meta_path = task_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            console.print(f"[bold]Task:[/bold] {meta.get('task_id')}")
            console.print(f"[bold]Name:[/bold] {meta.get('task_name')}")
            console.print(f"[bold]Status:[/bold] {meta.get('status')}")
            console.print(f"[bold]Started:[/bold] {meta.get('started')}")
            console.print(f"[bold]Finished:[/bold] {meta.get('finished')}")
            console.print(f"[bold]Events:[/bold] {meta.get('events')}")
    else:
        console.print("[yellow]Usage: /traces [list|show <task_id>][/yellow]")


def cmd_set(console: Console, args: list[str], session: dict[str, Any], **_kwargs: Any) -> None:
    """Handle /set key value."""
    if len(args) < 2:
        console.print("[bold]Current settings:[/bold]")
        for k, v in sorted(session.items()):
            console.print(f"  {k} = {v!r}")
        return
    key = args[0]
    value = " ".join(args[1:])
    if value.isdigit():
        session[key] = int(value)
    elif value.lower() in ("true", "false"):
        session[key] = value.lower() == "true"
    else:
        session[key] = value
    console.print(f"[green]set[/green] {key} = {session[key]!r}")


def cmd_config(console: Console, args: list[str], session: dict[str, Any], **_kwargs: Any) -> None:
    """Handle /config <path>."""
    if not args:
        current = session.get("config_path")
        if current:
            console.print(f"[bold]Current config:[/bold] {current}")
        else:
            console.print("[dim]No config loaded. Use /config <path> to load one.[/dim]")
        return
    path = Path(args[0])
    if not path.exists():
        console.print(f"[red]config file not found: {path}[/red]")
        return
    session["config_path"] = str(path)
    console.print(f"[green]loaded config:[/green] {path}")


def cmd_archive(console: Console, skills_dir: Path, **_kwargs: Any) -> None:
    """Archive learned skills and traces."""
    project_root = skills_dir.parent

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

    learned_count = 0
    for skill_path in sorted(skills_dir.iterdir()):
        if skill_path.is_dir() and skill_path.name not in core_skills:
            shutil.copytree(skill_path, backup_dir / "skills" / skill_path.name)
            shutil.rmtree(skill_path)
            learned_count += 1

    traces_dir = project_root / "traces"
    trace_count = 0
    if traces_dir.exists():
        for td in sorted(traces_dir.iterdir()):
            if td.is_dir():
                shutil.move(str(td), str(backup_dir / "traces" / td.name))
                trace_count += 1

    for db_name in ("tasks.db", "memory.db"):
        db_path = project_root / db_name
        if db_path.exists():
            shutil.copy2(db_path, backup_dir / "memory" / db_name)
            db_path.unlink()

    console.print(f"[green]Archived:[/green] {learned_count} skills, {trace_count} traces -> {backup_dir}")


def cmd_restore(console: Console, args: list[str], skills_dir: Path, **_kwargs: Any) -> None:
    """Restore from a backup directory."""
    if not args:
        backup_root = skills_dir.parent / "backup"
        if backup_root.exists():
            console.print("[bold]Available backups:[/bold]")
            for d in sorted(backup_root.iterdir(), reverse=True):
                if d.is_dir():
                    console.print(f"  {d}")
        else:
            console.print("[dim]No backups found.[/dim]")
        return

    backup_path = Path(args[0])
    if not backup_path.exists():
        console.print(f"[red]backup not found: {backup_path}[/red]")
        return

    skills_backup = backup_path / "skills"
    if skills_backup.exists():
        count = 0
        for sp in sorted(skills_backup.iterdir()):
            if sp.is_dir():
                target = skills_dir / sp.name
                if not target.exists():
                    shutil.copytree(sp, target)
                    count += 1
        console.print(f"  restored {count} skill(s)")

    memory_backup = backup_path / "memory"
    if memory_backup.exists():
        for db_name in ("tasks.db", "memory.db"):
            src = memory_backup / db_name
            if src.exists() and not (skills_dir.parent / db_name).exists():
                shutil.copy2(src, skills_dir.parent / db_name)

    console.print(f"[green]Restore complete from:[/green] {backup_path}")


def cmd_memory(console: Console, args: list[str], memory_db: Path, **_kwargs: Any) -> None:
    """Handle /memory subcommands."""
    from daedalus.memory import AgentMemory

    if not args:
        console.print("[yellow]Usage: /memory recall <query> | /memory forget[/yellow]")
        return

    if args[0] == "recall":
        query = " ".join(args[1:]) if len(args) > 1 else ""
        if not query:
            console.print("[yellow]Usage: /memory recall <query>[/yellow]")
            return
        if not memory_db.exists():
            console.print("[dim]No memory database yet.[/dim]")
            return
        mem = AgentMemory(memory_db)
        facts = mem.recall(query, limit=10)
        if not facts:
            console.print("[dim]No matching facts found.[/dim]")
            return
        for f in facts:
            console.print(f"  [{f.category}] {f.content}")
    elif args[0] == "forget":
        if memory_db.exists():
            memory_db.unlink()
            console.print("[green]Memory cleared.[/green]")
        else:
            console.print("[dim]No memory to clear.[/dim]")
    else:
        console.print("[yellow]Usage: /memory recall <query> | /memory forget[/yellow]")
