"""Pre-run confirmation prompt.

Renders the program plan as a Rich tree (skills, side effects, missing skills)
and asks the user to type ``approve`` (or pass ``--yes``). The user must
explicitly approve before any skill that has side effects on the controlled
host runs.
"""

from __future__ import annotations

import enum
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from daedalus.executor.dsl import AnyProgram, Program, PythonProgram, reference_statuses, summarize, summarize_any

if TYPE_CHECKING:
    from daedalus.learner.learner import NewSkillCandidate


class ConfirmDecision(enum.StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    DENY_WITH_COMMENTS = "deny_with_comments"


@dataclass
class ConfirmResult:
    decision: ConfirmDecision
    comments: str = ""


@dataclass
class ConfirmOptions:
    auto_yes: bool = False
    explain_only: bool = False


def render_program(console: Console, program: AnyProgram) -> None:
    summary = summarize_any(program)

    header_lines = [Text.from_markup(f"[bold]{program.name}[/bold]")]
    if program.description:
        header_lines.append(Text(program.description, style="dim"))

    version_label = "v2/python" if isinstance(program, PythonProgram) else "v1/sequential"
    header_lines.append(
        Text.from_markup(
            f"[cyan]{summary.step_count}[/cyan] skill call(s)  "
            f"[cyan]{len(summary.skills)}[/cyan] skill(s)  "
            f"daemons: [cyan]{summary.daemon_steps}[/cyan]  "
            f"format: [cyan]{version_label}[/cyan]"
        )
    )
    console.print(Panel(Text("\n").join(header_lines), title="Plan", border_style="cyan"))

    if isinstance(program, PythonProgram):
        syntax = Syntax(program.code, "python", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="Plan Code", border_style="green"))
    else:
        steps_table = Table(show_header=True, header_style="bold")
        steps_table.add_column("#", justify="right", style="dim", width=4)
        steps_table.add_column("skill", style="cyan", no_wrap=True)
        steps_table.add_column("inputs", overflow="fold")
        steps_table.add_column("note", overflow="fold", style="dim")
        for i, step in enumerate(program.steps):
            inputs_str = ", ".join(f"{k}={v!r}" for k, v in step.inputs.items()) or "—"
            steps_table.add_row(str(i), step.skill, inputs_str, step.description or "")
        console.print(steps_table)

        refs = reference_statuses(program)
        missing = [r for r in refs if r[1] != "registered"]
        if missing:
            body = Text()
            for sid, status in missing:
                body.append(f"  - {sid}  ", style="bold red")
                body.append(f"({status})\n", style="dim")
            console.print(Panel(body, title="Unresolved skills", border_style="red"))

    if summary.side_effects:
        console.print(
            Panel(
                Text(", ".join(summary.side_effects), style="yellow"),
                title="Side effects",
                border_style="yellow",
            )
        )


def confirm_program(
    program: AnyProgram,
    *,
    console: Console | None = None,
    auto_yes: bool = False,
    explain_only: bool = False,
) -> ConfirmResult:
    console = console or Console()
    render_program(console, program)
    if explain_only:
        return ConfirmResult(decision=ConfirmDecision.REJECT)
    if auto_yes:
        console.print("[yellow]auto-approved (--yes)[/yellow]")
        return ConfirmResult(decision=ConfirmDecision.APPROVE)
    if not sys.stdin.isatty() and not auto_yes:
        console.print("[bold red]No TTY detected and --yes not set; rejecting.[/bold red]")
        return ConfirmResult(decision=ConfirmDecision.REJECT)
    console.print(
        "[bold]This program will take control of the target host.[/bold]\n"
        "Type [bold green]approve[/bold green] to run, "
        "[bold yellow]deny[/bold yellow] to reject with comments, "
        "anything else to cancel."
    )
    answer = Prompt.ask(">>", default="cancel").strip().lower()
    if answer == "approve":
        return ConfirmResult(decision=ConfirmDecision.APPROVE)
    if answer == "deny":
        comments = Prompt.ask("Comments >>", default="").strip()
        return ConfirmResult(decision=ConfirmDecision.DENY_WITH_COMMENTS, comments=comments)
    return ConfirmResult(decision=ConfirmDecision.REJECT)


def confirm_skills(
    candidates: list[NewSkillCandidate],
    *,
    console: Console | None = None,
    auto_yes: bool = False,
) -> list[NewSkillCandidate]:
    """Show an overview of proposed skills, let the user approve/deny each one.

    Returns only the approved candidates.
    """
    console = console or Console()

    overview = Table(show_header=True, header_style="bold", title="Proposed New Skills")
    overview.add_column("#", justify="right", style="dim", width=4)
    overview.add_column("id", style="cyan", no_wrap=True)
    overview.add_column("description", overflow="fold")
    overview.add_column("rationale", overflow="fold", style="dim")
    overview.add_column("component skills", overflow="fold", style="yellow")
    for i, c in enumerate(candidates):
        overview.add_row(
            str(i),
            c.proposed_id,
            c.description,
            c.rationale,
            ", ".join(c.component_skills) or "—",
        )
    console.print(overview)

    if auto_yes:
        console.print("[yellow]auto-approving all skill candidates (--yes)[/yellow]")
        return list(candidates)

    if not sys.stdin.isatty():
        console.print("[bold red]No TTY; rejecting all skill candidates.[/bold red]")
        return []

    approved: list[NewSkillCandidate] = []
    for i, c in enumerate(candidates):
        console.print(
            f"\n[bold cyan]Skill {i}: {c.proposed_id}[/bold cyan]\n"
            f"  {c.description}\n"
            f"  Rationale: [dim]{c.rationale}[/dim]"
        )
        if c.inputs_hint:
            console.print(f"  Inputs:  [dim]{c.inputs_hint}[/dim]")
        if c.outputs_hint:
            console.print(f"  Outputs: [dim]{c.outputs_hint}[/dim]")
        answer = Prompt.ask(
            "  [bold green]approve[/bold green] or [bold red]deny[/bold red]?",
            default="deny",
        ).strip().lower()
        if answer == "approve":
            approved.append(c)
            console.print(f"  [green]approved[/green]")
        else:
            console.print(f"  [red]denied[/red]")

    console.print(f"\n[bold]{len(approved)}/{len(candidates)} skill(s) approved[/bold]")
    return approved
