"""ASCII banner and version display for the REPL."""

from __future__ import annotations

from rich.console import Console
from rich.text import Text


BANNER = r"""
     ___                  __      __
    / _ \ ___ _ ___  ___ / /___ _/ /__ __ ___
   / // // _ `// -_)/ _ \ / _ `// // // /(_-<
  /____/ \_,_/ \__/ \__/_/\_,_//_/ \_,_//___/
"""


def print_banner(console: Console, version: str = "0.0.1") -> None:
    """Print the Daedalus REPL welcome banner."""
    console.print(Text(BANNER, style="bold cyan"))
    console.print(f"  [dim]v{version} — interactive computer control agent[/dim]")
    console.print(f"  [dim]Type /help for commands, or type a goal in natural language.[/dim]")
    console.print()
