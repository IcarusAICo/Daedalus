"""Entry point for the `daedalus` command.

- With no arguments: launches the Ink terminal UI (Node.js frontend).
- With a subcommand (run, plan, skills, traces, etc.): delegates to the
  Python Typer CLI for headless/scripted usage.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _find_frontend_dist() -> Path | None:
    """Locate the built frontend dist/index.js relative to this package."""
    pkg_dir = Path(__file__).resolve().parent  # src/daedalus/
    candidates = [
        pkg_dir.parent.parent / "frontend" / "dist" / "index.js",  # dev layout
        Path(sys.prefix) / "share" / "daedalus" / "frontend" / "dist" / "index.js",  # installed
        Path.cwd() / "frontend" / "dist" / "index.js",  # cwd fallback
    ]

    for c in candidates:
        if c.exists():
            return c
    return None


def _find_node() -> str | None:
    """Find a compatible Node.js binary (>= 20)."""
    # Check well-known local install paths first
    local_node = Path.home() / ".local" / "node-v22.15.0-linux-x64" / "bin" / "node"
    if local_node.exists():
        return str(local_node)

    # Check PATH
    node = shutil.which("node")
    if node:
        return node
    return None


def _has_subcommand(argv: list[str]) -> bool:
    """Check if argv contains a known subcommand (not just flags)."""
    known_subcommands = {"run", "plan", "teach", "implement", "shell", "skills", "traces", "archive", "restore", "verify-litellm"}
    for arg in argv:
        if arg in known_subcommands:
            return True
        if arg == "--help" or arg == "-h":
            return True
    return False


def main() -> None:
    argv = sys.argv[1:]

    # If a subcommand is given, use the Python Typer CLI directly
    if _has_subcommand(argv):
        from daedalus.cli import app
        app()
        return

    # Otherwise, launch the Ink terminal UI
    frontend_js = _find_frontend_dist()
    if frontend_js is None:
        print(
            "error: Daedalus frontend not built. Run:\n"
            "  cd frontend && npm install && npm run build\n",
            file=sys.stderr,
        )
        sys.exit(1)

    node_bin = _find_node()
    if node_bin is None:
        print(
            "error: Node.js not found. Install Node.js >= 20 to use the interactive UI.\n"
            "       Or use `daedalus run --goal '...'` for headless mode.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Pass through any flags to the frontend (e.g. --goal, --config)
    cmd = [node_bin, str(frontend_js), "--cwd", str(Path.cwd())]
    cmd.extend(argv)

    try:
        os.execvp(node_bin, cmd)
    except FileNotFoundError:
        print(
            "error: Failed to launch Node.js frontend.\n"
            "       Or use `daedalus run --goal '...'` for headless mode.",
            file=sys.stderr,
        )
        sys.exit(1)
