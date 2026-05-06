"""Static safety lint for synthesized skill code.

We can't sandbox arbitrary Python perfectly, but we can refuse code that uses
the highest-risk APIs without explicit permission. The Implementor runs this
*before* attempting to load and execute generated code.

Hard bans (always rejected):
    - subprocess.*, os.system, os.popen, os.exec*, os.spawn*
    - eval, exec, compile, __import__
    - importlib.import_module
    - shutil.rmtree, os.remove, os.unlink (anything destructive)

Conditional bans (rejected unless the corresponding side_effect is declared):
    - network: socket, urllib, urllib2, requests, httpx, aiohttp, http.client,
      smtplib, ftplib, telnetlib, paramiko
    - filesystem_read / filesystem_write: bare open() and Path read/write
      methods

Each violation is collected with the source line so the Implementor can
attempt a single repair pass.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

# fmt: off
_HARD_BANNED_FUNC_PATHS = frozenset({
    "os.system", "os.popen",
    "os.execv", "os.execve", "os.execvp", "os.execvpe", "os.execl", "os.execle", "os.execlp",
    "os.spawnl", "os.spawnle", "os.spawnlp", "os.spawnv", "os.spawnve", "os.spawnvp",
    "os.remove", "os.unlink", "os.rmdir", "os.removedirs",
    "shutil.rmtree", "shutil.move", "shutil.copy", "shutil.copytree",
    "importlib.import_module",
})
_HARD_BANNED_BUILTINS = frozenset({"eval", "exec", "compile", "__import__"})
_HARD_BANNED_MODULES = frozenset({
    "subprocess", "ctypes", "cffi", "mmap", "multiprocessing",
    "_thread", "code", "codeop", "pty", "pickle", "shelve", "marshal",
    "signal", "fcntl",
})

_NETWORK_MODULES = frozenset({
    "socket", "urllib", "urllib.request", "urllib2", "requests",
    "httpx", "aiohttp", "http", "http.client", "smtplib", "ftplib",
    "telnetlib", "paramiko", "websockets", "websocket",
})
_FILESYSTEM_PATH_METHODS = frozenset({
    "write_text", "write_bytes", "read_text", "read_bytes", "open",
    "unlink", "rename", "rmdir", "touch", "chmod",
})
# fmt: on


@dataclass
class SafetyViolation:
    rule: str
    detail: str
    lineno: int


class SafetyVisitor(ast.NodeVisitor):
    """AST walker. Pass ``declared_side_effects`` from the spec.yaml to allow
    network / filesystem usage when the skill has declared it."""

    def __init__(self, declared_side_effects: set[str]) -> None:
        self._side_effects = declared_side_effects
        self.violations: list[SafetyViolation] = []

    # -- imports --------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_import(alias.name, node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        self._check_import(mod, node.lineno)
        self.generic_visit(node)

    def _check_import(self, mod_name: str, lineno: int) -> None:
        if not mod_name:
            return
        head = mod_name.split(".", 1)[0]
        if head in _HARD_BANNED_MODULES:
            self.violations.append(
                SafetyViolation(
                    rule="hard_banned_module",
                    detail=f"module {mod_name!r} is banned in synthesized skills",
                    lineno=lineno,
                )
            )
            return
        if head in _NETWORK_MODULES and "network" not in self._side_effects:
            self.violations.append(
                SafetyViolation(
                    rule="undeclared_network",
                    detail=f"imports {mod_name!r} but spec.yaml does not declare side_effect 'network'",
                    lineno=lineno,
                )
            )
        if head == "tempfile" and "filesystem_write" not in self._side_effects:
            self.violations.append(
                SafetyViolation(
                    rule="undeclared_filesystem",
                    detail=f"imports {mod_name!r} but spec.yaml does not declare side_effect 'filesystem_write'",
                    lineno=lineno,
                )
            )

    # -- calls ----------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        path = self._dotted_func_name(node.func)
        if path in _HARD_BANNED_FUNC_PATHS:
            self.violations.append(
                SafetyViolation(
                    rule="hard_banned_func",
                    detail=f"call to {path!r} is banned in synthesized skills",
                    lineno=node.lineno,
                )
            )
        elif path is not None:
            head = path.split(".", 1)[0]
            if head in _HARD_BANNED_MODULES:
                self.violations.append(
                    SafetyViolation(
                        rule="hard_banned_module_call",
                        detail=f"call into banned module {path!r}",
                        lineno=node.lineno,
                    )
                )

        # Bare builtin dangerous calls.
        if isinstance(node.func, ast.Name) and node.func.id in _HARD_BANNED_BUILTINS:
            self.violations.append(
                SafetyViolation(
                    rule="hard_banned_builtin",
                    detail=f"call to builtin {node.func.id!r} is banned",
                    lineno=node.lineno,
                )
            )

        # Bare open() requires fs side-effects.
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "open"
            and not (self._side_effects & {"filesystem_read", "filesystem_write"})
        ):
            self.violations.append(
                SafetyViolation(
                    rule="undeclared_filesystem",
                    detail="bare open() requires filesystem_read or filesystem_write side_effect",
                    lineno=node.lineno,
                )
            )

        # Path.write_text / read_text style.
        if isinstance(node.func, ast.Attribute) and node.func.attr in _FILESYSTEM_PATH_METHODS:
            attr = node.func.attr
            is_write = attr.startswith("write") or attr in {
                "unlink", "rename", "touch", "chmod", "rmdir",
            }
            is_read = attr.startswith("read") or attr == "open"
            if is_write and "filesystem_write" not in self._side_effects:
                self.violations.append(
                    SafetyViolation(
                        rule="undeclared_filesystem",
                        detail=f"call to .{attr}() requires side_effect 'filesystem_write'",
                        lineno=node.lineno,
                    )
                )
            elif is_read and "filesystem_read" not in self._side_effects:
                self.violations.append(
                    SafetyViolation(
                        rule="undeclared_filesystem",
                        detail=f"call to .{attr}() requires side_effect 'filesystem_read'",
                        lineno=node.lineno,
                    )
                )

        self.generic_visit(node)

    @staticmethod
    def _dotted_func_name(node: ast.AST) -> str | None:
        parts: list[str] = []
        cur = node
        while True:
            if isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            elif isinstance(cur, ast.Name):
                parts.append(cur.id)
                break
            else:
                return None
        return ".".join(reversed(parts))


def lint_skill_source(
    source: str, declared_side_effects: set[str] | None = None
) -> list[SafetyViolation]:
    """Parse ``source`` and return any safety violations."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [SafetyViolation(rule="syntax_error", detail=str(exc), lineno=exc.lineno or 0)]
    visitor = SafetyVisitor(declared_side_effects or set())
    visitor.visit(tree)
    return visitor.violations
