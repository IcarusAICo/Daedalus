# Security Policy — Daedalus (Computer Control Agent)

This document describes the security posture of the Daedalus project: what the
agent can do, what we defend against, the layers in place, and the known gaps.

## Agent Capabilities

Daedalus operates as an autonomous desktop agent with the following capabilities:

- **Full input control.** Keyboard and mouse events on the target host via a VNC
  connection, including arbitrary key combinations and pointer movement.
- **Screen observation.** The vision-language model (VLM) reads pixel content from
  the VNC framebuffer to decide its next action.
- **Skill execution.** The planner composes registered skills into programs; the
  executor runs those programs step by step.
- **LLM invocation.** Multiple LLM calls (planner, implementor, teacher) are made
  per task cycle through LiteLLM.
- **Local filesystem access.** Traces, the task database, and generated skill code
  are read from and written to the host filesystem.

## Threat Model

| Threat surface | Trust level | Notes |
|---|---|---|
| User commands | **Trusted.** The operator issues tasks and approves programs. | |
| LLM outputs (planner, implementor, teacher) | **Untrusted.** Model responses may contain injected or hallucinated actions. | Treated as adversarial input at every consumption point. |
| Target host screen content | **Potentially untrusted.** A compromised or adversarial application could present misleading UI to the VLM. | Prompt-injection via rendered text is a recognized risk. |
| Supply chain (PyPI, LiteLLM) | **Monitored.** The LiteLLM compromise incident of March 2026 demonstrated that transitive dependencies can carry malicious payloads. | See "LiteLLM Pin" below. |

## Defense Layers

### 1. User Confirmation

Every generated program requires explicit operator approval before execution
begins. The approval prompt displays the full skill sequence so the user can
inspect it. No program runs without a deliberate "approve" action.

### 2. Overlay and Abort

While a program is running, an always-on-top "AGENT ACTIVE" banner is rendered
on screen with an abort button. Pressing the button or sending Ctrl-C in the
controlling terminal immediately halts execution. If the optional `pynput`
dependency is installed, a configurable global hotkey provides a third abort
path that works even when the terminal is not focused.

### 3. LiteLLM Version Pin

Versions 1.82.7 and 1.82.8 of LiteLLM are explicitly banned in
`pyproject.toml`. The script `scripts/verify_litellm.sh` scans the installed
environment for known IOC artifacts (unexpected network calls, injected
code paths) associated with the March 2026 supply-chain incident.

### 4. Implementor Static Lint

Before any generated skill code is loaded, an AST-based safety walker inspects
the parse tree and rejects programs that reference:

- `subprocess`, `os.system`, `os.popen`
- `eval`, `exec`, `compile`, `__import__`
- Undeclared network access (e.g., `socket`, `requests`, `urllib`)
- Undeclared filesystem access outside the sanctioned trace/task directories

The walker operates as a deny list over the AST; see "Known Limitations" for
caveats.

### 5. Implementor Subprocess Sandbox

Skill code that passes static lint is executed in an isolated subprocess with:

- A stripped environment (no inherited secrets or tokens).
- CPU time limits enforced via resource controls.
- A hard wall-clock timeout after which the subprocess is killed.

### 6. Sensitive-Field Redaction

Skills can declare `sensitive_inputs` in their metadata. Any field marked
sensitive is redacted in traces, logs, and the task database so that
credentials and PII do not persist on disk in cleartext.

### 7. Registry Enforcement

The planner and executor resolve skills exclusively by registered skill ID.
Skill IDs are validated against the registry before dispatch. An LLM cannot
hallucinate a new skill into existence — only IDs present in the registry are
callable.

## Known Limitations

- **VNC password in plaintext.** Depending on the VNC server configuration, the
  session password may be transmitted without encryption. Use an SSH tunnel or
  TLS-enabled VNC server in sensitive environments.
- **Global hotkey requires `pynput`.** Without the optional `pynput` dependency,
  abort is limited to the overlay button and Ctrl-C. Install `pynput` for the
  full abort surface.
- **AST lint is best-effort.** The static safety walker is a deny-list
  heuristic, not a true capability sandbox. Determined adversarial code may
  evade it through dynamic dispatch, codec tricks, or ctypes. It reduces
  attack surface but does not eliminate it.
- **Secrets in VNC traffic.** Text entered via `type_text` is visible in the VNC
  stream. For credentials and other secrets, use `type_text_secret` backed by
  environment variables so that the literal value is never transmitted as
  keystrokes in the clear.

## Reporting Vulnerabilities

If you discover a security issue in Daedalus, please report it through responsible
disclosure:

1. Email the maintainer with a description of the vulnerability and
   reproduction steps.
2. Do not open a public issue until a fix is available or 90 days have elapsed.
3. Expect an acknowledgment within 48 hours and a remediation timeline within
   7 business days.

We appreciate the effort of security researchers and will credit reporters in
the release notes (unless anonymity is requested).
