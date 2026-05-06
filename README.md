# Daedalus

A multistage learning system for computer control. The agent runs on a "control plane" machine and drives a target host (Windows/macOS/Linux) over a remote-desktop backend (VNC). Programs are built by composing typed, versioned, testable **skills** out of a growing library. On failure, a **learner** loop analyzes traces, proposes fixes and new skills, and re-plans automatically.

## Quick start

```bash
# Install (editable) — makes `daedalus` available on your PATH
uv sync --all-extras
uv pip install -e .

# Verify your installed LiteLLM is not affected by the March 2026 supply-chain attack
bash scripts/verify_litellm.sh

# Run the unit tests (uses the deterministic MockBackend, no VNC required)
daedalus verify-litellm
pytest

# List the skills in the library
daedalus skills list

# Smoke-test a hand-written program against the mock backend
daedalus run --program examples/mock_smoke.yaml --backend mock

# Run a goal-driven task (explorer generates a plan, then executes it)
daedalus run --goal "Open Firefox and navigate to example.com" \
    --config config.local.yaml --backend vnc --host <YOUR_HOST> --port 5900

# Run a previously generated plan with screen recording
daedalus run --program traces/t_<id>/plan.yaml --config config.local.yaml \
    --backend vnc --host <YOUR_HOST> --port 5900 --record

# Analyze recent traces with the learner
daedalus teach --config config.local.yaml

# Interactive REPL for ad-hoc skill execution
daedalus shell --config config.local.yaml --backend vnc --host <YOUR_HOST>
```

## Architecture

```
User Goal
    │
    ▼
┌──────────────┐  recall()  ┌─────────────┐
│   Explorer   │◄───────────│ AgentMemory │
│  (tool-loop) │            └─────────────┘
│              │──► skills from Librarian (BM25 retrieval)
│   plan()     │──► PlanResult { Program | PythonProgram }
└──────┬───────┘
       ▼
  Confirm Plan (approve / deny with comments / cancel)
       │
       ▼
┌────────────────────────────────────────────────────┐
│ Retry Loop (up to --max-retries attempts)          │
│                                                    │
│  Execute ──► Executor ──► TraceRecorder            │
│     │                                              │
│     ├─ SUCCESS ──► Evaluator ──► GoalVerdict       │
│     │   └─ if --learn-on-succeed: Learner          │
│     │                                              │
│     └─ FAILURE ──► Learner (tool-loop analysis)    │
│         ├─ propose new skills ──► user approve     │
│         ├─ propose skill amendments                │
│         ├─ suggestions ──► Explorer re-plans       │
│         └─ confirm new plan ──► loop               │
└────────────────────────────────────────────────────┘
```

### Key components

| Component | Path | Role |
|-----------|------|------|
| **Skill library** | `skills/<skill_id>/` | On-disk `spec.yaml` + `skill.py` + `tests/`. Core skills defined in `skills/CORE.yaml`. |
| **Core** | `src/daedalus/core/` | `AtomicSkill`, `DaemonSkill`, `SkillSpec`, `Registry`, `ExecutionContext`, `TaskState`, error hierarchy. |
| **Explorer** | `src/daedalus/explorer/` | LLM-backed tool-calling loop that decomposes goals into executable programs. |
| **Planner** | `src/daedalus/planner/` | LLM-backed goal-to-program planner with strategy decomposition, success criteria generation, and repair loop. |
| **Learner** | `src/daedalus/learner/` | Tool-calling loop that analyzes execution traces (events + screenshots). On failure: diagnoses root cause, proposes fixes, proposes new skills or amendments. On success: suggests optimizations. |
| **Implementor** | `src/daedalus/implementor/` | LLM-backed skill synthesis with AST safety linting and sandbox testing. |
| **Evaluator** | `src/daedalus/evaluator/` | Post-execution goal verification via visual, trace, and state criteria. |
| **Executor** | `src/daedalus/executor/` | Runs validated DSL programs (v1 step-based YAML or v2 Python) with per-step timeouts and daemon lifecycle management. |
| **Backends** | `src/daedalus/backends/` | `RemoteDesktop` protocol; `MockBackend` (tests) and `VNCBackend` (production). |
| **Library** | `src/daedalus/library/` | Skill loading, BM25 retrieval index (Librarian), and core skill manifest management. |
| **Tracing** | `src/daedalus/tracing/` | JSONL events + screenshots per run, stored under `traces/`. |
| **Memory** | `src/daedalus/memory/` | Persistent cross-run fact store so the planner learns from past successes and failures. |
| **LLM gateway** | `src/daedalus/llm/` | LiteLLM wrapper with role-based model routing (planner, explorer, learner, implementor, vision, cheap). |
| **UI** | `src/daedalus/ui/` | Rich CLI plan confirmation with deny-with-comments; interactive per-skill approval; Tk overlay with abort hotkey. |
| **REPL** | `src/daedalus/repl/` | Interactive shell for ad-hoc skill execution and debugging. |

### Skills (19 built-in)

Core skills (cannot be amended by the learner):

`click_mouse`, `type_text`, `type_shortcut`, `type_shortcuts`, `type_text_secret`, `scroll`, `wait`, `view_screen`, `vision_query`, `click_element`, `locate_element`, `locate_elements`, `click_all`, `assert_screen_contains`, `store_query`, `populate_store_from_analysis`, `monitor_text_region`, `tick_counter`

Additional skills can be proposed by the learner and synthesized by the implementor at runtime.

## CLI reference

```
daedalus run         Execute a goal or pre-built program
daedalus plan        Plan without executing
daedalus teach       Run the learner on recent traces
daedalus implement   Synthesize a skill from a spec
daedalus shell       Interactive REPL
daedalus skills      list | test [SKILL_ID] | sync
daedalus traces      list | show TASK_ID
daedalus archive     Archive old traces
daedalus restore     Restore archived traces
daedalus verify-litellm   Check for supply-chain compromise
```

### Key flags for `daedalus run`

| Flag | Default | Description |
|------|---------|-------------|
| `--goal` | — | Natural-language task description (triggers explorer/planner). |
| `--program` | — | Path to a pre-built plan YAML to execute directly. |
| `--config` | — | Path to YAML config (LLM roles, backend settings, paths). |
| `--backend` | `vnc` | Backend kind: `vnc` or `mock`. |
| `--host` | from config | VNC host to connect to. |
| `--port` | `5900` | VNC port. |
| `--max-retries` / `-r` | 3 | Learner-driven retry loops on execution failure. |
| `--explore-steps` | 50 | Max tool-calling steps for the explorer. |
| `--learn-on-succeed` | off | Run the learner even after successful execution. |
| `--record` | off | Record the screen during execution (requires ffmpeg on remote host). |
| `--record-fps` | 4 | Frames per second for screen recording. |
| `--yes` / `-y` | off | Auto-approve all confirmation prompts. |
| `--no-overlay` | off | Disable the "AGENT ACTIVE" overlay window. |
| `--no-strategy` | off | Skip the strategy phase. |
| `--verbose` | off | Enable DEBUG logging. |

## Configuration

Create a YAML config (see `config.example.yaml`):

```yaml
backend:
  kind: vnc
  host_os: macos
  vnc:
    host: my-mac
    port: 5900
    password_env: DAEDALUS_VNC_PASSWORD
    username_env: DAEDALUS_VNC_USERNAME
    max_width: 1728
    max_height: 1117

llm:
  roles:
    planner:     bedrock/us.anthropic.claude-opus-4-6-v1
    explorer:    bedrock/us.anthropic.claude-opus-4-6-v1
    implementor: bedrock/us.anthropic.claude-sonnet-4-6
    learner:     bedrock/us.anthropic.claude-sonnet-4-6
    vision:      bedrock/us.anthropic.claude-sonnet-4-6
    cheap:       bedrock/us.anthropic.claude-sonnet-4-6
  aws_region: us-east-1
  creative_temperature: 0.7
  analytical_temperature: 0.0

paths:
  skills_dir: skills
  traces_dir: traces
  tasks_db: tasks.db

executor:
  default_screen_width: 1728
  default_screen_height: 1117
  step_timeout_s: 120

ui:
  overlay: false
  confirm: true

grounding:
  endpoint: http://localhost:8420
  timeout_s: 5
```

### LLM roles

| Role | Used by | Recommendation |
|------|---------|----------------|
| `planner` | Goal decomposition, strategy | Opus-class for complex reasoning |
| `explorer` | Interactive tool-calling exploration | Opus-class for complex reasoning |
| `implementor` | Skill code generation | Sonnet-class |
| `learner` | Trace analysis and feedback | Sonnet-class |
| `vision` | Screenshot understanding | Sonnet-class (multimodal) |
| `cheap` | Simple classification tasks | Sonnet-class or smaller |

## Screen recording

When `--record` is passed, Daedalus records the target host's screen during execution and saves it as an MP4 in the trace folder. On macOS hosts this uses `ffmpeg` with `avfoundation` via a `launchctl` agent (to access Screen Recording permissions). On Linux it uses `x11grab`.

**Requirements for macOS recording:**
1. `ffmpeg` installed on the Mac (`brew install ffmpeg`)
2. SSH key access to the Mac from the control plane
3. `ffmpeg` granted Screen Recording permission in System Settings > Privacy & Security

## Program DSLs

Daedalus supports two program formats:

**v1 (step-based YAML):** Declarative sequence of skill invocations with parameters.

**v2 (Python):** Full Python code that calls skills via `ctx.call(skill_id, **params)`. More flexible for loops, conditionals, and complex logic like puzzle solving.

## Security

LiteLLM versions **1.82.7** and **1.82.8** were compromised by the TeamPCP supply-chain attack on March 24, 2026. This project pins `litellm>=1.83.0` and refuses to install the compromised versions. Run `daedalus verify-litellm` or `bash scripts/verify_litellm.sh` to scan for known indicators of compromise.

## Transparency

- Every run requires explicit user approval of the plan before execution. Users can **deny with comments** to request a revised plan.
- When the learner proposes new skills, each skill is presented for individual approve/deny.
- Core skills (defined in `skills/CORE.yaml`) cannot be amended by the learner.
- While the executor is active, a local always-on-top overlay window is shown ("AGENT ACTIVE"), and `Ctrl+Shift+Esc` aborts the run (disable with `--no-overlay`).
- All execution traces (events, screenshots, timing data) are recorded under `traces/` for post-hoc inspection.
