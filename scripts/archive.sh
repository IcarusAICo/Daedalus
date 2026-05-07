#!/usr/bin/env bash
set -euo pipefail

# Archives generated traces and learned skills to backup/<timestamp>/
# Core skills are preserved; only learned skills are archived.
#
# Usage:
#   ./scripts/archive.sh                  # archive everything (with confirmation)
#   ./scripts/archive.sh --skills         # only archive learned skills
#   ./scripts/archive.sh --traces         # only archive traces
#   ./scripts/archive.sh --db             # only archive databases
#   ./scripts/archive.sh --sandbox        # only archive sandbox skills
#   ./scripts/archive.sh --skills --traces  # combine flags
#   ./scripts/archive.sh --yes            # skip confirmation prompt

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CORE_YAML="skills/CORE.yaml"
if [ ! -f "$CORE_YAML" ]; then
    echo "ERROR: $CORE_YAML not found" >&2
    exit 1
fi

# Parse core skills from CORE.yaml (lines matching "  - <name>")
mapfile -t CORE_SKILLS < <(grep -E '^\s+-\s+' "$CORE_YAML" | sed 's/^[[:space:]]*-[[:space:]]*//')

is_core_skill() {
    local skill="$1"
    for core in "${CORE_SKILLS[@]}"; do
        if [[ "$skill" == "$core" ]]; then
            return 0
        fi
    done
    return 1
}

# --- Parse flags ---
DO_SKILLS=false
DO_TRACES=false
DO_DB=false
DO_SANDBOX=false
SKIP_CONFIRM=false
SELECTIVE=false

for arg in "$@"; do
    case "$arg" in
        --skills)  DO_SKILLS=true; SELECTIVE=true ;;
        --traces)  DO_TRACES=true; SELECTIVE=true ;;
        --db)      DO_DB=true; SELECTIVE=true ;;
        --sandbox) DO_SANDBOX=true; SELECTIVE=true ;;
        --yes|-y)  SKIP_CONFIRM=true ;;
        *)
            echo "Unknown flag: $arg" >&2
            echo "Usage: $0 [--skills] [--traces] [--db] [--sandbox] [--yes|-y]" >&2
            exit 1
            ;;
    esac
done

# No selective flags = archive everything
if [ "$SELECTIVE" = false ]; then
    DO_SKILLS=true
    DO_TRACES=true
    DO_DB=true
    DO_SANDBOX=true
fi

# --- Preview what will be archived ---
echo "=== Archive Preview ==="
echo ""

# Collect learned skills
learned_skills=()
if [ "$DO_SKILLS" = true ] && [ -d skills ]; then
    for dir in skills/*/; do
        [ -d "$dir" ] || continue
        skill_id=$(basename "$dir")
        if ! is_core_skill "$skill_id"; then
            learned_skills+=("$skill_id")
        fi
    done
fi

if [ "$DO_SKILLS" = true ]; then
    if [ ${#learned_skills[@]} -gt 0 ]; then
        echo "Skills to archive (${#learned_skills[@]}):"
        for s in "${learned_skills[@]}"; do
            echo "  - $s"
        done
    else
        echo "Skills to archive: (none)"
    fi
    echo ""
fi

# Count traces
trace_count=0
if [ "$DO_TRACES" = true ] && [ -d traces ] && [ "$(ls -A traces 2>/dev/null)" ]; then
    for td in traces/*/; do
        [ -d "$td" ] || continue
        trace_count=$((trace_count + 1))
    done
fi
if [ "$DO_TRACES" = true ]; then
    echo "Traces to archive: $trace_count"
fi

# List databases
dbs_to_archive=()
if [ "$DO_DB" = true ]; then
    for db in tasks.db memory.db; do
        if [ -f "$db" ]; then
            dbs_to_archive+=("$db")
        fi
    done
    if [ ${#dbs_to_archive[@]} -gt 0 ]; then
        echo "Databases to archive: ${dbs_to_archive[*]}"
    else
        echo "Databases to archive: (none)"
    fi
fi

# Count sandbox skills
sandbox_count=0
if [ "$DO_SANDBOX" = true ] && [ -d .daedalus/implementor_sandbox ]; then
    sandbox_count=$(find .daedalus/implementor_sandbox -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
fi
if [ "$DO_SANDBOX" = true ]; then
    echo "Sandbox skills to archive: $sandbox_count"
fi

echo ""
echo "========================"

# Confirmation
if [ "$SKIP_CONFIRM" = false ]; then
    read -rp "Proceed with archive? [y/N] " confirm
    if [[ "$confirm" != [yY] && "$confirm" != [yY][eE][sS] ]]; then
        echo "Archive cancelled."
        exit 0
    fi
fi

echo ""

# --- Perform archive ---
BACKUP_DIR="backup/$(date +%Y%m%d_%H%M%S)"
echo "Creating backup at: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# Archive learned skills (non-core)
if [ "$DO_SKILLS" = true ] && [ ${#learned_skills[@]} -gt 0 ]; then
    mkdir -p "$BACKUP_DIR/skills"
    for skill_id in "${learned_skills[@]}"; do
        echo "  archiving skill: $skill_id"
        cp -r "skills/$skill_id" "$BACKUP_DIR/skills/"
        rm -rf "skills/$skill_id"
    done
    echo "  archived ${#learned_skills[@]} learned skill(s)"
fi

# Archive traces
if [ "$DO_TRACES" = true ] && [ "$trace_count" -gt 0 ]; then
    mkdir -p "$BACKUP_DIR/traces"
    archived_traces=0
    for td in traces/*/; do
        [ -d "$td" ] || continue
        mv "$td" "$BACKUP_DIR/traces/"
        archived_traces=$((archived_traces + 1))
    done
    echo "  archived $archived_traces trace(s)"
fi

# Archive databases
if [ "$DO_DB" = true ] && [ ${#dbs_to_archive[@]} -gt 0 ]; then
    mkdir -p "$BACKUP_DIR/memory"
    for db in "${dbs_to_archive[@]}"; do
        echo "  archiving database: $db"
        cp "$db" "$BACKUP_DIR/memory/"
        rm "$db"
    done
fi

# Archive implementor sandbox
if [ "$DO_SANDBOX" = true ] && [ "$sandbox_count" -gt 0 ]; then
    mkdir -p "$BACKUP_DIR/sandbox"
    echo "  archiving $sandbox_count sandbox skill(s)"
    mv .daedalus/implementor_sandbox "$BACKUP_DIR/sandbox/"
    mkdir -p .daedalus/implementor_sandbox
fi

echo ""
echo "Backup complete: $BACKUP_DIR"
if [ "$SELECTIVE" = false ]; then
    echo "Agent is now reset to core skills only."
fi
