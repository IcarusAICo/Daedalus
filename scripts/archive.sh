#!/usr/bin/env bash
set -euo pipefail

# Archives generated traces and learned skills to backup/<timestamp>/
# Core skills are preserved; only learned skills are archived.

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

BACKUP_DIR="backup/$(date +%Y%m%d_%H%M%S)"
echo "Creating backup at: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"/{skills,traces,memory,sandbox}

# Archive learned skills (non-core)
learned_count=0
if [ -d skills ]; then
    for dir in skills/*/; do
        [ -d "$dir" ] || continue
        skill_id=$(basename "$dir")
        if ! is_core_skill "$skill_id"; then
            echo "  archiving skill: $skill_id"
            cp -r "$dir" "$BACKUP_DIR/skills/"
            rm -rf "$dir"
            learned_count=$((learned_count + 1))
        fi
    done
fi
echo "  archived $learned_count learned skill(s)"

# Archive traces
trace_count=0
if [ -d traces ] && [ "$(ls -A traces 2>/dev/null)" ]; then
    for td in traces/*/; do
        [ -d "$td" ] || continue
        mv "$td" "$BACKUP_DIR/traces/"
        trace_count=$((trace_count + 1))
    done
fi
echo "  archived $trace_count trace(s)"

# Archive databases
for db in tasks.db memory.db; do
    if [ -f "$db" ]; then
        echo "  archiving database: $db"
        cp "$db" "$BACKUP_DIR/memory/"
        rm "$db"
    fi
done

# Archive implementor sandbox
if [ -d .daedalus/implementor_sandbox ]; then
    sandbox_count=$(find .daedalus/implementor_sandbox -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    if [ "$sandbox_count" -gt 0 ]; then
        echo "  archiving $sandbox_count sandbox skill(s)"
        mv .daedalus/implementor_sandbox "$BACKUP_DIR/sandbox/"
        mkdir -p .daedalus/implementor_sandbox
    fi
fi

echo ""
echo "Backup complete: $BACKUP_DIR"
echo "Agent is now reset to core skills only."
