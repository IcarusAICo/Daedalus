#!/usr/bin/env bash
set -euo pipefail

# Restores learned skills, traces, and databases from a backup directory.
# Usage: scripts/restore.sh backup/20260504_120000

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <backup_directory>"
    echo ""
    echo "Available backups:"
    if [ -d backup ]; then
        ls -1 backup/ 2>/dev/null | sort -r | head -20 | sed 's/^/  /'
    else
        echo "  (none)"
    fi
    exit 1
fi

BACKUP="$1"

if [ ! -d "$BACKUP" ]; then
    echo "Error: backup directory not found: $BACKUP"
    exit 1
fi

echo "Restoring from: $BACKUP"

# Restore learned skills
if [ -d "$BACKUP/skills" ] && [ "$(ls -A "$BACKUP/skills" 2>/dev/null)" ]; then
    skill_count=0
    for dir in "$BACKUP/skills"/*/; do
        [ -d "$dir" ] || continue
        skill_id=$(basename "$dir")
        target="skills/$skill_id"
        if [ -d "$target" ]; then
            echo "  skipping skill $skill_id (already exists)"
        else
            cp -r "$dir" "$target"
            echo "  restored skill: $skill_id"
            skill_count=$((skill_count + 1))
        fi
    done
    echo "  restored $skill_count skill(s)"
fi

# Restore traces
if [ -d "$BACKUP/traces" ] && [ "$(ls -A "$BACKUP/traces" 2>/dev/null)" ]; then
    mkdir -p traces
    trace_count=0
    for td in "$BACKUP/traces"/*/; do
        [ -d "$td" ] || continue
        task_id=$(basename "$td")
        if [ -d "traces/$task_id" ]; then
            echo "  skipping trace $task_id (already exists)"
        else
            cp -r "$td" "traces/$task_id"
            trace_count=$((trace_count + 1))
        fi
    done
    echo "  restored $trace_count trace(s)"
fi

# Restore databases
if [ -d "$BACKUP/memory" ]; then
    for db in tasks.db memory.db; do
        if [ -f "$BACKUP/memory/$db" ]; then
            if [ -f "$db" ]; then
                echo "  skipping $db (already exists)"
            else
                cp "$BACKUP/memory/$db" "$db"
                echo "  restored database: $db"
            fi
        fi
    done
fi

# Restore sandbox
if [ -d "$BACKUP/sandbox/implementor_sandbox" ]; then
    mkdir -p .daedalus
    if [ -d .daedalus/implementor_sandbox ] && [ "$(ls -A .daedalus/implementor_sandbox 2>/dev/null)" ]; then
        echo "  skipping sandbox (already has content)"
    else
        cp -r "$BACKUP/sandbox/implementor_sandbox" .daedalus/
        echo "  restored implementor sandbox"
    fi
fi

echo ""
echo "Restore complete from: $BACKUP"
