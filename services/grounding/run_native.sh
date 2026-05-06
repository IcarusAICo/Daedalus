#!/bin/bash
# Run the grounding service natively (no Docker needed).
# Uses its own venv with pinned transformers for Florence-2 compatibility.
#
# Usage:
#   cd services/grounding
#   ./run_native.sh
#
# The service will be available at http://localhost:8420

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Download weights if not present
if [ ! -f weights/icon_detect/model.pt ]; then
    echo "Downloading OmniParser V2 weights..."
    mkdir -p weights
    hf download microsoft/OmniParser-v2.0 \
        icon_detect/train_args.yaml icon_detect/model.pt icon_detect/model.yaml \
        icon_caption/config.json icon_caption/generation_config.json icon_caption/model.safetensors \
        --local-dir weights
    mv weights/icon_caption weights/icon_caption_florence
    echo "Weights downloaded."
fi

# Create/use a dedicated venv for the grounding service.
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating grounding service venv..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"
    echo "Venv created and dependencies installed."
fi

echo "Starting grounding service on port ${GROUNDING_PORT:-8420}..."
exec "$VENV_DIR/bin/python" server.py
