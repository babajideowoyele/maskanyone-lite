#!/usr/bin/env bash
# Dev-only bootstrap: create a venv and install masker deps to run
# worker/masker.py directly without Docker. For deployment, use
# docker compose instead.
#
# Rationale: Docker Desktop's VM on Windows taxes CPU-bound ML ~10x.
# Linux hosts see no such overhead — if you're on Linux, `docker compose
# up -d` is the right path and this script is unnecessary.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO/.venv"
PY="${PYTHON:-python3}"

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "error: python3 not found. set PYTHON=<path> or install Python 3.10+." >&2
  exit 1
fi

echo "creating venv at $VENV ..."
"$PY" -m venv "$VENV"

# shellcheck disable=SC1091
if [[ -f "$VENV/Scripts/activate" ]]; then
  source "$VENV/Scripts/activate"   # Windows (git-bash)
else
  source "$VENV/bin/activate"       # Linux / macOS
fi

python -m pip install --upgrade pip
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.4.1 torchvision==0.19.1
python -m pip install -r "$REPO/scripts/native_dev_requirements.txt"

echo
echo "done. activate with:"
echo "  source $VENV/Scripts/activate   # windows git-bash"
echo "  source $VENV/bin/activate       # linux / macos"
echo "then run the masker directly:"
echo "  python worker/masker.py input.mp4 output.mp4 blur quick"
echo "  python worker/masker.py input.mp4 output.mp4 blur precision"
