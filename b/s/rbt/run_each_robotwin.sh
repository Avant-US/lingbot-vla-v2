#!/usr/bin/env bash
# Thin wrapper: activate nothing here; the Python driver puts venv/bin on PATH.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV="${VENV:-/tmp/itnvla15rbt20}"
export PATH="${VENV}/bin:${PATH}"
exec "${VENV}/bin/python" "${ROOT}/b/s/rbt/run_each_robotwin.py" --config "${ROOT}/b/s/rbt/config.yaml" "$@"
