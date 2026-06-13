#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRESET="${1:-default}"
shift || true
cd "$ROOT"
python train.py --config "configs/${PRESET}.json" "$@"
