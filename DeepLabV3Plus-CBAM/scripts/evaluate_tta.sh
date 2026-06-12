#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${1:?usage: evaluate_tta.sh CHECKPOINT [extra args...]}"
shift
cd "$ROOT"
python evaluate_tta.py --checkpoint "$CHECKPOINT" "$@"
