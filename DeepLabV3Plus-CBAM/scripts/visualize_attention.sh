#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${1:?usage: visualize_attention.sh CHECKPOINT [extra args...]}"
shift
cd "$ROOT"
python visualize_attention.py --checkpoint "$CHECKPOINT" "$@"
