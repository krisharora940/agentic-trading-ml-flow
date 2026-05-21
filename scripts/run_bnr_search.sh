#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

THREAD_ID="${BNR_THREAD_ID:-bnr-desk-governor-local}"

exec python3 tools/run_bnr_desk_governor_workflow.py \
  --local-only \
  --summary-only \
  --thread-id "$THREAD_ID" \
  "$@"
