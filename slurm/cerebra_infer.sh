#!/bin/bash
# Run one iteration on the persistent Cerebra warm worker (no model reload).
# Usage: cerebra_infer.sh <task> <benchmark.jsonl> <limit> <out.jsonl> [thinking]
set -euo pipefail
REQ_DIR=/fs04/scratch2/ub62/ssim0070/Cerebra/cerebra_cache/dermarena/warm/requests
mkdir -p "$REQ_DIR"
TASK=$1; JSONL=$2; LIMIT=$3; OUT=$4; THINK=${5:-false}
rm -f "$OUT" "$OUT.done"
REQ="$REQ_DIR/req_$(date +%s%N 2>/dev/null || echo $$).json"
cat > "$REQ" <<EOF
{"task":"$TASK","jsonl":"$JSONL","limit":$LIMIT,"out":"$OUT","thinking":$THINK}
EOF
echo "[cerebra_infer] queued $TASK -> $OUT ; waiting for worker..."
while [ ! -f "$OUT.done" ]; do sleep 3; done
echo "[cerebra_infer] done: $OUT"
