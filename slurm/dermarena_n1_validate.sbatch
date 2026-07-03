#!/bin/bash
#SBATCH --job-name=dermarena-n1
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/fs04/scratch2/ub62/ssim0070/Cerebra/slurm/logs/dermarena-n1-%j.out
#SBATCH --error=/fs04/scratch2/ub62/ssim0070/Cerebra/slurm/logs/dermarena-n1-%j.err

# n=1-per-task end-to-end validation of the DermArena agent:
#   classify (Step 1) + PROPOSE + RUN (GPU vision) + DEBATE (reasoning ON) -> grade.
# Compute nodes HAVE internet (verified), so OpenRouter (Qwen3.5-27B + judge) + local
# GPU weights run in one job. One shared case per task (first row of the dev300 subset).

set -euo pipefail

CEREBRA=/fs04/scratch2/ub62/ssim0070/Cerebra
source "$CEREBRA/slurm/env.sh"          # single source of truth for ALL paths/keys
export PYTHONPATH="$CEREBRA"
OUT="$CEREBRA/cerebra_cache/dermarena/n1_validate"
rm -rf "$OUT"; mkdir -p "$OUT" "$CEREBRA/slurm/logs"   # clean → no stale-file / resume gotchas

declare -A DEV=(
  [rds]="$DERMARENA_DATA/stratified/MM_RDS_benchmark_dev500.jsonl"
  [rdc]="$DERMARENA_DATA/stratified/MM_RDC_benchmark_dev500.jsonl"
  [dxtest]="$DERMARENA_DATA/stratified/DxTest_benchmark_dev500.jsonl"
)

# ── PREFLIGHT: fail loud before wasting the GPU on a broken setup ────────────
echo "=================== PREFLIGHT ==================="
"$PY" -m cerebra.dermarena.preflight || { echo "PREFLIGHT FAILED — aborting run."; exit 1; }

echo "[gpu] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
cd "$CEREBRA"

# ── 1. Run the full pipeline, n=1 per task ──────────────────────────────────
for TASK in rds rdc dxtest; do
  echo "=================== PIPELINE $TASK (SELECT -> RUN -> VERIFY, no-think dev) ==================="
  $PY -m cerebra.dermarena.pipeline --task "$TASK" --jsonl "${DEV[$TASK]}" \
      --limit 1 --out "$OUT/${TASK}_pred.jsonl" || echo "[warn] $TASK pipeline failed (continuing)"
  # 1-row benchmark matching the single predicted case (clean n=1 grade, not diluted by misses)
  head -1 "${DEV[$TASK]}" > "$OUT/${TASK}_bench1.jsonl"
done

# ── 2. Grade ────────────────────────────────────────────────────────────────
echo "=================== GRADE rds/rdc (score_dx.py, +hypernyms) ==================="
for TASK in rds rdc; do
  # judge = $DERMARENA_JUDGE_MODEL (gemini-2.5-flash) — NOT the invalid default gemini-3-flash.
  "$PY" "$DERMARENA_EVAL/score_dx.py" --benchmark "$OUT/${TASK}_bench1.jsonl" \
      --predictions "$OUT/${TASK}_pred.jsonl" --output-dir "$OUT/${TASK}_grade" \
      --hypernyms "$DERMARENA_HYPERNYMS" --model "$DERMARENA_JUDGE_MODEL" || echo "[warn] score_dx $TASK failed"
done

echo "=================== GRADE dxtest (grade_dxtest.py) ==================="
"$PY" "$DERMARENA_EVAL/grade_dxtest.py" --benchmark "$OUT/dxtest_bench1.jsonl" \
    --predictions "$OUT/dxtest_pred.jsonl" --output-dir "$OUT/dxtest_grade" \
    || echo "[warn] grade_dxtest failed"

echo "=================== DONE — predictions + grades under $OUT ==================="
