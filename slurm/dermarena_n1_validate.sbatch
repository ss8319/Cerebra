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
EVAL=/fs04/scratch2/ub62/ssim0070/DermArena/dataset_collection/eval
DATA=/fs04/scratch2/ub62/ssim0070/dermarena_dx_v2
declare -A DEV=(
  [rds]="$DATA/stratified/MM_RDS_benchmark_dev500.jsonl"
  [rdc]="$DATA/stratified/MM_RDC_benchmark_dev500.jsonl"
  [dxtest]="$DATA/stratified/DxTest_benchmark_dev500.jsonl"
)
HYP=/fs04/scratch2/ub62/ssim0070/DermArena/dataset_collection/data/benchmark/icd11_hypernym.json
PY=/fs04/scratch2/ub62/ssim0070/dermagent/bin/python
OUT=$CEREBRA/cerebra_cache/dermarena/n1_validate
mkdir -p "$OUT" "$CEREBRA/slurm/logs"

# Local weights (no HF download) + OpenRouter for base MLLM/judge (needs internet).
export HF_HOME=/fs04/scratch2/ub62/ssim0070/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$CEREBRA
export DERMARENA_BUDGET_USD=10
export OPENROUTER_API_KEY=$(grep -m1 '^OPENROUTER_API_KEY=' /fs04/scratch2/ub62/ssim0070/DermArena_data/.env | cut -d= -f2)
# Graders' judge-client writes a ledger; its default /mnt/hdd doesn't exist on M3 -> set it.
export LLM_LEDGER_PATH="$OUT/llm_ledger.jsonl"

echo "[gpu] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
cd "$CEREBRA"

declare -A BENCH=( [rds]="$DATA/MM_RDS_benchmark.jsonl" [rdc]="$DATA/MM_RDC_benchmark.jsonl" [dxtest]="$DATA/DxTest_benchmark.jsonl" )

# ── 1. Run the full pipeline, n=1 per task ──────────────────────────────────
for TASK in rds rdc dxtest; do
  echo "=================== PIPELINE $TASK (classify + RUN + --thinking) ==================="
  $PY -m cerebra.dermarena.pipeline --task "$TASK" --jsonl "${DEV[$TASK]}" \
      --limit 1 --thinking --out "$OUT/${TASK}_pred.jsonl" || echo "[warn] $TASK pipeline failed (continuing)"
  # 1-row benchmark matching the single predicted case (clean n=1 grade, not diluted by misses)
  head -1 "${DEV[$TASK]}" > "$OUT/${TASK}_bench1.jsonl"
done

# ── 2. Grade ────────────────────────────────────────────────────────────────
echo "=================== GRADE rds/rdc (score_dx.py, +hypernyms) ==================="
for TASK in rds rdc; do
  $PY "$EVAL/score_dx.py" --benchmark "$OUT/${TASK}_bench1.jsonl" \
      --predictions "$OUT/${TASK}_pred.jsonl" --output-dir "$OUT/${TASK}_grade" \
      --hypernyms "$HYP" || echo "[warn] score_dx $TASK failed"
done

echo "=================== GRADE dxtest (grade_dxtest.py) ==================="
$PY "$EVAL/grade_dxtest.py" --benchmark "$OUT/dxtest_bench1.jsonl" \
    --predictions "$OUT/dxtest_pred.jsonl" --output-dir "$OUT/dxtest_grade" \
    || echo "[warn] grade_dxtest failed"

echo "=================== DONE — predictions + grades under $OUT ==================="
