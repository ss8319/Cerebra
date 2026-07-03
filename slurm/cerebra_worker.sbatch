#!/bin/bash
#SBATCH --job-name=cerebra-warm
#SBATCH --partition=m3h
#SBATCH --qos=m3h
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/fs04/scratch2/ub62/ssim0070/Cerebra/slurm/logs/cerebra-warm-%j.out
#SBATCH --error=/fs04/scratch2/ub62/ssim0070/Cerebra/slurm/logs/cerebra-warm-%j.err

# Persistent Cerebra-pipeline worker: holds ONE GPU for up to 12h, loads
# PanDerm+DermoGPT+MedGemma once, and services every iteration via a filesystem
# queue — no requeue, no model reload per run. Base model is OpenRouter (no GPU).
#
# Run an iteration:  bash slurm/cerebra_infer.sh <task> <benchmark.jsonl> <limit> <out.jsonl>
# Stop:              touch $CEREBRA_REQ_DIR/STOP   (or scancel the job)

set -uo pipefail
CEREBRA=/fs04/scratch2/ub62/ssim0070/Cerebra
source "$CEREBRA/slurm/env.sh"          # all paths/keys
export PYTHONPATH="$CEREBRA"
export CEREBRA_REQ_DIR="$CEREBRA/cerebra_cache/dermarena/warm/requests"
mkdir -p "$CEREBRA_REQ_DIR" "$CEREBRA/slurm/logs"

echo "=================== PREFLIGHT ==================="
"$PY" -m cerebra.dermarena.preflight || { echo "PREFLIGHT FAILED — aborting."; exit 1; }
echo "[gpu] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

cd "$CEREBRA"
"$PY" -m cerebra.dermarena.worker
