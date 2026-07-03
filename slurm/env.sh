# Cerebra DermArena — single source of truth for every path/key.
# `source` this at the top of any run/grade script so nothing defaults to a dead value.
SCRATCH=/fs04/scratch2/ub62/ssim0070
export CEREBRA="$SCRATCH/Cerebra"
export DERMARENA_DATA="$SCRATCH/dermarena_dx_v2"
export DERMARENA_EVAL="$SCRATCH/DermArena/dataset_collection/eval"
export PY="$SCRATCH/dermagent/bin/python"

# --- vision tool weights (local; no downloads) --------------------------------
export MEDGEMMA_PATH="$SCRATCH/models/medgemma-1.5-4b-it"
export DERMOGPT_PATH="$SCRATCH/DermAgent/model-weights/DermoGPT-RL"
export DERM1M_SRC="$SCRATCH/DermAgent/Derm1M/src"
export HF_HOME="$SCRATCH/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- base MLLM + grader judge (OpenRouter, OpenAI-compatible) ------------------
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
_ORKEY=$(grep -m1 '^OPENROUTER_API_KEY=' "$SCRATCH/DermArena_data/.env" | cut -d= -f2)
export OPENROUTER_API_KEY="$_ORKEY"   # score_dx / grade_dxtest judge read THIS name
export OPENAI_API_KEY="$_ORKEY"       # base-MLLM / backbone client
export DERMARENA_BASE_MLLM=qwen/qwen3.5-27b
export DERMARENA_JUDGE_MODEL=google/gemini-2.5-flash   # NOT gemini-3-flash (invalid id → all-zero scores)

# --- budget + ledgers (NEVER /mnt/hdd, which doesn't exist on M3) --------------
export DERMARENA_BUDGET_USD=10
export DERMARENA_LEDGER="$CEREBRA/cerebra_cache/dermarena/ledger.json"
export LLM_LEDGER_PATH="$CEREBRA/cerebra_cache/dermarena/llm_ledger.jsonl"

# --- DermArena eval-harness image root (score_dx / infer_dx _resolve) ----------
export DERMARENA_REPO_ROOT="$DERMARENA_DATA"

# --- grading artifacts --------------------------------------------------------
export DERMARENA_HYPERNYMS="$SCRATCH/DermArena/dataset_collection/data/benchmark/icd11_hypernym.json"

mkdir -p "$(dirname "$LLM_LEDGER_PATH")"
