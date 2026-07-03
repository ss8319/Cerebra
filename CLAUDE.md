# Cerebra — working rules

Cerebra is being adapted (from OctoTools) into an agentic framework for the
DermArena dx-track. See `PROGRESS.md` for goal + architecture, `FM_REPRODUCTION.md`
for verified FM loading recipes.

## RULE: Always run foundation models per their provider's documented setup

Whenever we integrate or invoke a foundation model (LLM / VLM / vision encoder —
MedGemma, DermoGPT-RL, PanDerm/DermLIP, Qwen, UNI2-h, CONCHv1.5, etc.), **before
writing or trusting any inference code, check the model provider's own docs** — the
HF repo (README, `generation_config.json`, `config.json`, `preprocessor_config.json`,
`chat_template.jinja`) and/or the provider's GitHub — for the *ideal* setup:

- **Sampling params** — `do_sample`, `temperature`, `top_p`, `top_k`,
  `repetition_penalty`, `max_new_tokens`. Use the creator's recommended values; do
  not invent defaults or silently inherit HF library defaults.
- **Loading idiom** — the correct `AutoModel*` class, dtype (usually bf16), device_map,
  attention impl, and any required `timm_kwargs` / custom builder.
- **Preprocessing** — chat-template usage and content-block format for VLMs; exact
  resize/crop/normalization for vision encoders.

Why: small setup differences (wrong sampling params, missing chat template, wrong
image transform) silently degrade quality and make results non-reproducible and
unfair to compare. Getting the provider-recommended setup right is a cheap, high-value
step.

How to apply:
- Local model files are the first source of truth (`generation_config.json` etc.);
  they exist offline under the local weight dirs (compute nodes have NO internet).
- This aligns with the DermArena eval policy (every model uses its creator's official
  sampling params). Central registry: `sglang/slurm/sampling_params.py`.
- In tool code, don't hardcode a guessed value — pull from the registry / model config,
  and leave a `TODO` if a value is still a placeholder.

## RULE: Anticipate thinking/reasoning-budget exhaustion

Reasoning ("thinking") models (Qwen3.x, o-series, etc.) spend tokens on hidden CoT
BEFORE emitting the answer. If reasoning consumes the whole `max_tokens` budget, the
model returns **empty `content`** — and you still PAY for the wasted reasoning tokens
(~10-40x cost). This is silent: the call "succeeds" with a blank answer, so a pipeline
can limp on with empty intermediate steps (dead debate, empty candidate lists).

Whenever you enable reasoning:
- **Budget generously** — reasoning needs thousands of tokens on top of the answer;
  a 512-token cap will reliably return empty. Size `max_tokens` for reasoning + answer.
- **Detect + retry on empty** — if `content` is blank, retry once with reasoning OFF
  (guarantees a usable answer, stops paying for wasted thinking). Implemented in
  `cerebra/dermarena/llm.py` (`OpenRouterMLLM.chat` retry-on-empty).
- **Verify intermediates, not just the final output** — inspect the trace
  (`show_trace.py`): if PROPOSE/experts are blank but the final answer is populated,
  the reasoning is being thrown away. A green final result can hide empty steps.
- **Cost-check** — thinking is ~10-40x the tokens; confirm the budget cap still holds
  (`DERMARENA_BUDGET_USD` / the ledger) before a full run.

Sampling params for thinking vs non-thinking differ — use the matching preset per the
FM-docs rule above (Qwen3.5-27B thinking = temp 1.0/top_p 0.95; nonthinking = temp 0.7/top_p 0.8).

## RULE: API base model + GPU tools → hold ONE persistent GPU worker, don't batch-per-run

When the base/reasoning model is an API (OpenRouter/Qwen etc.) and only the *tools* need a
GPU (PanDerm, DermoGPT, MedGemma — all fit on one L40S/A100), **do NOT submit a fresh batch
job per run.** Each batch job re-queues (queue wait) AND reloads the 8B+4B model weights
(~2-3 min) every time — pure waste, since the GPU is idle between the brief tool inferences.

Instead: **hold ONE GPU with a persistent resident worker** that loads the tools once and
services every iteration via a filesystem request queue. Drive iterations by dropping a
request file; no re-queue, no reload.
- Worker: `cerebra/dermarena/worker.py` · sbatch: `slurm/cerebra_worker.sbatch` (multi-hour hold)
- Run an iteration: `bash slurm/cerebra_infer.sh <task> <jsonl> <limit> <out.jsonl>`
- Stop it: `touch $CEREBRA_REQ_DIR/STOP` (frees the GPU — do this when done iterating).
- Same pattern for DermAgent: `slurm/dermagent_worker.sbatch` + `warm_infer.sh`.

Caveat: a long walltime is harder to backfill on a full cluster (needs a long gap). Balance
hold-length vs start-time; when the cluster is saturated, no job config conjures a free GPU.

## Other essentials
- **Env for running FMs:** `/fs04/scratch2/ub62/ssim0070/dermagent/bin/python`
  (torch 2.9.1+cu128, transformers 4.57.6). The login-node `~/.local` torch is broken.
- **Weights load from LOCAL paths only** (no compute-node internet); set `HF_HUB_OFFLINE=1`.
- **Tool contract** (`cerebra/tools/base.py`): subclass `BaseTool`, name the class `*Tool`,
  `set_metadata(...)`, `execute(...)` returns a `Dataset`. The Initializer instantiates
  each tool **with no args** during discovery, so **lazy-load weights inside `execute()`**,
  never in `__init__`.
