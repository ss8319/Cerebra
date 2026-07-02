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

## Other essentials
- **Env for running FMs:** `/fs04/scratch2/ub62/ssim0070/dermagent/bin/python`
  (torch 2.9.1+cu128, transformers 4.57.6). The login-node `~/.local` torch is broken.
- **Weights load from LOCAL paths only** (no compute-node internet); set `HF_HUB_OFFLINE=1`.
- **Tool contract** (`cerebra/tools/base.py`): subclass `BaseTool`, name the class `*Tool`,
  `set_metadata(...)`, `execute(...)` returns a `Dataset`. The Initializer instantiates
  each tool **with no args** during discovery, so **lazy-load weights inside `execute()`**,
  never in `__init__`.
