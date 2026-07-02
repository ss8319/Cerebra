# Cerebra × DermArena — PROGRESS

_High-level working doc. Last updated: 2026-07-02._

## Goal
Build an **agentic diagnostic system on the Cerebra scaffold** that solves the
DermArena dx-track, and beats the DermAgent + MedAgent-Pro baselines.

Cerebra already gives us a two-tier architecture (orchestrator `SuperAgent` →
specialist agents → `SummaryAgent` fusion) with planner/executor/memory loops.
We **keep the scaffold, swap the modality agents** for dermatology image agents.

## Dataset (DermArena dx-track)
- 3 tasks: **MM_RDS** (screening: text+images→dx), **MM_RDC** (confirmation:
  +shown test results→dx), **DxTest** (pre-test presentation→which test to order).
- ~14.8k PMC case reports. Masked narrative + `images[]` (per-image `modality`
  label) + `tables[]` + (RDC/DxTest) `examination_results`. GT = free-text
  `diagnosis` → ICD-11 / Orphanet. Long tail: 2000+ dx, 37% singletons, rare-heavy.
- Image modality mix: **unknown 39.8%**, clinical_photo 26%, histopathology 16.4%,
  radiology 8.4%, ihc 5.9%, dermoscopy 1.8%, others ~2%.
- Path: `/fs04/scratch2/ub62/ssim0070/dermarena_dx_v2/`. Sanity viewer:
  `v2_sanity_n15_v2.html`.

## Approach — image agent + modality router
Replace Cerebra's EHR/note agents with an **image agent that routes per-image**:

| Image modality | Tool |
|---|---|
| clinical_photo, dermoscopy | **PanDerm** (DermLIP zero-shot) + **DermoGPT-RL** (Qwen3-VL-8B VQA) |
| histopath, radiology, ihc, unknown, other | **MedGemma-1.5-4b-it** (general medical VLM) |

- PanDerm / DermoGPT-RL tool wrappers exist in the **DermAgent** repo
  (`skin_agent/tools/skin_tools.py`, LangChain `BaseTool`) — must be re-wrapped
  to Cerebra's `tools/base.py` interface.
- MedGemma weights: `/fs04/scratch2/ub62/ssim0070/models/medgemma-1.5-4b-it`.
- SummaryAgent fuses per-image findings + narrative + tables → final answer.

## Target agent design (DermArena Agent v1)
Built fresh for per-case QA, borrowing only Cerebra's `BaseTool`/`Dataset` contract +
the multi-agent debate. NOT a faithful port of Cerebra's dynamic orchestrator.

**Key reframe:** in the masked "supply-and-don't-bind" setting, image findings are
stripped from the narrative — so text alone CANNOT produce a real DDx. Vision is
necessary; the diagnosis emerges from fusion over image findings + narrative, never
from a text-only step.

```
 0. LOAD      row → case (report + images + exam/tables + task_type)
 1. ANALYSE   CLASSIFY each image's modality with the base MLLM (do NOT trust the
             dataset label — ~40% are `unknown`), then route → tool plan
             (`classify.py`; classification runs non-thinking/cheap)
 2. PROPOSE   base MLLM (sees images + report) → candidate diagnosis list  ★LLM
 3. RUN       clinical_photo/dermoscopy → PanDerm + DermoGPT re-rank the candidates;
              other + unknown → MedGemma describes                     (deterministic exec)
 4. DEBATE    multi-agent debate over candidates + all image findings + narrative
              → task head: RDS/RDC = ranked top-5 dx; DxTest = test(s) to order  ★LLM
 5. EMIT      predictions JSONL in grader format + postchecks
```

**Decisions locked:**
- Observe is dropped; ANALYSE = parse + route (no patient_id, no DataAgent file loading).
- PanDerm is closed-set → its candidates come from Step 2's **base MLLM** proposal
  (multimodal, image-grounded — honors "no DDx from text").
- **Base MLLM (Step 2 proposer) = `qwen/qwen3.5-27b` via OpenRouter** (confirmed
  input_modalities = text+image+video). base_url `https://openrouter.ai/api/v1`, key in
  `DermArena_data/.env`; reach via Cerebra's `litellm` engine (`openrouter/qwen/qwen3.5-27b`).
  Sampling: registry `Qwen/Qwen3.5-27B` → use `nonthinking_general` to protect budget.
  Cost ~$0.195/M in, $1.56/M out ≈ **~$0.0014/case**. **Budget cap: USD $10 for now**
  (~dev300 pass ≈ $0.40; track with the openrouter-cost skill).
- Keep multi-agent debate for fusion (Cerebra concept), task-branched.
- Only Steps 2 & 4 are real LLM calls; 1 & 3 are deterministic.

**Dropped from Cerebra:** 15-step dynamic orchestrator loop, inner reason_and_execute
plan/replan loop + "no-inference-before-training" guard, DataAgent, ehr/note agents,
*_trainer/*_inference tools, patient_id, vllm setup.

## Key design decisions / open questions
1. **MedGemma is the workhorse** (~72% of images), not a fallback. Size infra for it.
2. **Modality field is unreliable** (39.8% `unknown`) — router needs a policy /
   cheap modality classifier for `unknown`.
3. **PanDerm is closed-set** — use as evidence/differential, never bind final dx.
4. **Text prior dominates** — always run a text path; images are supplements.
   The win must be shown over a text-only baseline.
5. **Serving/VRAM** — 3 weight sets + orchestrator LLM. Separate endpoints vs
   lazy load/unload (M3 + Apptainer; serve-model-sglang workflow). TBD.
6. **Sampling params** per creator spec (DermoGPT-RL, MedGemma) — wire into registry.
7. **RAG = leakage risk** vs PMC source articles — start WITHOUT RAG.
8. **Three output heads** (dx vs test) must conform to existing ICD-hypernym +
   LLM-judge grader. Reuse seed-pinned dev subset (~300); postcheck every run.
9. **Orchestrator backbone** — point SuperAgent off gpt-4o default → gemini-2.5-flash;
   Qwen3.6-27B as free judge.

## Framework understanding (how Cerebra actually works)
Read of the core contracts, to reuse rather than fight them:

- **Tool contract** (`cerebra/tools/base.py`): subclass `BaseTool`, name the class
  `*Tool`, call `set_metadata(...)`, implement `execute(*args, **kwargs)`. Tools
  return a `Dataset` (`cerebra/utils/dataset.py`) via `Dataset.create_agent_output(...)`.
- **Tool auto-discovery** (`agents/modules/initializer.py`): the Initializer walks
  `cerebra/tools/<agent_name>/**/tool.py`, imports every class ending in `Tool`.
  Two gotchas we must respect:
  1. It instantiates each tool class **with NO args** during discovery
     (`run_demo_commands`) → `__init__` must not require `model_string` and must not
     load weights. **Lazy-load models inside `execute()`** (MRI_Analyzer_Tool does this).
  2. Only files literally named `tool.py` are discovered — helper modules
     (e.g. `modality_router.py`) are safely ignored.
- **Planner loop** (`agents/modules/planner.py`): plan→execute→reflect. `analyze_goal`
  → `generate_next_step` (picks agent/tool + subgoal, structured output `NextStep`) →
  executor runs → `verificate_context` (STOP/CONTINUE) → `generate_final_output`.
- **Two data abstractions coexist**: tools return `Dataset`; agents/orchestrator pass
  `Metadata` (`utils/metadata.py`). `Dataset` asserts
  `set(data.keys()) == set(feature_description.keys())` and assumes list-valued
  (tabular) features. This is a wart for per-case QA — flagged for refactor.
- **Cruft to watch**: `BaseAgent` (agents/base.py) and `LightweightAgent` both exist;
  real agents (e.g. `image_agent`) subclass `LightweightAgent`. `BaseAgent` passes
  `available_tools/toolbox_metadata` kwargs the current `Planner` signature doesn't
  accept — treat `BaseAgent` as stale.
- **Local weights mandatory**: compute nodes have no internet — models load from local
  paths, never HF hub ids. MedGemma-1.5 loads via `AutoModelForImageTextToText` +
  `AutoProcessor` from `/fs04/.../models/medgemma-1.5-4b-it` (bf16, chat template).

## Status
- [x] Understand DermArena dx-track structure + modality mix.
- [x] Confirm PanDerm / DermoGPT-RL wrappers exist in DermAgent; MedGemma weights present.
- [x] Understand Cerebra tool/agent/planner contracts + discovery rules + gotchas.
- [x] **Modality router** — `cerebra/tools/image_agent/modality_router.py` (tested; pure-python).
- [x] **MedGemma tool port** — `cerebra/tools/image_agent/medgemma_derm_analyzer/tool.py`
      (local weights, lazy load, derm prompt, returns `Dataset`; code done, GPU-untested
      — login node has a broken numpy/torch env, must test in serving env).
- [x] **Port PanDerm + DermoGPT-RL** from DermAgent (LangChain → Cerebra `BaseTool`):
      `panderm_classifier/tool.py`, `dermogpt_vqa/tool.py` (code done, GPU-untested).
- [x] **Pin FM inference params per provider docs** (see `CLAUDE.md` rule):
      - MedGemma-1.5: greedy `do_sample=False`, bf16 (README + generation_config).
      - DermoGPT-RL: sampling `do_sample=True, temp=0.7, top_p=0.8, top_k=20`, **float16**
        (its generation_config; overrides DermAgent's greedy+bf16, which ignored its own temp).
      - PanDerm: provider template `"This is a skin image of {}"` + model's own 224px/CLIP transform.
- [x] **Step 0-1 LOAD + ANALYSE** — `cerebra/dermarena/{case,analyse}.py`. Tested on all 3
      tasks: correct routing, text-only handling, mixed-modality split, GT-leak assertion passes.
- [x] **Step 2 PROPOSE** — `cerebra/dermarena/{llm,propose}.py`. `qwen/qwen3.5-27b` via
      OpenRouter (multimodal), provider sampling params, file-backed **budget ledger + hard
      $10 cap** (`cerebra_cache/dermarena/ledger.json`). Tested live (~$0.0003/case).
- [ ] **Step 3 RUN** (needs GPU): execute the ANALYSE plan — feed PROPOSE candidates into
      PanDerm/DermoGPT (clinical/dermoscopy), MedGemma for other/unknown → per-image findings.
- [ ] **Step 4 DEBATE** → task heads (RDS/RDC top-5 dx, DxTest test) → EMIT predictions JSONL.
- [x] **Grader schema pinned** (via research of DermArena/dataset_collection/eval/): graders
      read only `_id` + `prediction`, where `prediction` is a SINGLE STRING — RDS/RDC
      (score_dx.py): `"1. dx;"` per line; DxTest (grade_dxtest.py): tests under
      `Recommended Tests:`. EMIT now conforms (`_format_prediction`). Judge = LLM via OpenRouter.
- [x] **Fixed Qwen thinking bug**: Qwen3.5-27B left thinking exhausts max_tokens → empty
      content + ~13x cost. `reasoning:{enabled:False}` (1175→32 tokens). Now ~$0.0018/case.
- [x] **Dev300 subsets** — `dermarena_dx_v2/dev_slices/{rds,rdc,dxtest}_dev300.jsonl` (300
      shared cases, seed 20260702, + manifest) via `make_dev_subset.py`.
- [x] **Client concurrency** — pipeline `--workers` (thread-safe ledger; forced to 1 when GPU RUN on).
- [x] **Step 1 = modality CLASSIFICATION** (`classify.py`) — don't trust dataset labels;
      base MLLM re-derives each image's modality, routing uses that. Tested (agrees on
      clean cases; real value on the ~40% `unknown`). Toggle: `--no-classify`.
- [x] **Reasoning toggle** (`--thinking`) — HF thinking params + 3k budget.
- [x] **Compute nodes have internet** (verified openrouter_http=200) → whole pipeline +
      grading runs in ONE GPU sbatch; no phase-split / local serving needed.
- [~] **n=1-per-task GPU+reasoning validation submitted** — `slurm/dermarena_n1_validate.sbatch`
      (job 58075205): full pipeline on 1 shared case/task → grade (score_dx +hypernyms / grade_dxtest).
- [x] **n=1 GPU validation ran** (job 58075205, L40S, 28 min, full pipeline + reasoning).
      Findings on case pmid_27484467_1 (GT=IBD, small-bowel histopath — a non-derm case):
      - ✅ Classification correct (both images → histopathology → MedGemma, not derm specialists).
      - ✅ Vision ran; RDC top-1 = Crohn's disease (clinically an IBD).
      - ❌ **`--thinking` returns EMPTY content on most calls** (PROPOSE + all experts empty;
        moderator carried it) — reasoning exceeds the 3000 budget. FIX: retry-on-empty
        (reasoning-off) in `llm.py`. RDS moderator also truncated → blank prediction.
      - ❌ Grading errored: `/mnt/hdd` perm denied — forgot `LLM_LEDGER_PATH` (known gotcha).
        FIX: set in sbatch; re-graded on login node → works (scored=1).
      - 📊 Grade 0/0/0, but INSIGHT: predicting a SUBTYPE (Crohn's) of the gold umbrella
        (IBD) gets NO family credit (hypernym is broader-term only). Strategy implication.
- [ ] Re-run n=1 with fixes (retry-on-empty + ledger path); then dev300 + postchecks.

## Caveats surfaced
- **PanDerm license conflict**: DermLIP README body says `cc-by-nc-nd-4.0` (non-commercial,
  no-derivatives) while its YAML header says `cc-by-4.0`. Clarify before any release/publication.
- All FM tools **GPU-untested** — validate in `/fs04/scratch2/ub62/ssim0070/dermagent/bin/python`
  on a GPU node (login-node `~/.local` torch is broken).
- SGLang registry's MedGemma entry is for MedGemma **1** (`medgemma-4b-it`, sampling temp 0.6),
  NOT the 1.5 checkpoint we use — don't apply it to 1.5.

## Related repos
- Cerebra (this repo) — orchestrator scaffold (built from OctoTools).
- DermAgent — `/fs04/scratch2/ub62/ssim0070/DermAgent` (PanDerm/DermoGPT-RL tools, baseline).
- DermArena data — `/fs04/scratch2/ub62/ssim0070/dermarena_dx_v2`.
