# Session notes — 2026-07-02 (DermArena agent)

Read this with fresh eyes tomorrow. Companion docs: `PROGRESS.md` (living status),
`CLAUDE.md` (rules), `debate_redesign.html` (the pending change). PR: **#1**
(`ss8319/Cerebra`, branch `dermarena-agent-wip`).

---

## 1. The goal (locked this session)
Benchmark matrix on **DermArena dx_v2** (3 tasks: RDS/RDC/DxTest):

| System | Backbone (held constant) |
|---|---|
| Qwen3.5-27B **bare** (single call) | qwen/qwen3.5-27b |
| **DermAgent** | qwen/qwen3.5-27b |
| **Our Agent** (Cerebra dermarena) | qwen/qwen3.5-27b |

Then vary backbone: **+ Qwen3.5-9B, + GPT-5**. Fair rule: same model, same dev set,
same grader. Holding 27B constant isolates the *agent-scaffold* effect.

## 2. What we built (all in `Cerebra/cerebra/dermarena/`)
A lean per-case pipeline (NOT Cerebra's legacy dynamic orchestrator):

```
0 LOAD → 1 ANALYSE(classify+route) → 2 PROPOSE → 3 RUN(vision) → 4 DEBATE → 5 EMIT
```
- `case.py` loader (GT quarantined), `analyse.py` router, `propose.py` (Qwen candidates),
  `run.py` (vision tools), `debate.py` (panel fusion), `pipeline.py` driver, `llm.py`
  (OpenRouter client + $10 budget ledger), `classify.py`, `make_dev_subset.py`,
  `render_traces.py` (HTML trace viewer).
- 3 ported vision tools in `cerebra/tools/image_agent/`: MedGemma-1.5, PanDerm, DermoGPT-RL
  (+ `modality_router.py`).

## 3. Key design decisions
- **Don't trust the dataset `modality` label** → Step 1 re-classifies each image with the
  base MLLM (~40% are `unknown`). Routing uses the classified modality.
- **DDx must come from vision, not text** — the narrative is masked (image findings removed),
  so a text-only differential is invalid. PROPOSE is multimodal.
- **PanDerm is closed-set** → seed it with PROPOSE candidates; it re-ranks them (evidence,
  never a bound answer).
- **Base MLLM = `qwen/qwen3.5-27b`** via OpenRouter (multimodal, cheap). Reasoning is a
  toggle (`--thinking`); OFF by default.
- **Grader contract:** `prediction` must be a single STRING — RDS/RDC `"1. dx;"` per line;
  DxTest under `Recommended Tests:`. (Not a list — that scores 0.)
- **Dev set = stratified dev500** (`dermarena_dx_v2/stratified/*_dev500.jsonl`, seed=42,
  v1 rules: image_status×is_rare×chapter strata + disease_cap 3/journal_cap 16).

## 4. Bugs found + fixed (each is now a committed fix)
1. **Thinking empty-content** — Qwen reasoning ate the token budget → empty PROPOSE/experts/
   RDS prediction + ~13x cost. Fix: retry-on-empty (reasoning-off fallback) in `llm.py`.
   Now a standing RULE in `CLAUDE.md`.
2. **Grader `/mnt/hdd` crash** — forgot `LLM_LEDGER_PATH` (known M3 gotcha). Fixed in sbatch.
3. **Transient OpenRouter non-JSON response** killed the whole GPU job. Fix: retry-with-backoff
   in `llm.py` + per-case isolation in `pipeline.py` + resilient sbatch.
4. **prediction emitted as a list** → would score 0. Fix: `_format_prediction` → grader string.

## 5. What's validated
- ✅ Pipeline CPU/API path (classify, propose, debate, both task heads, grader-format output).
- ✅ **Vision path on GPU (L40S)** — classifier split schematic vs clinical_photo per image;
  PanDerm re-ranked candidates (SCLE @0.90); strong prediction (GT "Drug-induced SCLE" →
  pred #1 "Erlotinib-induced SCLE"). This was the "use the image tools" run.
- ⏳ Full n=1 grade across all 3 tasks: **job 58077821 was PENDING at session end** — check it
  tomorrow (`squeue -j 58077821`; results under `cerebra_cache/dermarena/n1_validate/`).

## 6. OPEN — decide/act tomorrow
- **Dynamic debate panel (APPROVED, not yet built).** Current debate = 3 hardcoded personas
  (Dermatologist/Dermatopathologist/Internist) that mismatch non-derm cases and just agree.
  Plan (see `debate_redesign.html`): new `panel.py` selects 2–4 case-specific specialists from
  modalities + candidates (GI→gastroenterologist, histo→pathologist…), each defends a DIFFERENT
  candidate (adversarial), skip debate on decisive cases. Keep `--static-panel` for A/B.
  → Build this before scaling to dev500.
- **Candidate parse nit:** PROPOSE sometimes captures a prose preamble as candidate #1
  (model prefixes text before the JSON array). Tighten `_parse_candidates`.
- **Grading semantics insight:** predicting a SUBTYPE (Crohn's) of the gold umbrella (IBD) gets
  NO family credit (hypernym = broader-term only). May want the debate to also surface umbrella
  terms. Watch this when reading dev500 scores.

## 7. How to run things (env matters)
- **Python:** `/fs04/scratch2/ub62/ssim0070/dermagent/bin/python` (login-node `~/.local` torch is broken).
- **Pipeline (CPU/API):** `PYTHONPATH=. $PY -m cerebra.dermarena.pipeline --task rds --jsonl <dev500> --limit N [--thinking] [--no-run] [--workers K]`
- **GPU full run:** `sbatch slurm/dermarena_n1_validate.sbatch` (L40S; classify+RUN+thinking+grade).
- **View traces:** `PYTHONPATH=. $PY -m cerebra.dermarena.render_traces preds.jsonl --out t.html`
- **Budget:** ledger at `cerebra_cache/dermarena/ledger.json`, cap `$DERMARENA_BUDGET_USD` ($10).
- **Grader ledger:** always `export LLM_LEDGER_PATH=<writable>` (not /mnt/hdd).

## 8. Suggested first moves tomorrow
1. `squeue -j 58077821` → if done, read grades + `render_traces` the 3 cases.
2. Build the dynamic debate panel (`panel.py` + edits) per `debate_redesign.html`.
3. Fix the candidate-preamble parse nit.
4. Then: 3-system benchmark harness on dev500 (bare Qwen-27B vs DermAgent vs Our Agent).
