# Cerebra — Agentic Pipeline Walkthrough

_Step-by-step walk of how the Cerebra agentic system actually runs, grounded in the
source. Companion to `PROGRESS.md` (goal + architecture) and `CLAUDE.md` (working
rules). Last updated: 2026-07-02._

## The big picture: a two-tier agentic loop

Cerebra (forked from OctoTools) is an **orchestrator → specialist-agents → fusion**
system. There are two nested "plan → act → verify" loops:

- **Outer loop (orchestrator tier):** `SuperAgent` picks *which specialist agent* to
  run next.
- **Inner loop (agent tier):** each specialist (`LightweightAgent` subclass) picks
  *which tool* to run next.
- **Tools** are the leaves — they load models and do the actual work (MedGemma,
  PanDerm, etc.), returning a `Dataset`.

Two data objects flow through: tools emit `Dataset`; agents/orchestrator pass
`Metadata` (`cerebra/utils/metadata.py`). Everything is threaded by *saved file
paths*, not raw values.

```
run_cerebra.py
  └─ SuperAgent.run()                         [OUTER LOOP: pick an agent]
       observe → analyze → [plan → execute → verify]*
         └─ <Specialist>Agent.run()           [INNER LOOP: pick a tool]
              reason_and_execute():
                plan → reason → execute → evaluate (loop)
                  └─ Tool.execute()           [LEAF: load model, do work → Dataset]
         └─ SummaryAgent.run()                [FUSION: debate → final prediction]
```

---

## Step 0 — Entry point

`tasks/run_cerebra.py` parses CLI args (`--patient_id`, `--query`, `--llm_engine`,
`--file_paths`), builds the task string, then:

```python
orchestrator = SuperAgent(llm_engine_name=args.llm_engine)   # default gpt-4o
result_metadata = orchestrator.run(task=..., patient_id=..., file_paths=...)
```

It pulls `final_orchestration` out of the returned `Metadata` and prints/saves it.
(`run_cerebra.py:89-116`)

---

## Step 1 — SuperAgent construction

`SuperAgent.__init__` (`cerebra/orchestrator/super_agent.py:7-21`) wraps
`BaseOrchestrator` with a fixed agent roster: `ehr_agent`, `note_agent`,
`image_agent`, `summary_agent`, `max_steps=15`.

Inside `BaseOrchestrator.__init__` (`cerebra/orchestrator/base.py:18-90`):
- `_load_agents` dynamically imports and instantiates each enabled agent.
- `_get_agent_metadata` collects each agent's `register_agent_capabilities()` → this
  is the "menu" the planner reads to choose an agent.
- Builds the orchestrator-level `Planner`, `Memory`, `Executor`, and the LLM engine.

---

## Step 2 — The orchestrator run loop (outer loop)

`BaseOrchestrator.run` (`cerebra/orchestrator/base.py:360-413`) is
`OBSERVE → ANALYZE → [PLAN → EXECUTE → VERIFY]*`:

1. **OBSERVE** (`base.py:115`) — LLM extracts the patient ID from the task; sets a
   hardcoded `data_info` describing available modalities (EHR/Note/Image).
   ⚠️ This `data_info` is still boilerplate/legacy text, not derived from the actual
   DermArena case.
2. **ANALYZE** (`base.py:164`) — `planner.analyze_agent_goal` produces a one-shot
   strategy for which modalities/agents to use.
3. **The loop** (up to `max_steps=15`):
   - **PLAN** (`base.py:180`) — `planner.generate_next_step` returns structured
     `{context, sub_goal, agent_name}`, choosing one agent. `agent_name == "STOP"`
     breaks the loop.
   - **EXECUTE** (`base.py:203`) — see Step 3.
   - **VERIFY** (`base.py:297`) — `planner.verificate_context` returns `STOP` or
     `CONTINUE`.

The `SuperAgent` system prompt (`super_agent.py:37-52`) nudges the planner to: detect
modalities → run modality agents → **always finish with SummaryAgent** for the final
prediction.

---

## Step 3 — EXECUTE: handing off to a specialist agent

`BaseOrchestrator.execute` (`cerebra/orchestrator/base.py:203-292`) does the wiring:
- For `summary_agent`, it reuses `self.current_metadata` (the accumulated results).
  For any other agent, it calls `DataAgent.run(mode="local", ...)` to load that
  agent's input files into a `Metadata`. (`base.py:217-227`)
- Runs the agent: `agent_instance.run(task=sub_goal, input_metadata=...)`.
- Stashes the result under `result_dict_merged[f"{agent}_outputs"]`, plus a hardcoded
  "grab image_path" hack (`base.py:236-250` — flagged as a hack in the code), then
  rebuilds `current_metadata` so the next agent (and eventually SummaryAgent) sees
  everything accumulated.
- Records the step into `Memory` so the planner's next PLAN/VERIFY can reason about
  history.

---

## Step 4 — Inside a specialist agent (inner loop)

Every real agent subclasses `LightweightAgent` (`cerebra/agents/lightweight_agent.py`).
The `ImageAgent` (`cerebra/agents/image_agent.py`) just declares `enabled_tools` and
calls `self.reason_and_execute(task, input_metadata)`.

`reason_and_execute` (`lightweight_agent.py:243-441`) is a **plan-then-iterate** loop
(max 5 iterations):

1. **Overall plan** (`_create_overall_plan`, `lightweight_agent.py:171`) — LLM produces
   an XML multi-step plan from the tool menu (`initializer.toolbox_metadata`). Includes
   a guard: don't plan inference before a trained model exists.
2. **Reason** (`_reason_about_task_with_context`, `lightweight_agent.py:443`) — picks
   ONE tool + parameters as XML, given the plan + all previous attempts. Retries up to
   10× if no tool is chosen.
3. **Execute** (`_execute_tool`, `lightweight_agent.py:689`) — builds a
   `tool.execute(...)` command string and hands it to the `Executor`.
4. **Evaluate** (`_evaluate_execution_result`, `lightweight_agent.py:575`) — LLM judges
   `is_tool_completed` / `is_task_completed` / `needs_plan_revision`. Two
   self-correction mechanisms: evaluator-triggered replan, and a **safety net** that
   force-resets the plan after 3 consecutive failures (`lightweight_agent.py:285-307`).
5. Loop until `is_task_completed`, then return a `Metadata` with the merged execution
   context.

---

## Step 5 — Tools: discovery + execution

**Discovery** (`Initializer`, `cerebra/agents/modules/initializer.py:37-93`): walks
`cerebra/tools/<agent_name>/**/tool.py`, imports every class ending in `Tool`, and
instantiates each **with no args** to harvest metadata. This is why the contract
(`cerebra/tools/base.py`) demands: `__init__` takes no required args and **must not
load weights** — models lazy-load inside `execute()`. Only files literally named
`tool.py` are discovered (helpers like `modality_router.py` are safely ignored).

**Execution** (`Executor.execute_tool_command`,
`cerebra/agents/modules/executor.py:87-177`): imports
`tools.<agent>.<tool>.tool`, instantiates the class, injects the execution context as
local vars, and `exec()`s the `tool.execute(...)` string under a SIGALRM timeout. So
string params like `dataset['test_data']['saved_path']` resolve against the live
context.

**A real tool** — `MedGemmaDermAnalyzerTool`
(`cerebra/tools/image_agent/medgemma_derm_analyzer/tool.py`): sets metadata in
`__init__`, lazy-loads MedGemma-1.5-4b-it from local weights on first `execute()`, runs
the derm-analysis prompt per image, and returns `Dataset.create_agent_output(...)` with
per-image `findings`. (`tool.py:159-214`)

---

## Step 6 — Fusion: SummaryAgent → final answer

`SummaryAgent.run` (`cerebra/agents/summary_agent.py:342`) calls
`_multi_agent_debate_analysis` (`summary_agent.py:391`), which uses the
`Multi_Agent_Debate_Tool` (`cerebra/tools/summary_agent/multi_agent_debate/tool.py`) to
fuse every specialist's evidence (`_extract_agent_evidence`) + the narrative into one
final prediction. This becomes `final_orchestration` in the returned `Metadata` that
`run_cerebra.py` prints.

---

## The modality router (DermArena core idea)

`cerebra/tools/image_agent/modality_router.py` (pure Python, tested) groups a case's
`images[]` by target route:

| Image modality              | Route             | Tool                        |
|-----------------------------|-------------------|-----------------------------|
| clinical_photo, dermoscopy  | `DERM_SPECIALIST` | PanDerm + DermoGPT-RL        |
| everything else + `unknown` | `GENERAL_VLM`     | MedGemma-1.5-4b-it (workhorse) |

Key policy: `unknown` (~40% of images, a first-class VLM label, not missing data) routes
to the general VLM by design — sending a mislabeled histopath/radiology image into
PanDerm (closed-set) is worse than useless. PanDerm output is **evidence/differential**,
never the bound final dx.

---

## Two data abstractions (a known wart)

- Tools return `Dataset` (`cerebra/utils/dataset.py`) via `Dataset.create_agent_output`.
- Agents/orchestrator pass `Metadata` (`cerebra/utils/metadata.py`).
- `Dataset` asserts `set(data.keys()) == set(feature_description.keys())` and assumes
  list-valued (tabular) features — awkward for per-case QA. Flagged for refactor.

---

## Status: DermArena-ready vs. legacy scaffold

Per `PROGRESS.md`. The loop machinery, the modality router, and the MedGemma tool are
done. **Still legacy / TODO:**

- `ImageAgent` still enables the old MRI tools (`image_model_trainer`,
  `image_model_inference`) and describes itself as an MRI agent — it does **not** yet
  apply the router or call MedGemma/PanDerm.
- PanDerm + DermoGPT-RL tools not yet ported from DermAgent (LangChain → Cerebra
  `BaseTool`).
- The orchestrator's `data_info`, the "image_path hack," and the `Dataset` tabular-list
  assumption are all flagged warts for the per-case QA setting.
- MedGemma tool is code-complete but **GPU-untested** (login node has a broken torch
  env; test in the serving env).

### Cruft to watch
- `BaseAgent` (`agents/base.py`) and `LightweightAgent` both exist; real agents subclass
  `LightweightAgent`. `BaseAgent` passes kwargs the current `Planner` signature doesn't
  accept — treat `BaseAgent` as stale.
- `BaseAgent` is also re-exported as an alias of `BaseOrchestrator` at the bottom of
  `orchestrator/base.py` — different thing, same name.
