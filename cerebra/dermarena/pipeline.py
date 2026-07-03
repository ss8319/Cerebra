"""DermArena agent pipeline — simplified SELECT -> EXECUTE -> VERIFY(confirm|rerun).

  Step 1 SELECT : Qwen sees case + images + tables -> chooses which tools to run.
  Step 2 EXECUTE: run the selected tools (max 2 retries per tool).
  Step 3 VERIFY : Qwen sees case + tool outputs -> CONFIRM (emit dx / test) or RERUN one
                  tool (max 1 rerun of the loop). EMIT the prediction string.

CLI:
    PY=/fs04/scratch2/ub62/ssim0070/dermagent/bin/python
    PYTHONPATH=. $PY -m cerebra.dermarena.pipeline --task rds --limit 5 --out preds.jsonl
    # --no-run skips the GPU tools (verify straight from the narrative)
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

from cerebra.dermarena.case import Case, load_cases
from cerebra.dermarena.select import select_tools
from cerebra.dermarena.verify import verify_or_rerun
from cerebra.dermarena.llm import OpenRouterMLLM


def _format_prediction(task: str, answer: Any) -> str:
    """Render the answer as the STRING the DermArena grader reads (`prediction` field).

    score_dx.py (RDS/RDC): top-5, one per line, "1. Name;". grade_dxtest.py (DxTest):
    tests under a literal "Recommended Tests:" header. Grader joins on `_id` and parses
    this string — a list/object would score as zero.
    """
    items = answer if isinstance(answer, list) else [answer]
    items = [str(x).strip() for x in items if str(x).strip()]
    if task in ("rds", "rdc"):
        return "\n".join(f"{i}. {dx};" for i, dx in enumerate(items[:5], 1))
    return "Recommended Tests:\n" + "\n".join(f"- {t}" for t in items)


def run_case(
    case: Case,
    mllm: OpenRouterMLLM,
    tools: Optional[Dict[str, Any]] = None,
    do_run: bool = True,
    max_rerun: int = 1,
) -> Dict[str, Any]:
    # Step 1 — SELECT: Qwen chooses which tools to run on which images.
    sel = select_tools(case, mllm=mllm)
    present = sel["present_images"]
    tool_calls = sel["tool_calls"]

    # Step 1b — CANDIDATES: if a closed-set classifier (PanDerm/DermoGPT) is selected,
    # Qwen produces a top-10 differential (from text + tables + images) that PanDerm ranks.
    candidates: List[str] = []
    candidates_raw = None
    if do_run and any(c["tool"] in ("panderm", "dermogpt") for c in tool_calls):
        from cerebra.dermarena.candidates import propose_candidates
        prop = propose_candidates(case, mllm=mllm, n=10)
        candidates = prop["candidates"]
        candidates_raw = prop["raw"]
        for c in tool_calls:
            if c["tool"] in ("panderm", "dermogpt"):
                c["candidate_diseases"] = candidates

    # Step 2 + 3 — EXECUTE selected tools, then VERIFY (confirm | rerun ≤ max_rerun).
    findings: List[Dict[str, Any]] = []
    rerun_log: List[Dict[str, Any]] = []
    verify: Dict[str, Any] = {}
    rerun_count = 0

    if do_run and tool_calls:
        from cerebra.dermarena.run import run_calls  # lazy: pulls torch only on the GPU
        findings = run_calls(case, tool_calls, tools=tools, max_retries=2)

    while True:
        verify = verify_or_rerun(case, findings, present, mllm=mllm,
                                 force_confirm=(rerun_count >= max_rerun))
        if verify["action"] == "rerun" and rerun_count < max_rerun and verify.get("rerun") and do_run:
            rr = verify["rerun"]
            idxs = rr.get("image_indices") or []
            paths = [present[i]["abs_path"] for i in idxs if isinstance(i, int) and 0 <= i < len(present)]
            call = {"tool": str(rr.get("tool", "")).strip().lower(),
                    "image_paths": paths or [im["abs_path"] for im in present],
                    "candidate_diseases": rr.get("candidate_diseases") or []}
            from cerebra.dermarena.run import run_calls
            findings += run_calls(case, [call], tools=tools, max_retries=2)
            rerun_log.append({"why": rr.get("why"), "call": call})
            rerun_count += 1
            continue
        break

    answer = verify.get("tests") if case.task == "dxtest" else verify.get("diagnoses")
    return {
        "_id": case.id,
        "task": case.task,
        "prediction": _format_prediction(case.task, answer or []),
        "prediction_list": answer or [],
        "n_findings": len(findings),
        "rerun_count": rerun_count,
        "ground_truth": case.ground_truth.get("diagnosis"),
        "diagnostic_test_gt": case.ground_truth.get("diagnostic_test_gt"),
        # image paths + selected modality-hint, kept for the trace viewer's image panel
        "case_images": {im["abs_path"]: im.get("modality") for im in present},
        "candidates": candidates,   # Qwen top-10 DDx fed to PanDerm (if closed-set selected)
        "trace": {
            "select_reasoning": sel.get("reasoning"),
            "select_raw": sel.get("raw"),
            "tool_calls": tool_calls,
            "candidates": candidates,
            "candidates_raw": candidates_raw,
            "findings": findings,
            "verify_action": verify.get("action"),
            "verify_raw": verify.get("raw"),
            "rerun_log": rerun_log,
        },
        "cumulative_usd": verify.get("usage", {}).get("cumulative_usd", 0.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["rds", "rdc", "dxtest"])
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--jsonl", default=None, help="explicit input JSONL (e.g. dev subset)")
    ap.add_argument("--out", default="dermarena_preds.jsonl")
    ap.add_argument("--no-run", action="store_true", help="skip GPU vision tools (verify from narrative only)")
    ap.add_argument("--thinking", action="store_true",
                    help="run Qwen3.5-27B with reasoning ON (HF thinking params + 3k token "
                         "budget). Better quality, ~13x cost — watch the $10 ledger cap.")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent cases in flight. >1 only with --no-run (GPU tools are "
                         "not thread-safe); the LLM ledger IS thread-safe.")
    args = ap.parse_args()

    mllm = OpenRouterMLLM(thinking=args.thinking)
    tools: Dict[str, Any] = {}  # shared model cache across cases (RUN)
    cases = list(load_cases(args.task, limit=args.limit, jsonl_path=args.jsonl))

    # Concurrency safety: GPU tools (RUN) share one model instance and are not
    # thread-safe, so parallelism is only allowed when RUN is skipped.
    workers = args.workers
    if workers > 1 and not args.no_run:
        print("[warn] --workers>1 requires --no-run (GPU tools not thread-safe); forcing workers=1")
        workers = 1

    def _work(c):
        # Never let one case abort a long run — record the error and continue.
        try:
            return run_case(c, mllm, tools=tools, do_run=not args.no_run)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"_id": c.id, "task": c.task, "prediction": "", "prediction_list": [],
                    "error": f"{type(e).__name__}: {e}", "n_findings": 0,
                    "ground_truth": c.ground_truth.get("diagnosis"),
                    "diagnostic_test_gt": c.ground_truth.get("diagnostic_test_gt"),
                    "cumulative_usd": mllm.ledger.spent()}

    records = []
    with open(args.out, "w") as fout:
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_work, c): c for c in cases}
                for fut in as_completed(futs):
                    rec = fut.result()
                    records.append(rec)
                    fout.write(json.dumps(rec) + "\n"); fout.flush()
                    print(f"[{rec['task']}] {rec['_id'][:18]:18} pred={rec['prediction']}  "
                          f"GT={rec['ground_truth']!r}  ${rec['cumulative_usd']:.4f}")
        else:
            for c in cases:
                rec = _work(c)
                records.append(rec)
                fout.write(json.dumps(rec) + "\n"); fout.flush()
                print(f"[{c.task}] {c.id[:18]:18} pred={rec['prediction']}  "
                      f"GT={rec['ground_truth']!r}  ${rec['cumulative_usd']:.4f}")
    if records:
        print(f"\nWrote {len(records)} predictions -> {args.out}  (spend ${records[-1]['cumulative_usd']:.4f})")


if __name__ == "__main__":
    main()
