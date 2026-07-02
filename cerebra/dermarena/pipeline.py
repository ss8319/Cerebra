"""End-to-end DermArena agent pipeline: LOAD -> ANALYSE -> PROPOSE -> RUN -> DEBATE -> EMIT.

CLI:
    PY=/fs04/scratch2/ub62/ssim0070/dermagent/bin/python
    PYTHONPATH=. $PY -m cerebra.dermarena.pipeline --task rds --limit 5 --out preds.jsonl
    # add --no-run to skip GPU vision tools (debate over narrative + candidates only)
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

from cerebra.dermarena.case import Case, load_cases
from cerebra.dermarena.analyse import build_plan
from cerebra.dermarena.propose import propose_candidates
from cerebra.dermarena.debate import debate
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
    revise: bool = False,
    classify: bool = True,
) -> Dict[str, Any]:
    # Step 1: classify modality (don't trust the dataset label), then route on it.
    modality_by_path = None
    if classify and case.has_images:
        from cerebra.dermarena.classify import classify_case
        # non-thinking client (perception task), shares the ledger for budget accounting
        classify_mllm = OpenRouterMLLM(thinking=False, ledger=mllm.ledger)
        modality_by_path = classify_case(case, mllm=classify_mllm)
    plan = build_plan(case, modality_by_path=modality_by_path)
    proposal = propose_candidates(case, mllm=mllm)
    candidates = proposal["candidates"]

    findings: List[Dict[str, Any]] = []
    if do_run and plan["tool_calls"]:
        from cerebra.dermarena.run import run_plan  # lazy: pulls torch only when needed
        findings = run_plan(case, plan, candidates, tools=tools)

    result = debate(case, candidates, findings, mllm=mllm, revise=revise)
    return {
        "_id": case.id,
        "task": case.task,
        # grader-facing field: a single formatted string
        "prediction": _format_prediction(case.task, result["prediction"]),
        # structured copy for our own analysis (ignored by grader)
        "prediction_list": result["prediction"],
        "candidates": candidates,
        "n_findings": len(findings),
        "ground_truth": case.ground_truth.get("diagnosis"),
        "diagnostic_test_gt": case.ground_truth.get("diagnostic_test_gt"),
        "classified_modality": modality_by_path,
        "trace": {
            "plan_routing": plan["routing"],
            "used_classified_modality": plan["used_classified_modality"],
            "propose_raw": proposal.get("raw_response"),
            "findings": findings,
            "expert_opinions": result["expert_opinions"],
            "moderator_raw": result["moderator_raw"],
        },
        "cumulative_usd": result["usage"]["cumulative_usd"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["rds", "rdc", "dxtest"])
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--jsonl", default=None, help="explicit input JSONL (e.g. dev subset)")
    ap.add_argument("--out", default="dermarena_preds.jsonl")
    ap.add_argument("--no-run", action="store_true", help="skip GPU vision tools (debate only)")
    ap.add_argument("--no-classify", action="store_true",
                    help="trust the dataset modality label instead of re-classifying (Step 1 off)")
    ap.add_argument("--revise", action="store_true", help="add a debate revision round")
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
            return run_case(c, mllm, tools=tools, do_run=not args.no_run, revise=args.revise,
                            classify=not args.no_classify)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"_id": c.id, "task": c.task, "prediction": "", "prediction_list": [],
                    "error": f"{type(e).__name__}: {e}", "candidates": [], "n_findings": 0,
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
