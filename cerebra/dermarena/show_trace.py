"""Human-readable trace viewer for DermArena agent predictions.

Renders the full agentic chain per case so you can inspect what the system did:
Step 1 modality classification (provided vs predicted) -> PROPOSE candidates ->
RUN tool findings -> DEBATE expert opinions -> moderator -> final prediction vs GT.

    PY=/fs04/scratch2/ub62/ssim0070/dermagent/bin/python
    PYTHONPATH=. $PY -m cerebra.dermarena.show_trace <preds.jsonl> [--full]
"""
from __future__ import annotations

import argparse
import json
import os


def _wrap(s, n=1_000_000):
    return (s or "").strip()


def show(rec: dict, full: bool = False):
    W = 78
    print("=" * W)
    print(f"CASE {rec.get('_id')}   task={rec.get('task')}   spend=${rec.get('cumulative_usd',0):.4f}")
    print(f"GROUND TRUTH dx : {rec.get('ground_truth')!r}")
    if rec.get("diagnostic_test_gt"):
        print(f"GROUND TRUTH test: {rec.get('diagnostic_test_gt')!r}")
    tr = rec.get("trace", {})

    # Step 1 — modality classification (don't trust the label)
    cls = rec.get("classified_modality") or {}
    if cls:
        print("\n── Step 1  MODALITY (base MLLM re-classification) ──")
        for path, pred in cls.items():
            print(f"   {os.path.basename(path):32} → {pred}")
    print(f"   routing: {tr.get('plan_routing')}")

    # Step 2 — PROPOSE candidates
    print("\n── Step 2  PROPOSE candidates (multimodal) ──")
    for i, c in enumerate(rec.get("candidates", []), 1):
        print(f"   {i}. {c}")
    if full and tr.get("propose_raw"):
        print("   [raw]:", _wrap(tr["propose_raw"])[:600])

    # Step 3 — RUN tool findings
    print("\n── Step 3  RUN tool findings ──")
    for f in tr.get("findings", []):
        head = f"[{f.get('tool')}]"
        if f.get("modality"):
            head += f" ({f['modality']})"
        if f.get("image_path"):
            head += f" {os.path.basename(f['image_path'])}"
        body = _wrap(f.get("summary"))
        print(f"   {head}")
        print("      " + (body if full else body[:400].replace("\n", "\n      ")))
        if f.get("final_diagnosis"):
            print(f"      → final_diagnosis: {f['final_diagnosis']}")
    if not tr.get("findings"):
        print("   (none — vision skipped or no images)")

    # Step 4 — DEBATE
    print("\n── Step 4  DEBATE ──")
    for o in tr.get("expert_opinions", []):
        op = _wrap(o.get("opinion"))
        print(f"   {o.get('role')}: {op if full else op[:300]}")
    if full and tr.get("moderator_raw"):
        print("   MODERATOR:", _wrap(tr["moderator_raw"])[:600])

    # Final
    print("\n── FINAL PREDICTION (grader string) ──")
    print("   " + (rec.get("prediction") or "").replace("\n", "\n   "))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", help="predictions JSONL from pipeline.py")
    ap.add_argument("--full", action="store_true", help="show full (untruncated) text + raw LLM outputs")
    args = ap.parse_args()
    with open(args.preds) as f:
        for line in f:
            line = line.strip()
            if line:
                show(json.loads(line), full=args.full)


if __name__ == "__main__":
    main()
