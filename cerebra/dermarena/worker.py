"""Persistent Cerebra pipeline worker — loads the vision tools ONCE, stays resident.

The base model is OpenRouter (no GPU); only PanDerm + DermoGPT + MedGemma need the GPU,
and they all fit on one L40S/A100. So we hold one GPU, load the tools once, and service
every iteration through a filesystem queue — no requeue, no model reload per run.

Protocol (drop a request JSON into $CEREBRA_REQ_DIR):
  {"task":"rds","jsonl":"<benchmark.jsonl>","limit":1,"out":"<preds.jsonl>","thinking":false}
The worker writes predictions to "out" then touches "<out>.done".
Stop with:  touch $CEREBRA_REQ_DIR/STOP   (or scancel the job).
"""
import glob
import importlib
import json
import os
import time
import traceback

from cerebra.dermarena.case import load_cases
from cerebra.dermarena.llm import OpenRouterMLLM
import cerebra.dermarena.select as _sel
import cerebra.dermarena.candidates as _cand
import cerebra.dermarena.verify as _ver
import cerebra.dermarena.pipeline as _pipe


def _reload_prompts():
    """Hot-reload the pure-Python step modules so PROMPT edits (select/candidates/verify)
    take effect on the next request WITHOUT reloading the GPU tools. run.py (torch tools)
    is imported lazily inside run_case and stays cached → tools remain warm."""
    for m in (_sel, _cand, _ver, _pipe):
        importlib.reload(m)
    return _pipe.run_case

REQ_DIR = os.environ.get("CEREBRA_REQ_DIR",
                         "/fs04/scratch2/ub62/ssim0070/Cerebra/cerebra_cache/dermarena/warm/requests")
os.makedirs(REQ_DIR, exist_ok=True)

TOOLS: dict = {}  # shared across requests → models load on the FIRST run, then stay warm

print(f"[worker] up — tools load lazily on the first request; polling {REQ_DIR}", flush=True)

while True:
    if os.path.exists(os.path.join(REQ_DIR, "STOP")):
        print("[worker] STOP — exiting.", flush=True)
        break
    reqs = sorted(glob.glob(os.path.join(REQ_DIR, "*.json")))
    if not reqs:
        time.sleep(2)
        continue
    req_path = reqs[0]
    try:
        req = json.load(open(req_path))
    except Exception as e:  # noqa: BLE001
        print(f"[worker] bad request {req_path}: {e}", flush=True)
        os.remove(req_path)
        continue
    os.remove(req_path)  # claim it

    run_case = _reload_prompts()  # pick up any prompt edits (tools stay warm)
    mllm = OpenRouterMLLM(thinking=bool(req.get("thinking", False)))
    out = req["out"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    t0 = time.time()
    n = 0
    with open(out, "w") as f:
        for c in load_cases(req["task"], jsonl_path=req.get("jsonl"), limit=req.get("limit")):
            try:
                rec = run_case(c, mllm, tools=TOOLS, do_run=True)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                rec = {"_id": c.id, "task": c.task, "prediction": "", "prediction_list": [],
                       "error": f"{type(e).__name__}: {e}", "n_findings": 0,
                       "ground_truth": c.ground_truth.get("diagnosis"),
                       "diagnostic_test_gt": c.ground_truth.get("diagnostic_test_gt"),
                       "cumulative_usd": mllm.ledger.spent()}
            f.write(json.dumps(rec) + "\n"); f.flush()
            n += 1
    open(out + ".done", "w").write("ok")
    print(f"[worker] served {req['task']} ({n} case) -> {out}  [{time.time()-t0:.0f}s]", flush=True)
