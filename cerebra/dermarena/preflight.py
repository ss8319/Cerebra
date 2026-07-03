"""Preflight validator — catch the whole class of silent setup bugs BEFORE a run.

Anticipates every failure we actually hit: dead /mnt/hdd defaults, missing weights,
unresolvable images, an invalid judge model, missing keys, missing python deps.
Run it after `source slurm/env.sh`; exits non-zero (loud) on any hard failure.

    source Cerebra/slurm/env.sh && $PY -m cerebra.dermarena.preflight
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

FAILS: list = []
WARNS: list = []


def ok(m): print(f"  ✅ {m}")
def fail(m): FAILS.append(m); print(f"  ❌ {m}")
def warn(m): WARNS.append(m); print(f"  ⚠️  {m}")


def env(name, must_path=False, forbid=None):
    v = os.environ.get(name)
    if not v:
        fail(f"${name} not set"); return None
    if forbid and forbid in v:
        fail(f"${name}={v}  (contains dead default '{forbid}')"); return v
    if must_path and not os.path.exists(v):
        fail(f"${name}={v}  (path does not exist)"); return v
    ok(f"${name} = {v}")
    return v


def main():
    print("== keys & paths ==")
    key = env("OPENROUTER_API_KEY")
    env("OPENAI_API_KEY")
    env("OPENAI_BASE_URL")
    env("MEDGEMMA_PATH", must_path=True)
    env("DERMOGPT_PATH", must_path=True)
    env("DERM1M_SRC", must_path=True)
    env("HF_HOME", must_path=True)
    led = env("LLM_LEDGER_PATH", forbid="/mnt/hdd")
    env("DERMARENA_LEDGER", forbid="/mnt/hdd")
    data = env("DERMARENA_REPO_ROOT", must_path=True)
    env("DERMARENA_HYPERNYMS", must_path=True)
    judge = os.environ.get("DERMARENA_JUDGE_MODEL", "google/gemini-2.5-flash")
    base = os.environ.get("DERMARENA_BASE_MLLM", "qwen/qwen3.5-27b")

    # ledger dir writable
    if led:
        d = os.path.dirname(led) or "."
        try:
            os.makedirs(d, exist_ok=True)
            t = os.path.join(d, ".wtest"); open(t, "w").close(); os.remove(t)
            ok(f"ledger dir writable: {d}")
        except Exception as e:  # noqa: BLE001
            fail(f"ledger dir NOT writable: {d} ({e})")

    print("== weights & data ==")
    dermlip = os.path.join(os.environ.get("HF_HOME", ""), "hub",
                           "models--redlessone--DermLIP_PanDerm-base-w-PubMed-256")
    (ok if os.path.isdir(dermlip) else fail)(f"DermLIP/PanDerm weights: {dermlip}")

    if data:
        for f in ("MM_RDS_benchmark.jsonl", "MM_RDC_benchmark.jsonl", "DxTest_benchmark.jsonl"):
            p = os.path.join(data, f)
            (ok if os.path.isfile(p) else fail)(f"benchmark {f}")
        try:  # a real image must resolve under DERMARENA_REPO_ROOT
            row = json.loads(open(os.path.join(data, "MM_RDS_benchmark.jsonl")).readline())
            imgs = row.get("images") or []
            if imgs:
                ip = os.path.join(data, imgs[0]["image_path"])
                (ok if os.path.isfile(ip) else fail)(
                    f"sample image resolves ({os.path.basename(ip)})"
                    if os.path.isfile(ip) else
                    f"sample image NOT found: {ip}  (fix DERMARENA_REPO_ROOT)")
        except Exception as e:  # noqa: BLE001
            warn(f"couldn't test a sample image: {e}")

    if "--no-deps" in sys.argv:
        print("== python deps == (skipped --no-deps; torch import is slow on login node)")
    else:
        print("== python deps ==")
        for m in ("openai", "torch", "transformers", "PIL"):
            try:
                __import__(m); ok(f"import {m}")
            except Exception as e:  # noqa: BLE001
                fail(f"import {m}: {e}")

    print("== OpenRouter models (network) ==")
    if key:
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/models",
                                         headers={"Authorization": f"Bearer {key}"})
            ids = {m["id"] for m in json.load(urllib.request.urlopen(req, timeout=20))["data"]}
            for mdl in (base, judge):
                (ok if mdl in ids else fail)(f"model on OpenRouter: {mdl}")
        except Exception as e:  # noqa: BLE001
            warn(f"couldn't verify models (network): {e}")
    else:
        warn("no key → skipped model check")

    print()
    if FAILS:
        print(f"PREFLIGHT FAILED — {len(FAILS)} hard issue(s):")
        for m in FAILS:
            print("   -", m)
        sys.exit(1)
    print(f"PREFLIGHT PASSED" + (f"  ({len(WARNS)} warning(s))" if WARNS else ""))


if __name__ == "__main__":
    main()
