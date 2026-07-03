"""Materialize seed-pinned dev subsets for the DermArena dx-track.

Picks a SHARED set of case `_id`s present in all three task files (so the same cases
are evaluated across RDS/RDC/DxTest — comparable, matches the prior dev50 convention),
seed-pinned for reuse across models. Writes one dev JSONL per task.

    PY=/fs04/scratch2/ub62/ssim0070/dermagent/bin/python
    $PY -m cerebra.dermarena.make_dev_subset --n 300 --seed 20260702
"""
from __future__ import annotations

import argparse
import json
import os
import random

from cerebra.dermarena.case import TASK_FILES, DEFAULT_BASE_DIR


def _ids_in(path):
    ids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line).get("_id"))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260702)
    ap.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    ap.add_argument("--out-dir", default=None, help="default: <base-dir>/dev_slices")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.base_dir, "dev_slices")
    os.makedirs(out_dir, exist_ok=True)

    paths = {t: os.path.join(args.base_dir, f) for t, f in TASK_FILES.items()}
    id_sets = {t: _ids_in(p) for t, p in paths.items()}
    for t, s in id_sets.items():
        print(f"{t}: {len(s)} cases")

    shared = set.intersection(*id_sets.values())
    print(f"shared across all 3 tasks: {len(shared)}")

    n = min(args.n, len(shared))
    rng = random.Random(args.seed)
    chosen = set(rng.sample(sorted(shared), n))  # sorted() => deterministic given seed
    print(f"sampling {n} shared cases (seed={args.seed})")

    for t, p in paths.items():
        out = os.path.join(out_dir, f"{t}_dev{n}.jsonl")
        kept = 0
        with open(p) as fin, open(out, "w") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                if json.loads(line).get("_id") in chosen:
                    fout.write(line + "\n")
                    kept += 1
        print(f"  wrote {kept} -> {out}")

    # Record provenance so the subset is reproducible / reusable.
    meta = {"n": n, "seed": args.seed, "task_files": TASK_FILES, "chosen_ids": sorted(chosen)}
    with open(os.path.join(out_dir, f"dev{n}_manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"manifest -> {os.path.join(out_dir, f'dev{n}_manifest.json')}")


if __name__ == "__main__":
    main()
