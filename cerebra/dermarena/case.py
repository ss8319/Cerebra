"""Case object + JSONL loader for the DermArena dx-track.

A `Case` is one benchmark row. It exposes ONLY what the model is allowed to see
(masked narrative + supplied images + tables + exam results) via `context_text()`
and `resolved_images()`, and keeps ground truth quarantined in `.ground_truth` for
grading — never fed to the model.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

# task_type is NOT in the row — it's determined by which benchmark file the row came from.
TASK_FILES = {
    "rds": "MM_RDS_benchmark.jsonl",     # screening: narrative + images -> dx
    "rdc": "MM_RDC_benchmark.jsonl",     # confirmation: + shown test results -> dx
    "dxtest": "DxTest_benchmark.jsonl",  # pre-test presentation + images -> test to order
}

DEFAULT_BASE_DIR = "/fs04/scratch2/ub62/ssim0070/dermarena_dx_v2"

# Ground-truth keys to quarantine (hidden from the model, used only for grading).
_GT_KEYS = (
    "diagnosis", "diagnostic_test_gt", "molecular_test_class", "molecular_test_modalities",
    "icd11_code", "icd11_title", "icd11_chapter", "icd11_match_status",
    "orphanet_id", "orphanet_name", "is_rare",
)


@dataclass
class Case:
    id: str
    task: str                       # "rds" | "rdc" | "dxtest"
    base_dir: str
    case_report: str = ""
    tables: List[Dict[str, Any]] = field(default_factory=list)
    examination_results: Optional[Any] = None   # rdc / dxtest only
    images: List[Dict[str, Any]] = field(default_factory=list)   # raw image dicts
    image_status: Optional[str] = None
    ground_truth: Dict[str, Any] = field(default_factory=dict)   # quarantined
    raw: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- factory
    @classmethod
    def from_row(cls, row: Dict[str, Any], task: str, base_dir: str = DEFAULT_BASE_DIR) -> "Case":
        gt = {k: row.get(k) for k in _GT_KEYS if k in row}
        return cls(
            id=str(row.get("_id", "")),
            task=task,
            base_dir=base_dir,
            case_report=row.get("case_report") or "",
            tables=row.get("tables") or [],
            examination_results=row.get("examination_results"),
            images=row.get("images") or [],
            image_status=row.get("image_status"),
            ground_truth=gt,
            raw=row,
        )

    # ---------------------------------------------------------------- model view
    def resolved_images(self) -> List[Dict[str, Any]]:
        """Image dicts with `image_path` joined to an absolute path (relative to base_dir)."""
        out = []
        for im in self.images:
            rel = im.get("image_path")
            ab = os.path.join(self.base_dir, rel) if rel and not os.path.isabs(rel) else rel
            out.append({**im, "abs_path": ab, "exists": bool(ab) and os.path.isfile(ab)})
        return out

    def context_text(self) -> str:
        """The text the model is allowed to see: narrative + tables + exam results.

        NEVER includes ground truth. Tables/exam results are rendered compactly.
        """
        parts = [f"CASE REPORT:\n{self.case_report.strip()}"]
        if self.tables:
            rendered = []
            for i, t in enumerate(self.tables, 1):
                cap = t.get("caption") or f"Table {i}"
                rows = t.get("structured_rows")
                rendered.append(f"[{cap}] " + (json.dumps(rows, ensure_ascii=False) if rows else ""))
            parts.append("TABLES:\n" + "\n".join(rendered))
        if self.examination_results:
            er = self.examination_results
            parts.append("EXAMINATION RESULTS:\n" + (er if isinstance(er, str) else json.dumps(er, ensure_ascii=False)))
        return "\n\n".join(parts)

    @property
    def has_images(self) -> bool:
        return len(self.images) > 0


def load_cases(
    task: str,
    base_dir: str = DEFAULT_BASE_DIR,
    limit: Optional[int] = None,
    jsonl_path: Optional[str] = None,
) -> Iterator[Case]:
    """Stream `Case` objects from a task's benchmark JSONL.

    Args:
        task: one of "rds" | "rdc" | "dxtest".
        base_dir: dataset folder (image paths resolve against this).
        limit: stop after N cases (dev/sanity).
        jsonl_path: explicit path (e.g. a materialized dev subset); overrides task file.
    """
    if task not in TASK_FILES:
        raise ValueError(f"Unknown task '{task}'. Expected one of {list(TASK_FILES)}.")
    path = jsonl_path or os.path.join(base_dir, TASK_FILES[task])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            yield Case.from_row(json.loads(line), task=task, base_dir=base_dir)
