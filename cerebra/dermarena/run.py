"""RUN step: execute the ANALYSE tool plan and collect normalized image findings.

Tool classes are imported LAZILY (they pull in torch/transformers) so this module is
safe to import on a CPU/login node — only calling `run_plan` on a GPU actually loads
weights. Instances are cached in `tools` so a batch run loads each model once.
"""
from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional

from cerebra.dermarena.case import Case

TOOL_IMPORTS = {
    "PanDermClassifierTool": ("cerebra.tools.image_agent.panderm_classifier.tool", "PanDermClassifierTool"),
    "DermoGPTVQATool": ("cerebra.tools.image_agent.dermogpt_vqa.tool", "DermoGPTVQATool"),
    "MedGemmaDermAnalyzerTool": ("cerebra.tools.image_agent.medgemma_derm_analyzer.tool", "MedGemmaDermAnalyzerTool"),
}

# Short names the SELECT step uses -> tool classes.
SHORT_TO_CLASS = {
    "panderm": "PanDermClassifierTool",
    "dermogpt": "DermoGPTVQATool",
    "medgemma": "MedGemmaDermAnalyzerTool",
}


def _get_tool(name: str, cache: Dict[str, Any]):
    if name not in cache:
        mod, cls = TOOL_IMPORTS[name]
        cache[name] = getattr(importlib.import_module(mod), cls)()
    return cache[name]


def _normalize(name: str, ds) -> List[Dict[str, Any]]:
    """Flatten a tool's Dataset output into per-image {tool, image_path, summary} dicts."""
    data = ds.get_dataset()["dataset"]
    out: List[Dict[str, Any]] = []
    if name == "PanDermClassifierTool":
        for path, ranked, scores in zip(data.get("image_path", []), data.get("ranked_diseases", []), data.get("scores", [])):
            summary = "; ".join(f"{d} ({s:.2f})" for d, s in zip(ranked, scores))
            out.append({"tool": name, "image_path": path, "summary": f"PanDerm visual ranking — {summary}"})
    elif name == "DermoGPTVQATool":
        for path, resp, final in zip(data.get("image_path", []), data.get("response", []), data.get("final_diagnosis", [])):
            out.append({"tool": name, "image_path": path, "summary": resp, "final_diagnosis": final})
    elif name == "MedGemmaDermAnalyzerTool":
        for path, mod, find in zip(data.get("image_path", []), data.get("modality", []), data.get("findings", [])):
            out.append({"tool": name, "image_path": path, "modality": mod, "summary": find})
    else:
        out.append({"tool": name, "summary": str(data)})
    return out


def run_plan(
    case: Case,
    plan: Dict[str, Any],
    candidates: Optional[List[str]] = None,
    tools: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Execute every tool call in the plan; return a flat list of normalized findings.

    `candidates` (from PROPOSE) seed the closed-set specialists. `tools` is an optional
    shared instance cache to reuse loaded models across cases.
    """
    tools = tools if tools is not None else {}
    cand_str = ", ".join(candidates) if candidates else None
    findings: List[Dict[str, Any]] = []

    for tc in plan.get("tool_calls", []):
        name, paths = tc["tool"], tc["image_paths"]
        tool = _get_tool(name, tools)
        try:
            if name == "PanDermClassifierTool":
                # Closed-set: rank PROPOSE candidates (or default list if none).
                ds = tool.execute(image_paths=paths, candidate_diseases=cand_str)
            elif name == "DermoGPTVQATool":
                if cand_str:
                    ds = tool.execute(image_paths=paths, candidate_diseases=cand_str)  # diagnosis mode
                else:
                    ds = tool.execute(image_paths=paths, query="Describe the lesion and give a differential.")
            elif name == "MedGemmaDermAnalyzerTool":
                ds = tool.execute(
                    image_paths=paths,
                    case_context=case.context_text()[:2000],
                    modality=tc.get("modalities"),
                )
            else:
                continue
            findings.extend(_normalize(name, ds))
        except Exception as e:  # a tool failure must not kill the case
            findings.append({"tool": name, "image_path": None, "summary": f"[{name} error: {e}]"})

    return findings


def run_calls(case: Case, tool_calls: List[Dict[str, Any]],
              tools: Optional[Dict[str, Any]] = None, max_retries: int = 2) -> List[Dict[str, Any]]:
    """Execute Qwen-SELECTED tool calls, with up to `max_retries` retries per tool (Step 2).

    tool_calls: [{"tool": <short>, "image_paths": [...], "candidate_diseases": [...]}]
    """
    tools = tools if tools is not None else {}
    findings: List[Dict[str, Any]] = []
    for tc in tool_calls:
        short = tc.get("tool")
        cls_name = SHORT_TO_CLASS.get(short)
        if not cls_name:
            findings.append({"tool": short, "summary": None, "error": f"unknown tool '{short}'"})
            continue
        tool = _get_tool(cls_name, tools)
        paths = tc.get("image_paths") or []
        cand = ", ".join(tc.get("candidate_diseases") or []) or None
        last_err = None
        for attempt in range(max_retries + 1):  # 1 try + up to max_retries
            try:
                if short == "panderm":
                    ds = tool.execute(image_paths=paths, candidate_diseases=cand)
                elif short == "dermogpt":
                    ds = (tool.execute(image_paths=paths, candidate_diseases=cand) if cand
                          else tool.execute(image_paths=paths, query="Describe the lesion and give a differential."))
                else:  # medgemma
                    ds = tool.execute(image_paths=paths, case_context=case.context_text()[:2000])
                for fnd in _normalize(cls_name, ds):
                    fnd["tool"] = short                         # short name in the trace
                    fnd["candidate_diseases"] = tc.get("candidate_diseases") or []
                    findings.append(fnd)
                last_err = None
                break
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
        if last_err is not None:
            findings.append({"tool": short, "image_path": None, "summary": None,
                             "error": f"[{short} failed after {max_retries} retries: {last_err}]"})
    return findings
