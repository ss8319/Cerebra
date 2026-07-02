"""PROPOSE step: base MLLM (Qwen3.5-27B, multimodal) proposes candidate diagnoses.

The narrative is masked (no image descriptions), so candidates must be image-grounded
— the proposer SEES the images + reads the report. Its output seeds the closed-set
specialist tools (PanDerm / DermoGPT diagnosis mode) and enters the debate. It is a
candidate list, NOT a committed final DDx.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from cerebra.dermarena.case import Case
from cerebra.dermarena.llm import OpenRouterMLLM, LLMResult

SYSTEM_PROMPT = (
    "You are an expert dermatologist and diagnostician. You are shown a clinical case "
    "narrative and its medical images (clinical photos, dermoscopy, histopathology, "
    "radiology, etc.). The narrative has had its image descriptions removed, so you MUST "
    "rely on the images for visual findings. Propose a broad, well-reasoned list of "
    "candidate diagnoses to investigate — favour breadth and cover distinct disease "
    "families. This is a differential to explore, not a final answer."
)

USER_TEMPLATE = """{context}

Based on the narrative AND the images shown, list the most plausible candidate diagnoses.
Return ONLY a JSON array of {n} diagnosis name strings, most likely first, e.g.
["diagnosis one", "diagnosis two", ...]. Use specific clinical diagnosis names."""


def _parse_candidates(text: str, k: int) -> List[str]:
    """Extract a JSON array of diagnosis strings; fall back to line parsing."""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            cands = [str(x).strip() for x in arr if str(x).strip()]
            if cands:
                return cands[:k]
        except json.JSONDecodeError:
            pass
    # fallback: numbered/bulleted lines
    lines = []
    for ln in text.splitlines():
        ln = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", ln).strip().strip('",')
        if ln and len(ln) < 120:
            lines.append(ln)
    return lines[:k]


def propose_candidates(
    case: Case,
    mllm: Optional[OpenRouterMLLM] = None,
    n: int = 8,
    max_tokens: Optional[int] = None,  # None -> inherit client budget (3000 thinking / 1024 not)
) -> Dict[str, Any]:
    """Return {candidates, raw, usage} for a case using the base MLLM."""
    mllm = mllm or OpenRouterMLLM()
    present = [im["abs_path"] for im in case.resolved_images() if im.get("exists")]
    user = USER_TEMPLATE.format(context=case.context_text(), n=n)

    res: LLMResult = mllm.chat(
        system=SYSTEM_PROMPT, text=user, image_paths=present, max_tokens=max_tokens,
    )
    candidates = _parse_candidates(res.text, n)
    return {
        "case_id": case.id,
        "task": case.task,
        "candidates": candidates,
        "n_images_seen": len(present),
        "raw_response": res.text,
        "usage": {
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
            "call_cost_usd": res.call_cost_usd,
            "cumulative_usd": res.cumulative_usd,
        },
    }
