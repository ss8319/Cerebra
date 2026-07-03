"""Candidate DDx generation for the closed-set classifiers (PanDerm / DermoGPT).

When PanDerm (or DermoGPT in diagnosis mode) is selected, Qwen-27B produces a top-N
differential from ALL available context — case text + tables + images. That list is
passed to PanDerm as its candidate set; PanDerm then RANKS the list by visual
similarity to the image. (Qwen proposes; PanDerm visually re-ranks.)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from cerebra.dermarena.case import Case
from cerebra.dermarena.llm import OpenRouterMLLM
from cerebra.dermarena.parse import extract_json

SYSTEM = ("You are an expert dermatologist generating a differential diagnosis for a "
          "patient case. A specialist image classifier will visually re-rank your list.")

USER_TEMPLATE = """{context}

Based on the case narrative, tables, AND the images shown, list the {n} most likely
diagnoses (a differential), most likely first. Cover distinct disease families and use
specific clinical diagnosis names — this list will be visually re-ranked against the image.
Return ONLY a JSON array of {n} diagnosis strings: ["dx1", "dx2", ..., "dx{n}"]."""


def propose_candidates(case: Case, mllm: Optional[OpenRouterMLLM] = None,
                       n: int = 10, max_tokens: Optional[int] = None) -> Dict[str, Any]:
    """Return {candidates: [top-n dx], raw}. Sees text + tables + images."""
    mllm = mllm or OpenRouterMLLM()
    present = [im for im in case.resolved_images() if im.get("exists")]
    user = USER_TEMPLATE.format(context=case.context_text(), n=n)
    res = mllm.chat(system=SYSTEM, text=user,
                    image_paths=[im["abs_path"] for im in present], max_tokens=max_tokens)
    arr = extract_json(res.text)
    cands: List[str] = []
    if isinstance(arr, list):
        cands = [str(x).strip() for x in arr if str(x).strip()]
    return {"candidates": cands[:n], "raw": res.text,
            "usage": {"cumulative_usd": res.cumulative_usd}}
