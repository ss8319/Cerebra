"""Step 1 — Tool selection. Qwen sees the case + images + tools and decides which
specialist tools to dispatch on which images (and, for closed-set classifiers, the
candidate diagnoses to test). This is the agentic 'decompose + dispatch' step.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from cerebra.dermarena.case import Case
from cerebra.dermarena.llm import OpenRouterMLLM
from cerebra.dermarena.parse import extract_json

TOOL_MENU = """Available specialist tools:
- panderm  : zero-shot classifier for CLINICAL PHOTO or DERMOSCOPY skin-lesion images.
             Ranks a list of candidate diagnoses YOU provide by visual similarity.
             Requires candidate_diseases. Use only for skin lesions.
- dermogpt : dermatology vision-language model for CLINICAL PHOTO or DERMOSCOPY.
             Describes morphology and (given candidates) picks the best fit.
             candidate_diseases optional. Use only for skin lesions.
- medgemma : general medical image analyzer for ANY OTHER modality — histopathology,
             immunohistochemistry, radiology, gross pathology, immunofluorescence, etc.
             Grounds its reading in the case. Use for non-skin-photo images."""

SYSTEM = ("You are an expert dermatologist orchestrating specialist AI vision tools to "
          "diagnose a patient case. You decide which tools to run.")

USER_TEMPLATE = """{context}

IMAGES (decide which tool(s) to run on which; the modality label is only a hint —
judge from the image itself):
{image_list}

{tool_menu}

Decompose the case and choose the tool calls that will give you the diagnostic evidence
you need. Return ONLY JSON:
{{"reasoning": "<one or two sentences>",
  "tool_calls": [
    {{"tool": "panderm|dermogpt|medgemma", "image_indices": [<int>, ...],
      "candidate_diseases": ["<dx>", "..."]}}
  ]}}
Rules: image_indices refer to the IMAGES list above. candidate_diseases is REQUIRED for
panderm, helpful for dermogpt, and ignored by medgemma. Select multiple tools/images as
needed; skip images that add nothing. If there are no useful images, return an empty tool_calls."""


def select_tools(case: Case, mllm: Optional[OpenRouterMLLM] = None,
                 max_tokens: Optional[int] = None) -> Dict[str, Any]:
    """Return {tool_calls, reasoning, present_images, raw}."""
    mllm = mllm or OpenRouterMLLM()
    present = [im for im in case.resolved_images() if im.get("exists")]
    image_list = "\n".join(
        f"[{i}] modality_hint={im.get('modality')}  ({os.path.basename(im['abs_path'])})"
        for i, im in enumerate(present)
    ) or "(no images available)"
    user = USER_TEMPLATE.format(context=case.context_text(), image_list=image_list, tool_menu=TOOL_MENU)

    res = mllm.chat(system=SYSTEM, text=user,
                    image_paths=[im["abs_path"] for im in present], max_tokens=max_tokens)
    obj = extract_json(res.text) or {}

    calls: List[Dict[str, Any]] = []
    for tc in (obj.get("tool_calls") or []):
        tool = str(tc.get("tool", "")).strip().lower()
        if tool not in ("panderm", "dermogpt", "medgemma"):
            continue
        idxs = tc.get("image_indices") or []
        paths = [present[i]["abs_path"] for i in idxs if isinstance(i, int) and 0 <= i < len(present)]
        if not paths:  # default to all present images if the model omitted indices
            paths = [im["abs_path"] for im in present]
        calls.append({
            "tool": tool,
            "image_paths": paths,
            "candidate_diseases": [str(d).strip() for d in (tc.get("candidate_diseases") or []) if str(d).strip()],
        })
    return {
        "tool_calls": calls,
        "reasoning": obj.get("reasoning", ""),
        "present_images": present,
        "raw": res.text,
        "usage": {"cumulative_usd": res.cumulative_usd},
    }
