"""Step 3 — Confirm-or-rerun. Qwen sees the case + all tool outputs and either
CONFIRMS (emits the final diagnosis / test) or requests a RERUN of one tool.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from cerebra.dermarena.case import Case
from cerebra.dermarena.llm import OpenRouterMLLM
from cerebra.dermarena.parse import extract_json

SYSTEM = ("You are an expert dermatologist. You have the patient case and the outputs of "
          "specialist AI vision tools. Weigh the tool evidence against the narrative.")

_CONFIRM_FMT = {
    "rds": '{"action":"confirm","diagnoses":["dx1","dx2","dx3","dx4","dx5"]}  (top-5, most likely first)',
    "rdc": '{"action":"confirm","diagnoses":["dx1","dx2","dx3","dx4","dx5"]}  (top-5, most likely first)',
    "dxtest": '{"action":"confirm","tests":["test1","test2"]}  (diagnostic test(s) to order next)',
}

USER_TEMPLATE = """{context}

SPECIALIST TOOL FINDINGS:
{findings_block}

{decision}

Return ONLY JSON, one of:
- confirm: {confirm_fmt}
- rerun  : {{"action":"rerun","rerun":{{"tool":"panderm|dermogpt|medgemma","image_indices":[<int>...],"candidate_diseases":["<dx>"...],"why":"<what's missing>"}}}}"""

_DECIDE = ("Decide ONE: (A) CONFIRM if the evidence lets you determine the answer, or "
           "(B) RERUN one tool if the evidence is insufficient or conflicting.")
_FORCE = ("You have used your rerun budget. You MUST CONFIRM now — give your best answer "
          "from the evidence available.")


def _findings_block(findings: List[Dict[str, Any]]) -> str:
    if not findings:
        return "(no tool findings — reason from the narrative alone)"
    lines = []
    for f in findings:
        tag = f.get("tool", "tool")
        body = f.get("summary") or f.get("error") or ""
        lines.append(f"[{tag}] {str(body).strip()}")
    return "\n".join(lines)


def verify_or_rerun(case: Case, findings: List[Dict[str, Any]], present_images: List[Dict[str, Any]],
                    mllm: Optional[OpenRouterMLLM] = None, force_confirm: bool = False,
                    max_tokens: Optional[int] = None) -> Dict[str, Any]:
    mllm = mllm or OpenRouterMLLM()
    user = USER_TEMPLATE.format(
        context=case.context_text(),
        findings_block=_findings_block(findings),
        decision=_FORCE if force_confirm else _DECIDE,
        confirm_fmt=_CONFIRM_FMT.get(case.task, _CONFIRM_FMT["rds"]),
    )
    res = mllm.chat(system=SYSTEM, text=user,
                    image_paths=[im["abs_path"] for im in present_images], max_tokens=max_tokens)
    obj = extract_json(res.text) or {}
    action = str(obj.get("action", "")).strip().lower()
    if force_confirm and action != "confirm":
        action = "confirm"  # never allow another rerun once budget is spent
    return {
        "action": action or "confirm",
        "diagnoses": [str(d).strip() for d in (obj.get("diagnoses") or []) if str(d).strip()],
        "tests": [str(t).strip() for t in (obj.get("tests") or []) if str(t).strip()],
        "rerun": obj.get("rerun") if action == "rerun" else None,
        "raw": res.text,
        "usage": {"cumulative_usd": res.cumulative_usd},
    }
