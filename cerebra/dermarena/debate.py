"""DEBATE step: multi-agent fusion over all evidence -> task-branched final answer.

Experts with distinct lenses each reason over the SAME evidence (narrative + PROPOSE
candidates + per-image tool findings), then a moderator synthesizes the final answer in
the format the task's grader expects. Text-only (vision was already extracted upstream),
so it's cheap. The DDx is BORN here from combined evidence — never a text-only prior.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from cerebra.dermarena.case import Case
from cerebra.dermarena.llm import OpenRouterMLLM

EXPERTS = [
    ("Dermatologist", "You are a board-certified dermatologist reasoning from clinical morphology."),
    ("Dermatopathologist", "You are an expert dermatopathologist weighing histopathology/IHC and radiology findings."),
    ("Internist", "You are a general internist alert to systemic, infectious, and rare (Orphanet) causes."),
]


def _evidence_block(case: Case, candidates: List[str], findings: List[Dict[str, Any]]) -> str:
    parts = [case.context_text()]
    if candidates:
        parts.append("CANDIDATE DIAGNOSES proposed from images + narrative:\n- " + "\n- ".join(candidates))
    if findings:
        lines = []
        for f in findings:
            tag = f.get("tool", "tool")
            mod = f" ({f['modality']})" if f.get("modality") else ""
            lines.append(f"[{tag}{mod}] {str(f.get('summary','')).strip()}")
        parts.append("IMAGE ANALYSIS FINDINGS:\n" + "\n".join(lines))
    else:
        parts.append("IMAGE ANALYSIS FINDINGS: (none — reason from narrative + candidates)")
    return "\n\n".join(parts)


def _task_instruction(task: str) -> str:
    if task in ("rds", "rdc"):
        return ('Return ONLY a JSON array of the 5 most likely FINAL diagnoses, most likely '
                'first: ["dx1","dx2","dx3","dx4","dx5"]. Use specific clinical diagnosis names.')
    if task == "dxtest":
        return ('Return ONLY a JSON object naming the diagnostic test(s) to order NEXT: '
                '{"tests": ["test1","test2"]}. Order by priority.')
    raise ValueError(f"Unknown task {task}")


def _parse_final(task: str, text: str) -> Any:
    if task in ("rds", "rdc"):
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                return [str(x).strip() for x in json.loads(m.group(0)) if str(x).strip()][:5]
            except json.JSONDecodeError:
                pass
        return []
    else:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj.get("tests", []) if isinstance(obj, dict) else []
            except json.JSONDecodeError:
                pass
        return []


def debate(
    case: Case,
    candidates: List[str],
    findings: List[Dict[str, Any]],
    mllm: Optional[OpenRouterMLLM] = None,
    n_experts: int = 3,
    revise: bool = False,
    max_tokens: Optional[int] = None,  # None -> inherit client budget (thinking-safe)
) -> Dict[str, Any]:
    """Run the debate; return {prediction, expert_opinions, moderator_raw, usage}."""
    mllm = mllm or OpenRouterMLLM()
    evidence = _evidence_block(case, candidates, findings)
    experts = EXPERTS[:n_experts]

    total_cost = 0.0
    opinions: List[Dict[str, str]] = []
    for role, persona in experts:
        prompt = (f"{evidence}\n\nAs the {role}, give your single most likely diagnosis (or, for "
                  f"a test-selection task, the key test to order) and a 2-3 sentence justification "
                  f"grounded in the evidence above.")
        r = mllm.chat(system=persona, text=prompt, max_tokens=max_tokens)
        total_cost = r.cumulative_usd
        opinions.append({"role": role, "opinion": r.text.strip()})

    if revise:
        peer_txt = "\n\n".join(f"{o['role']}: {o['opinion']}" for o in opinions)
        revised = []
        for (role, persona), o in zip(experts, opinions):
            prompt = (f"{evidence}\n\nPEER OPINIONS:\n{peer_txt}\n\nAs the {role}, reconsider given "
                      f"your peers. Give your final diagnosis/test and a 1-2 sentence reason.")
            r = mllm.chat(system=persona, text=prompt, max_tokens=max_tokens)
            total_cost = r.cumulative_usd
            revised.append({"role": role, "opinion": r.text.strip()})
        opinions = revised

    # Moderator synthesis -> task-formatted answer.
    panel = "\n\n".join(f"{o['role']}: {o['opinion']}" for o in opinions)
    mod_prompt = (f"{evidence}\n\nEXPERT PANEL:\n{panel}\n\nSynthesize the panel and the evidence "
                  f"into the final answer. {_task_instruction(case.task)}")
    mod = mllm.chat(
        system="You are the moderator of an expert diagnostic panel. Weigh image findings as "
               "support/refutation of candidates; do not let a single tool override strong consensus.",
        text=mod_prompt, max_tokens=max_tokens,
    )
    prediction = _parse_final(case.task, mod.text)

    return {
        "case_id": case.id,
        "task": case.task,
        "prediction": prediction,
        "expert_opinions": opinions,
        "moderator_raw": mod.text,
        "usage": {"cumulative_usd": mod.cumulative_usd},
    }
