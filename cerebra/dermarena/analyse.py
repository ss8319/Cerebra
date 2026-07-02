"""ANALYSE step: turn a Case into a deterministic tool execution plan.

No LLM here — routing is driven by the supplied per-image `modality` label via the
modality router. Output is a plan the RUN step executes:

  - clinical_photo / dermoscopy  -> PanDermClassifierTool + DermoGPTVQATool
  - everything else + unknown     -> MedGemmaDermAnalyzerTool
"""
from __future__ import annotations

from typing import Any, Dict, List

from cerebra.dermarena.case import Case
from cerebra.tools.image_agent.modality_router import (
    DERM_SPECIALIST, GENERAL_VLM, route_case_images, summarize_routing,
)

# Which tools serve each route.
ROUTE_TOOLS = {
    DERM_SPECIALIST: ["PanDermClassifierTool", "DermoGPTVQATool"],
    GENERAL_VLM: ["MedGemmaDermAnalyzerTool"],
}


def build_plan(case: Case) -> Dict[str, Any]:
    """Build the deterministic tool plan for a case.

    Returns a dict:
      {
        case_id, task, has_images, n_images_present, n_images_missing,
        routing: <summarize_routing>,
        tool_calls: [ {tool, route, image_paths, modalities}, ... ],
      }
    Only images that exist on disk are scheduled; missing ones are counted, not planned.
    """
    resolved = case.resolved_images()
    present = [im for im in resolved if im.get("exists")]
    missing = [im for im in resolved if not im.get("exists")]

    grouped = route_case_images(present)  # keys: DERM_SPECIALIST, GENERAL_VLM

    tool_calls: List[Dict[str, Any]] = []
    for route, imgs in grouped.items():
        if not imgs:
            continue
        image_paths = [im["abs_path"] for im in imgs]
        modalities = [im.get("modality") for im in imgs]
        for tool in ROUTE_TOOLS[route]:
            tool_calls.append({
                "tool": tool,
                "route": route,
                "image_paths": image_paths,
                "modalities": modalities,
            })

    return {
        "case_id": case.id,
        "task": case.task,
        "has_images": case.has_images,
        "n_images_present": len(present),
        "n_images_missing": len(missing),
        "missing_paths": [im.get("abs_path") for im in missing],
        "routing": summarize_routing(present),
        "tool_calls": tool_calls,
    }
