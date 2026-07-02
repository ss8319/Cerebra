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


def build_plan(case: Case, modality_by_path: Dict[str, str] = None) -> Dict[str, Any]:
    """Build the tool plan for a case.

    Args:
        modality_by_path: {abs_path: modality} from Step-1 classification. When given,
            it OVERRIDES the dataset's `modality` label (which we don't trust) for routing.

    Returns a dict:
      {
        case_id, task, has_images, n_images_present, n_images_missing,
        routing: <summarize_routing>, used_classified_modality: bool,
        tool_calls: [ {tool, route, image_paths, modalities}, ... ],
      }
    Only images that exist on disk are scheduled; missing ones are counted, not planned.
    """
    modality_by_path = modality_by_path or {}
    resolved = case.resolved_images()
    # Apply classified modality over the (untrusted) supplied label.
    for im in resolved:
        if im.get("abs_path") in modality_by_path:
            im["provided_modality"] = im.get("modality")
            im["modality"] = modality_by_path[im["abs_path"]]
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
        "used_classified_modality": bool(modality_by_path),
        "tool_calls": tool_calls,
    }
