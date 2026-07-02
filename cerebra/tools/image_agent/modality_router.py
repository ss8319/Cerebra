# cerebra/tools/image_agent/modality_router.py
"""
Modality router for the DermArena image agent.

DermArena supplies a per-image `modality` label. This module maps each image to
the tool best suited to it:

  - clinical_photo, dermoscopy  -> DERM_SPECIALIST  (PanDerm + DermoGPT-RL)
  - everything else + unknown   -> GENERAL_VLM      (MedGemma-1.5-4b-it)

Design notes
------------
* MedGemma is the WORKHORSE, not a fallback: clinical_photo + dermoscopy are only
  ~28% of images, while `unknown` alone is ~40%. So GENERAL_VLM sees the majority.
* The `modality` field is NOT trusted blindly for the specialist path. `unknown`
  (a first-class VLM label, not missing data) routes to GENERAL_VLM by policy,
  because sending a mislabeled histopath/radiology image into PanDerm (a closed-set
  clinical/dermoscopic classifier) is worse than useless.
* PanDerm is closed-set: its output is EVIDENCE (morphology + a ranked differential
  within its vocabulary), never the bound final diagnosis.
"""
from typing import Dict, List, Any

# Route identifiers
DERM_SPECIALIST = "derm_specialist"   # PanDerm + DermoGPT-RL
GENERAL_VLM = "general_vlm"           # MedGemma-1.5-4b-it

# Only these modalities are inside PanDerm/DermoGPT-RL's competence.
DERM_SPECIALIST_MODALITIES = {"clinical_photo", "dermoscopy"}

# Full modality vocabulary observed in dermarena_dx_v2 (verbatim `modality` values).
KNOWN_MODALITIES = {
    "unknown", "clinical_photo", "radiology", "histopathology", "dermoscopy",
    "ihc", "gross_pathology", "immunofluorescence", "schematic", "molecular",
}


def route_modality(modality: Any) -> str:
    """Return the target route (DERM_SPECIALIST | GENERAL_VLM) for one modality label.

    Anything not explicitly in the specialist set — including None, ``unknown``,
    and unseen labels — routes to the general VLM. Fail-safe by design.
    """
    if isinstance(modality, str) and modality.strip().lower() in DERM_SPECIALIST_MODALITIES:
        return DERM_SPECIALIST
    return GENERAL_VLM


def route_case_images(images: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group one case's `images[]` entries by target route.

    Args:
        images: list of DermArena image dicts, each with at least
                ``image_path`` and ``modality``.

    Returns:
        {DERM_SPECIALIST: [img, ...], GENERAL_VLM: [img, ...]}
        Each img dict is passed through untouched (keeps figure_id, panel_label, etc.),
        with an added ``_route`` key for downstream traceability.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {DERM_SPECIALIST: [], GENERAL_VLM: []}
    for img in images or []:
        route = route_modality(img.get("modality"))
        img = {**img, "_route": route}
        grouped[route].append(img)
    return grouped


def summarize_routing(images: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Small diagnostic summary (counts per route + per modality) for logging/traces."""
    grouped = route_case_images(images)
    per_modality: Dict[str, int] = {}
    for img in images or []:
        m = img.get("modality") or "None"
        per_modality[m] = per_modality.get(m, 0) + 1
    return {
        "n_images": len(images or []),
        "n_derm_specialist": len(grouped[DERM_SPECIALIST]),
        "n_general_vlm": len(grouped[GENERAL_VLM]),
        "per_modality": per_modality,
    }


if __name__ == "__main__":
    demo = [
        {"image_path": "a.jpg", "modality": "clinical_photo"},
        {"image_path": "b.jpg", "modality": "dermoscopy"},
        {"image_path": "c.jpg", "modality": "histopathology"},
        {"image_path": "d.jpg", "modality": "unknown"},
        {"image_path": "e.jpg", "modality": None},
    ]
    import json
    print(json.dumps(summarize_routing(demo), indent=2))
    for r, imgs in route_case_images(demo).items():
        print(r, [i["image_path"] for i in imgs])
