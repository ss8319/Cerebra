"""Step 1 modality classification: DO NOT trust the dataset's `modality` label.

~40% of images are labeled `unknown` and even labeled ones can be wrong, so we
re-derive each image's modality with the multimodal base MLLM and route on THAT.
Cheap perception task -> run non-thinking regardless of the debate's reasoning mode.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from cerebra.dermarena.case import Case
from cerebra.dermarena.llm import OpenRouterMLLM

LABELS = [
    "clinical_photo", "dermoscopy", "histopathology", "ihc", "immunofluorescence",
    "gross_pathology", "radiology", "molecular", "schematic", "unknown",
]

SYSTEM = "You are an expert medical-imaging triager. Identify the imaging modality of a single medical image."

USER = (
    "Classify this medical image into EXACTLY one of these modalities:\n"
    "- clinical_photo: naked-eye photograph of skin/body surface\n"
    "- dermoscopy: magnified, polarized/contact-fluid view of a skin lesion\n"
    "- histopathology: H&E-stained tissue microscopy\n"
    "- ihc: immunohistochemistry (brown/DAB or chromogen stain)\n"
    "- immunofluorescence: fluorescent microscopy (dark field, glowing signal)\n"
    "- gross_pathology: macroscopic photo of an excised specimen/organ\n"
    "- radiology: X-ray, CT, MRI, or ultrasound\n"
    "- molecular: gels, blots, karyotype, or sequencing traces\n"
    "- schematic: diagram, chart, or illustration\n"
    "- unknown: none of the above / cannot tell\n\n"
    "Reply with ONLY the single modality label (e.g. `dermoscopy`)."
)


def _match_label(text: str) -> str:
    t = (text or "").strip().lower()
    for lab in LABELS:  # exact/substring match against the known vocabulary
        if lab in t:
            return lab
    # loose synonyms
    if re.search(r"\bderm[o ]?scop", t):
        return "dermoscopy"
    if "clinical" in t or "photograph" in t:
        return "clinical_photo"
    if "h&e" in t or "histolog" in t or "microscop" in t:
        return "histopathology"
    return "unknown"


def classify_case(case: Case, mllm: Optional[OpenRouterMLLM] = None) -> Dict[str, str]:
    """Classify every present image; return {abs_path: predicted_modality}.

    Uses a NON-thinking client (perception, not reasoning) so it stays cheap even when
    the pipeline runs the debate with reasoning on.
    """
    mllm = mllm or OpenRouterMLLM(thinking=False)
    out: Dict[str, str] = {}
    for im in case.resolved_images():
        if not im.get("exists"):
            continue
        r = mllm.chat(system=SYSTEM, text=USER, image_paths=[im["abs_path"]], max_tokens=16)
        out[im["abs_path"]] = _match_label(r.text)
    return out
