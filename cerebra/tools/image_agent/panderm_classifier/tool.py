import os
import sys
import importlib
from pathlib import Path
from typing import List, Optional, Union, Any

import numpy as np
import torch
from PIL import Image

from cerebra.tools.base import BaseTool
from cerebra.utils.dataset import Dataset
from cerebra.utils.log_utils import setup_logger

# --- config ------------------------------------------------------------------
# DermLIP/PanDerm is loaded through Derm1M's *custom* open_clip (BEiT/timm arch).
# That source lives in the DermAgent repo; override with $DERM1M_SRC.
DERM1M_SRC = os.environ.get(
    "DERM1M_SRC", "/fs04/scratch2/ub62/ssim0070/DermAgent/Derm1M/src"
)
# HF hub id; resolved from the local HF cache under HF_HUB_OFFLINE=1
# (weights: .hf_cache/hub/models--redlessone--DermLIP_PanDerm-base-w-PubMed-256).
MODEL_ID = os.environ.get("PANDERM_MODEL_ID", "redlessone/DermLIP_PanDerm-base-w-PubMed-256")

log_dir = os.path.join("cerebra_cache", "image_agent", "logs")
os.makedirs(log_dir, exist_ok=True)
logger = setup_logger(log_dir)

# Broad default differential for the clinical_photo / dermoscopy specialist route.
DEFAULT_DISEASES = [
    "melanoma", "basal cell carcinoma", "squamous cell carcinoma", "nevus",
    "seborrheic keratosis", "actinic keratosis", "dermatofibroma",
    "vascular lesion", "pigmented benign keratosis", "atypical nevus",
]

# Prompt ensemble. First entry is the DermLIP/PanDerm provider's canonical template
# (README "Quick Start": `f'This is a skin image of {c}'`); the modality-specific
# variants are averaged in as a standard zero-shot ensemble. Image preprocessing
# (224px bicubic, CLIP norm) comes from the model's own `preprocess` transform.
TEMPLATES = [
    "This is a skin image of {}",
    "a dermoscopy image of {}",
    "a clinical photo of {}",
    "a skin lesion of {}",
]


class PanDermClassifierTool(BaseTool):
    """
    Zero-shot dermatology image classifier (DermLIP/PanDerm), specialist route for
    clinical_photo + dermoscopy.

    IMPORTANT — closed-set: PanDerm scores an image against a FIXED candidate label
    set via CLIP text prompts. DermArena has 2000+ diagnoses; the true answer is
    usually NOT in any short candidate list, so this tool's output is a ranked
    *visual differential* used as EVIDENCE, never the bound final diagnosis. A good
    pattern is for the calling agent to pass `candidate_diseases` from the text
    reasoner's differential, so PanDerm re-ranks those hypotheses visually.
    """

    require_llm_engine = False

    def __init__(self):
        super().__init__()
        self.model_id = MODEL_ID
        self.derm1m_src = DERM1M_SRC
        self.set_metadata(
            tool_name="PanDermClassifierTool",
            tool_description=(
                "Zero-shot classify a clinical or dermoscopic skin image against a set of "
                "candidate diseases using DermLIP/PanDerm (CLIP-style). Returns a ranked "
                "differential with confidence scores. CLOSED-SET: output is visual evidence, "
                "not a final diagnosis. Best used with candidate_diseases from a text hypothesis."
            ),
            tool_version="0.1.0",
            input_types={
                "image_paths(required)": "Union[str, List[str]] - path(s) to clinical/dermoscopic image(s)",
                "candidate_diseases": "List[str] or comma-sep str - candidate labels to rank (default: skin-cancer set)",
                "top_k": "int - number of ranked results to return per image (default 5)",
                "save_name": "str - name for cached results (default 'panderm')",
            },
            output_type="Dataset - per-image ranked differential (image_path, ranked_diseases, scores)",
            demo_commands=[
                {
                    "command": "result = tool.execute(image_paths='lesion.jpg', candidate_diseases='melanoma, nevus, basal cell carcinoma')",
                    "description": "Rank the given candidates for one clinical/dermoscopic image",
                },
            ],
            user_metadata={
                "limitations": [
                    "CLOSED-SET: only ranks the supplied/default candidates — not a diagnosis",
                    "Valid only for clinical_photo / dermoscopy (route others to MedGemma)",
                    "Requires Derm1M's custom open_clip on $DERM1M_SRC + local HF weights",
                ],
            },
        )
        # lazy-initialized
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.device = None
        self._open_clip = None

    # ------------------------------------------------------------------ model
    def _load_model(self):
        if self.model is not None:
            return
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading DermLIP/PanDerm ({self.model_id}) via {self.derm1m_src}")

        # Prioritize Derm1M's open_clip; purge any already-imported open_clip.
        original_path = sys.path.copy()
        sys.path = [p for p in sys.path if "MAKE" not in p]
        if self.derm1m_src not in sys.path:
            sys.path.insert(0, self.derm1m_src)
        for mod in [k for k in list(sys.modules) if k.startswith("open_clip")]:
            del sys.modules[mod]
        try:
            import open_clip as derm1m_open_clip
            self._open_clip = derm1m_open_clip
            self.model, self.preprocess = derm1m_open_clip.create_model_from_pretrained(
                f"hf-hub:{self.model_id}", device=self.device
            )
            self.tokenizer = derm1m_open_clip.get_tokenizer(f"hf-hub:{self.model_id}")
            self.model.eval()
        finally:
            sys.path = original_path
        logger.info("DermLIP/PanDerm loaded")

    def _build_text_features(self, class_names: List[str]) -> torch.Tensor:
        feats = []
        with torch.no_grad():
            for name in class_names:
                tokens = self.tokenizer([t.format(name) for t in TEMPLATES]).to(self.device)
                f = self.model.encode_text(tokens)
                f = f.mean(dim=0)
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f)
        return torch.stack(feats, dim=0)

    def _classify_one(self, image_path: str, diseases: List[str], top_k: int):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        text_features = self._build_text_features(diseases)
        with torch.no_grad():
            img_features = self.model.encode_image(image_tensor)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            probs = (100.0 * img_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]
        order = np.argsort(probs)[::-1][:top_k]
        ranked = [diseases[i] for i in order]
        scores = [float(probs[i]) for i in order]
        return ranked, scores

    # ---------------------------------------------------------------- execute
    def execute(
        self,
        image_paths: Union[str, List[str]],
        candidate_diseases: Optional[Union[str, List[str]]] = None,
        top_k: int = 5,
        save_name: str = "panderm",
        **kwargs,
    ) -> Dataset:
        cache_dir = os.path.join("cerebra_cache", "image_agent")
        try:
            self._load_model()
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            if candidate_diseases is None:
                diseases = DEFAULT_DISEASES
            elif isinstance(candidate_diseases, str):
                diseases = [d.strip() for d in candidate_diseases.split(",") if d.strip()]
            else:
                diseases = list(candidate_diseases)

            paths, ranked_all, scores_all = [], [], []
            for path in image_paths:
                logger.info(f"PanDerm classifying {path}")
                try:
                    ranked, scores = self._classify_one(path, diseases, top_k)
                except Exception as e:
                    logger.error(f"PanDerm failed on {path}: {e}")
                    ranked, scores = [f"[error: {e}]"], [0.0]
                paths.append(path)
                ranked_all.append(ranked)
                scores_all.append(scores)

            processed = {
                "image_path": paths,
                "ranked_diseases": ranked_all,
                "scores": scores_all,
                "candidate_set": [diseases] * len(paths),
            }
            feature_desc = {
                "image_path": "Path to the classified image",
                "ranked_diseases": "Top-k candidate diseases ranked by CLIP similarity (closed-set evidence)",
                "scores": "Softmax confidence per ranked disease",
                "candidate_set": "The closed candidate label set scored for this image",
            }
            return Dataset.create_agent_output(
                processed_data=processed,
                description=(
                    f"PanDerm zero-shot differential for {len(paths)} image(s) over "
                    f"{len(diseases)} candidates. Closed-set evidence, not a final dx."
                ),
                feature_descriptions=feature_desc,
                cache_directory=cache_dir,
            )
        except Exception as e:
            logger.error(f"PanDermClassifierTool failed: {e}")
            return Dataset.create_agent_output(
                processed_data={"status": ["error"], "error_message": [str(e)]},
                description="Error during PanDerm classification",
                feature_descriptions={"status": "status", "error_message": "error detail"},
                cache_directory=cache_dir,
            )


if __name__ == "__main__":
    tool = PanDermClassifierTool()
    print("tool:", tool.get_metadata()["tool_name"])
    # ds = tool.execute(image_paths="/path/to/lesion.jpg", candidate_diseases="melanoma, nevus")
    # print(ds.get_dataset()["dataset"])
