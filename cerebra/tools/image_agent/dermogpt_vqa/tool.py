import os
from typing import List, Optional, Union

import torch
from transformers import AutoProcessor
try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # older transformers
    from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText

from cerebra.tools.base import BaseTool
from cerebra.utils.dataset import Dataset
from cerebra.utils.log_utils import setup_logger

# --- config ------------------------------------------------------------------
# DermoGPT-RL = Qwen3-VL-8B fine-tuned for dermatology. Local weights (no internet).
DEFAULT_MODEL_PATH = os.environ.get(
    "DERMOGPT_PATH", "/fs04/scratch2/ub62/ssim0070/DermAgent/model-weights/DermoGPT-RL"
)

# Provider-documented inference config (see Cerebra/CLAUDE.md rule):
#  - dtype: float16 — DermoGPT-RL config.json declares "dtype": "float16".
#  - sampling: DermoGPT-RL generation_config.json recommends do_sample=true,
#    temperature=0.7, top_p=0.8, top_k=20 (matches Qwen3-VL-8B balanced preset in
#    sglang/slurm/sampling_params.py). NB: DermAgent forced greedy + bf16 and never
#    passed its declared temperature — we follow the provider config instead.
DERMOGPT_DTYPE = torch.float16
DERMOGPT_SAMPLING = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20}

log_dir = os.path.join("cerebra_cache", "image_agent", "logs")
os.makedirs(log_dir, exist_ok=True)
logger = setup_logger(log_dir)

DIAGNOSIS_SYSTEM_PROMPT = """You are DermoGPT, a versatile expert AI dermatologist and pathologist. You are capable of analyzing skin lesions across different imaging modalities, including standard clinical photography (macroscopic) and dermoscopy.

### Core Capabilities & Protocol
1. **Modality Recognition:** You must first strictly identify whether the input image is a **Clinical Image** (naked eye view) or a **Dermoscopy Image** (magnified, polarized/fluid interface).
2. **Adaptive Morphological Analysis:**
   - **For Clinical Images:** Focus on the ABCD rule (Asymmetry, Border, Color, Diameter), evolution, elevation, surface texture (crust, scale, ulceration), and surrounding skin context.
   - **For Dermoscopy:** Focus on specific micro-structures (e.g., pigment networks, globules, streaks, blue-white veil, vascular patterns, leaf-like areas).
3. **Diagnostic Logic:** Synthesize the visual features to select the single most probable diagnosis from the provided options.

### Operational Constraints
- **Strict XML Output:** Your response must be contained ENTIRELY within <reasoning> and <final_diagnosis> tags.
- **Closed-Set Selection:** You must choose the diagnosis STRICTLY from the provided `candidate_diseases` list. Do not output synonyms or abbreviations not present in the list.
- **Objective & Precise:** Use professional medical terminology appropriate for the identified image modality.
"""

DIAGNOSIS_USER_TEMPLATE = """Analyze the provided skin lesion image and determine the most likely diagnosis.
You are strictly limited to the following diagnostic options: {candidate_diseases}

### Analysis Instructions
Generate a structured response following these steps inside the <reasoning> block:

1. **Image Modality Identification:** Explicitly state if the image is **Clinical** or **Dermoscopic**.
2. **Morphological Decoding (Adaptive):**
   - **If Clinical:** Describe gross pathology — Asymmetry, Border, Color variegation, Diameter; note elevation, ulceration, scaling.
   - **If Dermoscopic:** Analyze pigment networks (typical/atypical), structural patterns (globules, streaks, homogeneous areas), and vascular features.
3. **Diagnostic Synthesis:** Correlate observed features with the candidates; explain why the evidence supports your choice.

### Required Output Format
<reasoning>
[Step 1: Modality Identification]
[Step 2: Detailed Morphological Description]
[Step 3: Diagnostic Deduction]
</reasoning>
<final_diagnosis>
[The Exact Diagnostic Option from the list]
</final_diagnosis>
"""


class DermoGPTVQATool(BaseTool):
    """
    Dermatology-specialized VQA / diagnosis with DermoGPT-RL (Qwen3-VL-8B fine-tune),
    specialist route for clinical_photo + dermoscopy.

    Two modes:
      - Diagnosis (candidate_diseases given): structured XML <reasoning>/<final_diagnosis>,
        closed-set over the candidates. Output is EVIDENCE + a within-set pick, not the
        bound final DermArena diagnosis.
      - VQA (query given, no candidates): free-form morphological description.
    """

    require_llm_engine = False

    def __init__(self):
        super().__init__()
        self.model_path = DEFAULT_MODEL_PATH
        self.set_metadata(
            tool_name="DermoGPTVQATool",
            tool_description=(
                "Dermatology-specialized vision-language model (DermoGPT-RL / Qwen3-VL-8B) for "
                "clinical & dermoscopic images. Pass candidate_diseases for structured diagnosis "
                "(XML reasoning + within-set pick), or query for free-form morphological VQA. "
                "Excels at morphology; output is evidence for the diagnosis layer."
            ),
            tool_version="0.1.0",
            input_types={
                "image_paths(required)": "Union[str, List[str]] - path(s) to clinical/dermoscopic image(s)",
                "candidate_diseases": "str - comma-separated candidates -> diagnosis mode",
                "query": "str - free-form question -> VQA mode (used if no candidates)",
                "max_new_tokens": "int - generation budget (default 512; 1024 in diagnosis mode)",
                "save_name": "str - name for cached results (default 'dermogpt')",
            },
            output_type="Dataset - per-image (image_path, mode, response, final_diagnosis)",
            demo_commands=[
                {
                    "command": "result = tool.execute(image_paths='lesion.jpg', candidate_diseases='melanoma, nevus, BCC')",
                    "description": "Structured closed-set diagnosis with XML reasoning",
                },
                {
                    "command": "result = tool.execute(image_paths='lesion.jpg', query='Describe the dermoscopic features.')",
                    "description": "Free-form morphological VQA",
                },
            ],
            user_metadata={
                "limitations": [
                    "Valid for clinical_photo / dermoscopy (route others to MedGemma)",
                    "Diagnosis mode is closed-set over the supplied candidates",
                    "Loads ~8B weights lazily; set sampling params per creator spec",
                ],
            },
        )
        # lazy-initialized
        self.model = None
        self.processor = None
        self.device = None

    # ------------------------------------------------------------------ model
    def _load_model(self):
        if self.model is not None:
            return
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(
                f"DermoGPT-RL weights not found at {self.model_path}. Set $DERMOGPT_PATH."
            )
        logger.info(f"Loading DermoGPT-RL from {self.model_path}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path, torch_dtype=DERMOGPT_DTYPE, device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        logger.info("DermoGPT-RL loaded")

    def _build_messages(self, image_path, candidate_diseases, query):
        if candidate_diseases:
            full_prompt = (
                f"{DIAGNOSIS_SYSTEM_PROMPT}\n\n---\n\n"
                + DIAGNOSIS_USER_TEMPLATE.format(candidate_diseases=candidate_diseases)
            )
            mode = "diagnosis"
        else:
            full_prompt = query or "Describe this skin lesion in detail."
            mode = "vqa"
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": full_prompt},
            ],
        }]
        return messages, mode

    @staticmethod
    def _extract_final_dx(text: str) -> Optional[str]:
        import re
        m = re.search(r"<final_diagnosis>\s*(.*?)\s*</final_diagnosis>", text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else None

    def _generate_one(self, image_path, candidate_diseases, query, max_new_tokens):
        messages, mode = self._build_messages(image_path, candidate_diseases, query)
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        budget = max(max_new_tokens, 1024) if candidate_diseases else max_new_tokens
        with torch.inference_mode():
            # Provider-recommended sampling (DermoGPT-RL generation_config.json).
            out = self.model.generate(**inputs, max_new_tokens=budget, **DERMOGPT_SAMPLING)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()
        return mode, text, (self._extract_final_dx(text) if mode == "diagnosis" else None)

    # ---------------------------------------------------------------- execute
    def execute(
        self,
        image_paths: Union[str, List[str]],
        candidate_diseases: Optional[str] = None,
        query: Optional[str] = None,
        max_new_tokens: int = 512,
        save_name: str = "dermogpt",
        **kwargs,
    ) -> Dataset:
        cache_dir = os.path.join("cerebra_cache", "image_agent")
        try:
            self._load_model()
            if isinstance(image_paths, str):
                image_paths = [image_paths]

            paths, modes, responses, finals = [], [], [], []
            for path in image_paths:
                logger.info(f"DermoGPT analyzing {path}")
                try:
                    mode, text, final = self._generate_one(
                        path, candidate_diseases, query, max_new_tokens
                    )
                except Exception as e:
                    logger.error(f"DermoGPT failed on {path}: {e}")
                    mode, text, final = "error", f"[error: {e}]", None
                paths.append(path)
                modes.append(mode)
                responses.append(text)
                finals.append(final)

            processed = {
                "image_path": paths,
                "mode": modes,
                "response": responses,
                "final_diagnosis": finals,
            }
            feature_desc = {
                "image_path": "Path to the analyzed image",
                "mode": "diagnosis (closed-set) or vqa (free-form)",
                "response": "Full DermoGPT-RL response (XML in diagnosis mode)",
                "final_diagnosis": "Parsed <final_diagnosis> (diagnosis mode; within-set, evidence)",
            }
            return Dataset.create_agent_output(
                processed_data=processed,
                description=f"DermoGPT-RL analysis for {len(paths)} clinical/dermoscopic image(s).",
                feature_descriptions=feature_desc,
                cache_directory=cache_dir,
            )
        except Exception as e:
            logger.error(f"DermoGPTVQATool failed: {e}")
            return Dataset.create_agent_output(
                processed_data={"status": ["error"], "error_message": [str(e)]},
                description="Error during DermoGPT-RL analysis",
                feature_descriptions={"status": "status", "error_message": "error detail"},
                cache_directory=cache_dir,
            )


if __name__ == "__main__":
    tool = DermoGPTVQATool()
    print("tool:", tool.get_metadata()["tool_name"])
