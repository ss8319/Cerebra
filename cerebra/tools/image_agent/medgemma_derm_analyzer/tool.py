import os
from typing import List, Optional, Dict, Union, Any

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from cerebra.tools.base import BaseTool
from cerebra.utils.dataset import Dataset
from cerebra.utils.log_utils import setup_logger

# --- config ------------------------------------------------------------------
# Compute nodes have NO internet (see project_m3_apptainer_gotchas): the model MUST
# be loaded from a local path, never a HF hub id. Override with $MEDGEMMA_PATH.
DEFAULT_MODEL_PATH = os.environ.get(
    "MEDGEMMA_PATH", "/fs04/scratch2/ub62/ssim0070/models/medgemma-1.5-4b-it"
)

log_dir = os.path.join("cerebra_cache", "image_agent", "logs")
os.makedirs(log_dir, exist_ok=True)
logger = setup_logger(log_dir)

DERM_SYSTEM_PROMPT = (
    "You are an expert dermatologist and dermatopathologist analyzing medical images "
    "from a case report. The image may be a clinical photograph, dermoscopy, "
    "histopathology, immunohistochemistry, radiology, or other modality. "
    "Describe ONLY what is visible in the image: lesion morphology, distribution, "
    "colour, borders, and any modality-specific features (e.g. dermoscopic pigment "
    "networks; histologic architecture and cellular detail). Then give a short ranked "
    "list of differential diagnoses these findings support. Be precise and use correct "
    "terminology. Do NOT invent clinical history that is not shown. If the image is "
    "uninformative, say so."
)


class MedGemmaDermAnalyzerTool(BaseTool):
    """
    General medical-image analyzer for DermArena, backed by MedGemma-1.5-4b-it (local weights).

    This is the GENERAL_VLM route of the modality router: it handles every image
    that is NOT clinical_photo / dermoscopy (histopathology, radiology, IHC,
    gross_pathology, immunofluorescence, molecular, schematic) plus the large
    `unknown` bucket. Output is per-image VISUAL EVIDENCE (morphology description +
    a ranked differential), consumed downstream by the summary/orchestrator layer —
    it is not itself the bound final diagnosis.
    """

    # NOTE: require_llm_engine stays False. We load MedGemma locally via transformers,
    # NOT through Cerebra's LLM engine factory. The Initializer instantiates tool
    # classes with NO args during discovery, so __init__ must not require model_string
    # and must not load weights (that happens lazily in execute()).
    require_llm_engine = False

    def __init__(self):
        super().__init__()

        self.model_path = DEFAULT_MODEL_PATH
        self.analysis_output_dir = os.path.join("cerebra_cache", "image_agent", "analyses")
        os.makedirs(self.analysis_output_dir, exist_ok=True)

        self.set_metadata(
            tool_name="MedGemmaDermAnalyzerTool",
            tool_description=(
                "Analyze non-dermoscopic / non-clinical-photo dermatology case images "
                "(histopathology, radiology, IHC, gross pathology, immunofluorescence, "
                "molecular, schematic, or unknown-modality) with MedGemma-1.5-4b-it. "
                "Returns per-image visual findings and a ranked differential to use as "
                "evidence for diagnosis."
            ),
            tool_version="0.1.0",
            input_types={
                "image_paths(required)": "Union[str, List[str]] - path(s) to case image(s)",
                "query": "str - specific question to ask about the image(s); optional",
                "case_context": "str - short masked clinical narrative for grounding; optional",
                "modality": "str - supplied modality label(s) for logging; optional",
                "max_new_tokens": "int - generation budget per image (default 512)",
                "save_name": "str - name for cached results (default 'medgemma_derm')",
            },
            output_type="Dataset - per-image findings (image_path, modality, query, findings)",
            demo_commands=[
                {
                    "command": "result = tool.execute(image_paths=['images/figures/PMC123/1.jpg'], modality='histopathology')",
                    "description": "Analyze one histopathology figure and return visual findings + differential",
                },
                {
                    "command": "result = tool.execute(image_paths=['a.jpg','b.jpg'], case_context='34yo man, weight loss, skin nodules', query='What do these figures show?')",
                    "description": "Analyze multiple figures grounded in the masked narrative",
                },
            ],
            user_metadata={
                "limitations": [
                    "Requires GPU; loads ~4B weights lazily on first execute()",
                    "Findings are visual evidence, NOT a bound final diagnosis",
                    "Weights loaded from a LOCAL path (no compute-node internet)",
                ],
                "best_practices": [
                    "Route clinical_photo / dermoscopy to the derm specialist tools instead",
                    "Pass case_context so the description is grounded, not generic",
                    "Set sampling params per MedGemma creator spec via the registry",
                ],
            },
        )

        # lazy-initialized
        self.model = None
        self.processor = None
        self.device = None

    # ------------------------------------------------------------------ model
    def _initialize_model(self):
        if self.model is not None:
            return
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(
                f"MedGemma weights not found at {self.model_path}. "
                f"Set $MEDGEMMA_PATH to the local model directory."
            )
        logger.info(f"Loading MedGemma from {self.model_path} ...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        logger.info(f"MedGemma loaded on {self.device}")

    def _load_image(self, image_path: str) -> Image.Image:
        img = Image.open(image_path)
        return img.convert("RGB") if img.mode != "RGB" else img

    def _analyze_one(self, image: Image.Image, query: str, case_context: Optional[str]) -> str:
        user_text = query
        if case_context:
            user_text = f"Masked clinical context: {case_context}\n\n{query}"
        messages = [
            {"role": "system", "content": [{"type": "text", "text": DERM_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image", "image": image},
                ],
            },
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            # Provider-documented: MedGemma README + generation_config.json (no sampling
            # params) prescribe GREEDY decoding (do_sample=False). Do not add temperature.
            gen = self.model.generate(**inputs, max_new_tokens=self._max_new_tokens, do_sample=False)
        gen = gen[0][input_len:]
        return self.processor.decode(gen, skip_special_tokens=True).strip()

    # ---------------------------------------------------------------- execute
    def execute(
        self,
        image_paths: Union[str, List[str]],
        query: str = "Describe the visual findings and give a ranked differential diagnosis.",
        case_context: Optional[str] = None,
        modality: Optional[Union[str, List[str]]] = None,
        max_new_tokens: int = 512,
        save_name: str = "medgemma_derm",
        **kwargs,
    ) -> Dataset:
        self._max_new_tokens = max_new_tokens
        cache_dir = os.path.join("cerebra_cache", "image_agent")
        try:
            self._initialize_model()

            if isinstance(image_paths, str):
                image_paths = [image_paths]
            if isinstance(modality, str) or modality is None:
                modality = [modality] * len(image_paths)

            paths, mods, findings = [], [], []
            for path, mod in zip(image_paths, modality):
                logger.info(f"MedGemma analyzing {path} (modality={mod})")
                try:
                    text = self._analyze_one(self._load_image(path), query, case_context)
                except Exception as e:  # per-image failure shouldn't kill the case
                    logger.error(f"Failed on {path}: {e}")
                    text = f"[error analyzing image: {e}]"
                paths.append(path)
                mods.append(mod)
                findings.append(text)

            processed = {
                "image_path": paths,
                "modality": mods,
                "query": [query] * len(paths),
                "findings": findings,
            }
            feature_desc = {
                "image_path": "Path to the analyzed image",
                "modality": "Supplied modality label for the image",
                "query": "Question posed to MedGemma",
                "findings": "MedGemma's visual findings + ranked differential (evidence, not final dx)",
            }
            return Dataset.create_agent_output(
                processed_data=processed,
                description=(
                    f"MedGemma-1.5 visual findings for {len(paths)} DermArena image(s). "
                    f"Evidence for downstream diagnosis; not a bound answer."
                ),
                feature_descriptions=feature_desc,
                cache_directory=cache_dir,
            )
        except Exception as e:
            logger.error(f"MedGemmaDermAnalyzerTool failed: {e}")
            return Dataset.create_agent_output(
                processed_data={"status": ["error"], "error_message": [str(e)]},
                description="Error during MedGemma derm analysis",
                feature_descriptions={
                    "status": "execution status",
                    "error_message": "error detail",
                },
                cache_directory=cache_dir,
            )


if __name__ == "__main__":
    tool = MedGemmaDermAnalyzerTool()
    print("tool metadata:", tool.get_metadata()["tool_name"])
    # Smoke test (needs GPU + a real image):
    # ds = tool.execute(image_paths=["/fs04/scratch2/ub62/ssim0070/dermarena_dx_v2/images/figures/PMC3459815/1752-1947-6-305-1.jpg"],
    #                   modality="clinical_photo", case_context="34yo man, weight loss, pneumonia")
    # print(ds.get_dataset()["dataset"]["findings"][0])
