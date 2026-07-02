# Foundation Models — Reproduction Summary

Three HF foundation models downloaded, verified, and validated on M3 (2026-07-02).

## Environment
- **Python:** `/fs04/scratch2/ub62/ssim0070/dermagent/bin/python` (torch 2.9.1+cu128, timm 1.0.27, transformers 4.57.6)
- **Extra dep installed:** `einops_exts` (required by CONCHv1.5 only)
- **HF token:** `HUGGINGFACE_TOKEN` in `/fs04/scratch2/ub62/ssim0070/.env` (account `shamussim`, has gated access to all three)
- **Run location:** login node = CPU (ok for encoders, ~15 min/case for MedGemma). Weights are local → submit GPU jobs with **no runtime download**.

## Weights (all under `/fs04/scratch2/ub62/ssim0070/models/`)
| Model | Path | Size | Key file |
|---|---|---|---|
| UNI2-h | `models/UNI2-h/` | 2.7 GB | `pytorch_model.bin` |
| CONCHv1.5 | `models/conchv1_5/` | 1.2 GB | `pytorch_model_vision.bin` |
| MedGemma 1.5 4b-it | `models/medgemma-1.5-4b-it/` | 8.1 GB | `model-0000{1,2}-of-00002.safetensors` |

Ready-to-run scripts: `models/run_uni2h.py`, `models/run_conchv1_5.py`, `models/run_medgemma.py`

## How to run
```bash
PY=/fs04/scratch2/ub62/ssim0070/dermagent/bin/python
cd /fs04/scratch2/ub62/ssim0070/models
$PY run_uni2h.py      <img.jpg>              # -> (1,1536) embedding
$PY run_conchv1_5.py  <img.jpg>              # -> (1,768)  embedding
$PY run_medgemma.py   <img.jpg> "question"   # -> generated text
```

## Documented params ("run properly")
- **UNI2-h** — ViT-g/14 via `timm`. MUST pass official `timm_kwargs` (SwiGLU/SiLU, `init_values=1e-5`, `reg_tokens=8`, `no_embed_class`, `dynamic_img_size`). Transform: 224px, `crop_pct=1.0`, bilinear, ImageNet mean/std. No extra deps.
- **CONCHv1.5** — ViT-L/16 @ **448px** + attentional pooler → 768-d. Standalone repo ships weights only; builder vendored from MahmoodLab **trident** into `models/conchv1_5/conchv1_5_model.py` (needs `einops_exts`). Transform: Resize(448)→CenterCrop(448)→ImageNet norm; fp16. Loader resizes pos-embed + `load_state_dict(strict=True)` — use `create_model_from_pretrained`, don't hand-roll.
- **MedGemma 1.5 4b-it** — `Gemma3ForConditionalGeneration` via `AutoProcessor` + `AutoModelForImageTextToText`, bf16. Build `{type:image}`/`{type:text}` content blocks → `apply_chat_template(add_generation_prompt=True)`. **Greedy decoding** (`do_sample=False`, `max_new_tokens=2000`) per HF docs (greedy is default since Jan 2026). Decode only tokens after `input_len`.

## Validation results
- **UNI2-h / CONCHv1.5** (real DermArena images): finite, no NaN; **determinism cos = 1.00000**; cross-image cos mean ~0.54–0.57 (range 0.21–0.73) → **discriminative, not collapsed**. ✓
- **MedGemma**: produced coherent, image-grounded dermatology descriptions + differentials (greedy). ✓
- ⚠️ **Caveat:** UNI2-h & CONCHv1.5 are **histopathology (WSI-tile)** encoders; DermArena images are clinical photos/dermoscopy → out-of-domain, expect weaker separability than on H&E tiles.
