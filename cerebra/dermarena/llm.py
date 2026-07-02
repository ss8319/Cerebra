"""Minimal OpenRouter client for the DermArena agent + a hard budget guard.

Self-contained (not via Cerebra's engine factory, which has no OpenRouter branch).
Handles multimodal messages (text + images as base64 data URLs), applies the
provider-documented Qwen3.5-27B sampling params, and enforces a USD spend cap.
"""
from __future__ import annotations

import base64
import io
import json
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PIL import Image

# --- config ------------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.environ.get("DERMARENA_BASE_MLLM", "qwen/qwen3.5-27b")
DEFAULT_ENV_FILE = "/fs04/scratch2/ub62/ssim0070/DermArena_data/.env"
LEDGER_PATH = os.environ.get(
    "DERMARENA_LEDGER", "cerebra_cache/dermarena/ledger.json"
)
BUDGET_USD = float(os.environ.get("DERMARENA_BUDGET_USD", "10.0"))

# Pricing for qwen/qwen3.5-27b (USD per token; from OpenRouter models API 2026-07-02).
PRICE_PROMPT = 0.000000195
PRICE_COMPLETION = 0.00000156

# Provider-documented sampling (sglang/slurm/sampling_params.py -> Qwen3.5-27B;
# source: HF model card Best Practices). See Cerebra/CLAUDE.md rule.
QWEN35_27B_NONTHINKING_GENERAL = {
    "temperature": 0.7,
    "top_p": 0.80,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}
QWEN35_27B_THINKING_GENERAL = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}
# OpenAI SDK native params vs. ones that must go through extra_body for OpenRouter.
_NATIVE_PARAMS = {"temperature", "top_p", "presence_penalty"}
_EXTRA_PARAMS = {"top_k", "min_p", "repetition_penalty"}


class BudgetExceeded(RuntimeError):
    pass


def _load_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    if os.path.isfile(DEFAULT_ENV_FILE):
        with open(DEFAULT_ENV_FILE) as f:
            for line in f:
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip()
    raise RuntimeError("OPENROUTER_API_KEY not found in env or DermArena_data/.env")


def _encode_image(path: str, max_side: int = 768) -> str:
    """Downscale (to cap image tokens/cost) and return a base64 JPEG data URL."""
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


class Ledger:
    """Thread-safe, file-backed cumulative spend tracker with a hard cap."""

    def __init__(self, path: str = LEDGER_PATH, budget_usd: float = BUDGET_USD):
        self.path = path
        self.budget_usd = budget_usd
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.isfile(path):
            self._write({"cost_usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "calls": 0})

    def _read(self) -> Dict[str, Any]:
        with open(self.path) as f:
            return json.load(f)

    def _write(self, d: Dict[str, Any]) -> None:
        with open(self.path, "w") as f:
            json.dump(d, f, indent=2)

    def spent(self) -> float:
        return self._read()["cost_usd"]

    def check(self) -> None:
        if self.spent() >= self.budget_usd:
            raise BudgetExceeded(
                f"DermArena budget cap hit: ${self.spent():.4f} >= ${self.budget_usd:.2f} "
                f"(ledger {self.path}). Raise DERMARENA_BUDGET_USD to continue."
            )

    def record(self, prompt_tokens: int, completion_tokens: int) -> Dict[str, Any]:
        cost = prompt_tokens * PRICE_PROMPT + completion_tokens * PRICE_COMPLETION
        with self._lock:
            d = self._read()
            d["cost_usd"] += cost
            d["prompt_tokens"] += prompt_tokens
            d["completion_tokens"] += completion_tokens
            d["calls"] += 1
            self._write(d)
        return {"call_cost_usd": cost, "cumulative_usd": d["cost_usd"]}


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    call_cost_usd: float
    cumulative_usd: float


class OpenRouterMLLM:
    """Thin multimodal chat client for OpenRouter, budget-guarded."""

    def __init__(self, model: str = DEFAULT_MODEL, ledger: Optional[Ledger] = None,
                 sampling: Optional[Dict[str, Any]] = None, thinking: bool = False,
                 default_max_tokens: Optional[int] = None):
        from openai import OpenAI
        self.model = model
        self.ledger = ledger or Ledger()
        # Qwen3.5-27B is a THINKING model. thinking=True -> reasoning ON with the HF-card
        # thinking params + a generous token budget (the earlier "empty content + 13x cost"
        # was token STARVATION at max_tokens=512, not reasoning itself). thinking=False ->
        # reasoning OFF (reasoning.enabled=False, verified 1175->32 tokens) + nonthinking params.
        self.thinking = thinking
        self.sampling = sampling or (QWEN35_27B_THINKING_GENERAL if thinking else QWEN35_27B_NONTHINKING_GENERAL)
        self.default_max_tokens = default_max_tokens or (3000 if thinking else 1024)
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=_load_api_key())

    def chat(self, system: str, text: str, image_paths: Optional[List[str]] = None,
             max_tokens: Optional[int] = None, max_images: int = 6) -> LLMResult:
        self.ledger.check()  # fail fast before spending
        max_tokens = max_tokens or self.default_max_tokens

        content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
        for p in (image_paths or [])[:max_images]:
            content.append({"type": "image_url", "image_url": {"url": _encode_image(p)}})

        native = {k: v for k, v in self.sampling.items() if k in _NATIVE_PARAMS}
        extra = {k: v for k, v in self.sampling.items() if k in _EXTRA_PARAMS}
        extra["reasoning"] = {"enabled": True} if self.thinking else {"enabled": False}

        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": content}]

        def _call(ex, mt):
            r = self.client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=mt, extra_body=ex, **native)
            u = r.usage
            return ((r.choices[0].message.content or ""),
                    getattr(u, "prompt_tokens", 0) or 0,
                    getattr(u, "completion_tokens", 0) or 0)

        text, pt, ct = _call(extra, max_tokens)
        # Thinking can spend the whole budget on reasoning and return EMPTY content.
        # Retry once with reasoning OFF so we always get a usable answer (and stop paying
        # for wasted thinking tokens on that call).
        if self.thinking and not text.strip():
            extra_off = dict(extra)
            extra_off["reasoning"] = {"enabled": False}
            t2, pt2, ct2 = _call(extra_off, max_tokens)
            text, pt, ct = t2, pt + pt2, ct + ct2

        rec = self.ledger.record(pt, ct)
        return LLMResult(
            text=text, prompt_tokens=pt, completion_tokens=ct,
            call_cost_usd=rec["call_cost_usd"], cumulative_usd=rec["cumulative_usd"],
        )
