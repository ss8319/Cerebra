"""Robust JSON extraction from LLM responses (they often wrap JSON in prose/fences)."""
from __future__ import annotations

import json
import re
from typing import Any, Optional


def extract_json(text: str) -> Optional[Any]:
    """Return the first valid JSON object/array found in text, else None."""
    if not text:
        return None
    # strip ```json fences
    t = re.sub(r"```(?:json)?", "", text)
    # try the whole thing first, then the first {...} or [...] block
    for candidate in (t.strip(), _first_block(t, "{", "}"), _first_block(t, "[", "]")):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _first_block(text: str, open_c: str, close_c: str) -> Optional[str]:
    start = text.find(open_c)
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == open_c:
            depth += 1
        elif text[i] == close_c:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None
