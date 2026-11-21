from __future__ import annotations

import json
import random
import re

import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def norm_text(s) -> str:
    """Collapse whitespace; non-strings (NaN, None) become ''."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s).strip()


def extract_json(text: str):
    """Pull the first JSON object/array out of an LLM reply (handles ``` fences and chatter)."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text).strip()
    starts = sorted((text.find(c), c, e) for c, e in (("{", "}"), ("[", "]")) if text.find(c) != -1)
    for start, _, close in starts:
        end = text.rfind(close)
        if end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def progress(it, **kw):
    """tqdm if installed, otherwise a no-op iterator."""
    try:
        from tqdm.auto import tqdm

        return tqdm(it, **kw)
    except ImportError:
        return it
