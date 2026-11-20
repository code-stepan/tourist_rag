from __future__ import annotations

import re
from collections.abc import Callable

import pandas as pd

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[А-ЯЁA-Z0-9«\"(])")


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENT_SPLIT.split((text or "").strip()) if s]


def _hard_split(sentence: str, count_tokens: Callable[[str], int], max_tokens: int) -> list[str]:
    words, parts, cur = sentence.split(), [], []
    for w in words:
        if cur and count_tokens(" ".join(cur + [w])) > max_tokens:
            parts.append(" ".join(cur))
            cur = []
        cur.append(w)
    if cur:
        parts.append(" ".join(cur))
    return parts


def chunk_text(
    text: str,
    count_tokens: Callable[[str], int],
    max_tokens: int = 200,
    overlap_sentences: int = 1
) -> list[str]:
    sents: list[str] = []
    for s in split_sentences(text):
        sents.extend(
            _hard_split(s, count_tokens, max_tokens)
            if count_tokens(s) > max_tokens
            else [s])
    chunks, cur = [], []
    for s in sents:
        if cur and count_tokens(" ".join(cur + [s])) > max_tokens:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sentences:] if overlap_sentences else []
            if cur and count_tokens(" ".join(cur + [s])) > max_tokens:
                cur = []
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def make_chunks(
    places: pd.DataFrame,
    count_tokens: Callable[[str], int] | None = None,
    max_tokens: int = 200,
    overlap_sentences: int = 1,
    include_captions: bool = True
) -> list[dict]:
    count_tokens = count_tokens or (lambda s: len(s.split()))
    chunks: list[dict] = []
    for r in places.itertuples(index=False):
        header = r.name + (f" ({r.city})" if r.city else "")
        base = {"place_id": r.place_id, "name": r.name, "city": r.city,
                "lat": None if pd.isna(r.lat) else float(r.lat),
                "lon": None if pd.isna(r.lon) else float(r.lon)}
        body = r.description or ""
        for i, piece in enumerate(chunk_text(body, count_tokens, max_tokens, overlap_sentences)):
            chunks.append({
                **base,
                "chunk_id": f"{r.place_id}#d{i}",
                "chunk_type": "description",
                "text": f"{header}. {piece}"})
        if include_captions and r.caption:
            chunks.append({
                **base,
                "chunk_id": f"{r.place_id}#c0",
                "chunk_type": "caption",
                "text": f"{header}. Как выглядит (по фотографии): {r.caption}"})
    return chunks
