"""Loading, schema detection, normalisation and per-place aggregation of the landmarks dataset."""
from __future__ import annotations

import base64
import os
import re
from io import BytesIO

import numpy as np
import pandas as pd

from .utils import norm_text

CANONICAL = ("name", "wikidata_id", "city", "lon", "lat", "description", "image", "caption")
REQUIRED = ("name", "description")

# order matters: `caption` is resolved before `description` so that e.g. `image_description`
# is not mistaken for the WikiData description.
_CANDIDATES = {
    "name": ["name", "title", "label", "place", "place_name", "landmark", "название"],
    "wikidata_id": ["wikidata_id", "wikidata", "wiki_id", "qid", "id", "item"],
    "city": ["city", "town", "location", "город"],
    "lon": ["lon", "lng", "longitude", "long", "x"],
    "lat": ["lat", "latitude", "y"],
    "caption": ["caption", "blip", "blip_caption", "image_description", "generated_description",
                "image_caption", "blip_description", "generated_caption", "en_txt", "en_text", "caption_en"],
    "description": ["description", "desc", "wiki_description", "text", "описание"],
    "image": ["image", "img", "image_base64", "photo", "base64"],
}
_SHORT = {"id", "x", "y", "lon", "lat", "text", "desc", "img", "long"}  # exact match only


def detect_schema(columns, overrides: dict | None = None) -> dict:
    """Map canonical field names to actual column names (exact match first, then substring)."""
    overrides = overrides or {}
    lowered = {str(c).lower().strip(): c for c in columns}
    used: set = set(overrides.values())
    schema: dict = {}
    for key, cands in _CANDIDATES.items():
        if key in overrides:
            schema[key] = overrides[key]
            continue
        found = next((lowered[c] for c in cands if c in lowered and lowered[c] not in used), None)
        if found is None:
            for low, orig in lowered.items():
                if orig in used:
                    continue
                if any(len(c) >= 4 and c not in _SHORT and c in low for c in cands):
                    found = orig
                    break
        schema[key] = found
        if found is not None:
            used.add(found)
    if not schema.get("caption"):
        import warnings
        warnings.warn(f"No image-caption (BLIP) column detected in {list(columns)}; visual chunks will be empty. "
                      f"Pass column_overrides={{'caption': '<col>'}}.")
    missing = [k for k in REQUIRED if not schema.get(k)]
    if missing:
        raise KeyError(
            f"Could not detect columns for {missing}. Columns found: {list(columns)}. "
            f"Pass column_overrides={{'name': '<col>', ...}}."
        )
    return schema


def standardize(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Return a frame with canonical columns only; text cleaned, coords numeric."""
    out = pd.DataFrame(index=df.index)
    for key in CANONICAL:
        col = schema.get(key)
        if col is not None:
            out[key] = df[col]
        else:
            out[key] = np.nan if key in ("lon", "lat") else ""
    for key in ("name", "wikidata_id", "city", "description", "caption"):
        out[key] = out[key].map(norm_text)
    out["image"] = out["image"].fillna("").astype(str)
    for key in ("lon", "lat"):
        out[key] = pd.to_numeric(out[key], errors="coerce")
    out = out.reset_index(drop=True)
    out.insert(0, "row_id", np.arange(len(out)))
    return out


def decode_image(b64: str):
    from PIL import Image

    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")


def download_dataset(url: str, path: str) -> str:
    import gdown

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        gdown.download(url, path, quiet=False)
    return path


def load_raw(path: str, url: str | None = None) -> pd.DataFrame:
    if not os.path.exists(path) and url:
        download_dataset(url, path)
    return pd.read_csv(path)


# --------------------------------------------------------------------------- places
def place_key(df: pd.DataFrame) -> pd.Series:
    """Stable identity of a landmark: WikiData id if present, else (name, city)."""
    wid = df["wikidata_id"].str.strip()
    fallback = df["name"].str.lower().str.strip() + "|" + df["city"].str.lower().str.strip()
    return pd.Series(np.where(wid != "", wid, fallback), index=df.index)


def _unique_texts(series, min_chars: int = 1, jaccard: float = 0.8) -> list[str]:
    """Distinct, non-trivial texts, longest first; near-duplicates (token Jaccard) removed."""
    texts = sorted({norm_text(t) for t in series if len(norm_text(t)) >= min_chars}, key=lambda t: (-len(t), t))
    kept, kept_tokens = [], []
    for t in texts:
        toks = set(re.findall(r"\w+", t.lower()))
        if any(len(toks & k) / max(1, len(toks | k)) >= jaccard for k in kept_tokens):
            continue
        kept.append(t)
        kept_tokens.append(toks)
    return kept


def select_description(descs: list[str], strategy: str = "concat", k: int = 3) -> str:
    """`longest`: best single description. `concat`: up to k distinct descriptions (diversity)."""
    if not descs:
        return ""
    if strategy == "longest":
        return descs[0]
    if strategy == "concat":
        return " ".join(descs[:k])
    raise ValueError(f"unknown strategy {strategy!r}")


def build_places(df: pd.DataFrame, strategy: str = "concat", max_descriptions: int = 3,
                 max_captions: int = 2, min_desc_chars: int = 10) -> pd.DataFrame:
    """Collapse multiple records of one landmark (different photos) into one searchable place."""
    df = df.copy()
    df["place_key"] = place_key(df)
    rows = []
    for i, (key, g) in enumerate(df.groupby("place_key", sort=False)):
        descs = _unique_texts(g["description"], min_desc_chars)
        caps = _unique_texts(g["caption"], 1)
        best = g["caption"].str.len().idxmax()
        rows.append({
            "place_id": f"p{i:05d}",
            "place_key": key,
            "name": g["name"].mode().iat[0],
            "city": g["city"].mode().iat[0],
            "lat": g["lat"].mean(),
            "lon": g["lon"].mean(),
            "description": select_description(descs, strategy, max_descriptions),
            "caption": "; ".join(caps[:max_captions]),
            "n_records": len(g),
            "image": g.loc[best, "image"],
        })
    places = pd.DataFrame(rows)
    keep = (places["description"].str.len() > 0) | (places["caption"].str.len() > 0)
    return places[keep].reset_index(drop=True)
