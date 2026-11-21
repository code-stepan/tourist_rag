"""Outlier / junk detection (memes, banners, selfies...) and stratified test sampling.

Three complementary signals, intentionally *semi-automatic* — final decisions are made after
a manual look at `review_table` (the assignment explicitly asks for it):
  1. TF-IDF: atypicality of vocabulary (distance to corpus centroid) + rare/frequent term report
  2. Rules: regex lexicon over BLIP captions / descriptions (RU + EN)
  3. Zero-shot embeddings: similarity to "junk" prompts minus similarity to "landmark" prompts
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_PLACE_WORDS = (r"building|church|tower|monument|statue|bridge|castle|palace|temple|street|city|square|park|"
                r"museum|cathedral|fountain|house|gate|wall|fortress|mosque|monastery|clock|skyline|river")
JUNK_PATTERNS = [
    r"\bmeme\b", r"screen ?shot", r"\bbanner\b", r"\bposter\b", r"\blogo\b", r"\bselfie\b",
    r"cell ?phone", r"\blaptop\b", r"\bcomputer\b", r"\bcartoon\b", r"\bdiagram\b", r"\bflyer\b",
    r"text (message|that says|on a)", r"\bmap of\b",
    r"\bмем\b", r"скриншот", r"баннер", r"логотип", r"селфи", r"\bплакат\b",
]
_PERSON_ONLY = re.compile(r"^(a|an|the)\s+(close up of\s+)?(man|woman|person|boy|girl|group of people|couple)\b", re.I)
_PLACE_RE = re.compile(_PLACE_WORDS, re.I)

JUNK_PROMPTS = [
    "мем или смешная картинка", "скриншот экрана телефона или компьютера", "рекламный баннер или плакат",
    "селфи или портрет человека", "логотип или текст на картинке", "фотография еды или предмета в помещении",
]
GOOD_PROMPTS = [
    "историческая достопримечательность или памятник", "архитектурное здание, собор, замок или дворец",
    "туристическое место в городе: площадь, мост, башня, музей", "памятник архитектуры или монумент",
]


# --------------------------------------------------------------------------- TF-IDF
def tfidf_terms(texts, top_n: int = 25, min_df: int = 2) -> dict[str, pd.DataFrame]:
    """Most frequent and rarest terms — the raw material for spotting junk vocabulary."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(lowercase=True, min_df=min_df, sublinear_tf=True)
    X = vec.fit_transform(texts)
    terms = np.array(vec.get_feature_names_out())
    dfc = np.asarray((X > 0).sum(axis=0)).ravel()
    table = pd.DataFrame({"term": terms, "df": dfc, "idf": vec.idf_})
    return {
        "frequent": table.sort_values("df", ascending=False).head(top_n).reset_index(drop=True),
        "rare": table.sort_values(["idf", "term"], ascending=[False, True]).head(top_n).reset_index(drop=True),
    }


def tfidf_atypicality(texts) -> np.ndarray:
    """1 - cosine(doc, corpus centroid) in TF-IDF space; high = unusual vocabulary."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    try:
        X = TfidfVectorizer(sublinear_tf=True, min_df=2).fit_transform(texts)
    except ValueError:  # empty vocabulary
        return np.zeros(len(texts))
    centroid = np.asarray(X.mean(axis=0))
    return 1.0 - cosine_similarity(X, centroid).ravel()


# --------------------------------------------------------------------------- rules
def rule_flags(df: pd.DataFrame) -> pd.Series:
    """'' if clean, else a short reason. Looks at BLIP caption and the description."""
    reasons = []
    for cap, desc in zip(df["caption"], df["description"]):
        text = f"{cap} {desc}".lower()
        hit = next((p for p in JUNK_PATTERNS if re.search(p, text)), None)
        if hit:
            reasons.append(f"pattern:{hit}")
        elif _PERSON_ONLY.search(cap or "") and not _PLACE_RE.search(cap or ""):
            reasons.append("person_only")
        elif len((desc or "").strip()) == 0 and len((cap or "").strip()) == 0:
            reasons.append("empty")
        else:
            reasons.append("")
    return pd.Series(reasons, index=df.index)


# --------------------------------------------------------------------------- zero-shot
def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    return (x - med) / (mad if mad > 1e-9 else (x.std() or 1.0))


def zero_shot_margin(texts, embedder, junk_prompts=None, good_prompts=None) -> np.ndarray:
    """max cos(text, junk prompt) - max cos(text, landmark prompt)."""
    junk = embedder.encode(junk_prompts or JUNK_PROMPTS, kind="query")
    good = embedder.encode(good_prompts or GOOD_PROMPTS, kind="query")
    emb = embedder.encode(list(texts), kind="passage")
    return (emb @ junk.T).max(axis=1) - (emb @ good.T).max(axis=1)


def flag_outliers(df: pd.DataFrame, embedder=None, z_threshold: float = 3.5) -> pd.DataFrame:
    """Adds rule_flag / tfidf_atyp / zs_margin / junk_score / is_junk. Does NOT drop anything."""
    out = df.copy()
    texts = (out["name"] + ". " + out["description"] + " " + out["caption"]).tolist()
    out["rule_flag"] = rule_flags(out)
    out["tfidf_atyp"] = tfidf_atypicality(texts)
    if embedder is not None:
        out["zs_margin"] = zero_shot_margin(texts, embedder)
        zs_z = robust_z(out["zs_margin"].to_numpy())
    else:
        out["zs_margin"] = 0.0
        zs_z = np.zeros(len(out))
    tf_z = robust_z(out["tfidf_atyp"].to_numpy())
    out["junk_score"] = (out["rule_flag"] != "") * 5.0 + np.clip(zs_z, 0, None) + 0.5 * np.clip(tf_z, 0, None)
    out["is_junk"] = (out["rule_flag"] != "") | (zs_z > z_threshold)
    return out


def review_table(flagged: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    cols = ["place_id", "name", "city", "caption", "rule_flag", "zs_margin", "tfidf_atyp", "junk_score", "is_junk"]
    return flagged.sort_values("junk_score", ascending=False).head(n)[[c for c in cols if c in flagged]]


def drop_places(df: pd.DataFrame, place_ids) -> pd.DataFrame:
    return df[~df["place_id"].isin(set(place_ids))].reset_index(drop=True)


# --------------------------------------------------------------------------- sampling
def stratified_sample(places: pd.DataFrame, n: int = 100, by: str = "city", seed: int = 42) -> pd.DataFrame:
    """Pick `n` distinct landmarks spread as evenly as possible over `by` (round-robin over strata).

    One row per landmark (places are already aggregated), so two samples never describe the same
    place — which answers the "clean samples of one place give near-identical descriptions" warning.
    """
    rng = np.random.default_rng(seed)
    groups = {k: list(rng.permutation(g.index.to_numpy())) for k, g in places.groupby(by, sort=True)}
    order = list(groups)
    picked: list[int] = []
    while len(picked) < min(n, len(places)):
        progressed = False
        for k in rng.permutation(order):
            if groups[k]:
                picked.append(groups[k].pop())
                progressed = True
                if len(picked) >= n:
                    break
        if not progressed:
            break
    return places.loc[picked].reset_index(drop=True)
