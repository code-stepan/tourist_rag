"""Light-weight EDA helpers (matplotlib only)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .data import decode_image


def overview(df: pd.DataFrame) -> pd.DataFrame:
    """Missing share, uniqueness and text length stats per column (image column excluded from lengths)."""
    rows = []
    for c in df.columns:
        if c == "row_id":
            continue
        s = df[c]
        empty = s.isna() | (s.astype(str).str.strip() == "")
        rows.append({"column": c, "empty_%": round(100 * empty.mean(), 2), "unique": s.nunique(),
                     "mean_len": round(s.astype(str).str.len().mean(), 1) if c != "image" else None})
    return pd.DataFrame(rows).set_index("column")


def plot_eda(df: pd.DataFrame, top_cities: int = 15):
    """Records per landmark, description length, records per city."""
    fig, ax = plt.subplots(1, 3, figsize=(17, 4))
    df.groupby("place_key").size().clip(upper=20).plot.hist(bins=20, ax=ax[0], title="записей на достопримечательность (cap 20)")
    df["description"].str.len().clip(upper=600).plot.hist(bins=40, ax=ax[1], title="длина описания, симв. (cap 600)")
    df[df["city"] != ""]["city"].value_counts().head(top_cities).sort_values().plot.barh(ax=ax[2], title=f"топ-{top_cities} городов")
    plt.tight_layout()
    return fig


def show_images(df: pd.DataFrame, n: int = 6, seed: int = 42, cols: int = 3):
    sample = df[df["image"] != ""].sample(min(n, len(df)), random_state=seed)
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
    for ax, r in zip(axes.ravel(), sample.itertuples()):
        ax.imshow(decode_image(r.image))
        ax.set_title(f"{r.name[:28]}\n{(r.caption or '')[:40]}", fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[len(sample):]:
        ax.axis("off")
    plt.tight_layout()
    return fig
