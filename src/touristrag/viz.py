"""PCA and UMAP projections of the embedding space (fitted independently), plotly figures."""
from __future__ import annotations

import numpy as np
import pandas as pd


def reduce_embeddings(embs: np.ndarray, method: str = "pca", seed: int = 42, **kw):
    """Fit on `embs`; returns (coords, fitted_model) so queries can be projected with `.transform`."""
    if method == "pca":
        from sklearn.decomposition import PCA

        model = PCA(n_components=2, random_state=seed)
    elif method == "umap":
        import umap

        model = umap.UMAP(n_components=2, n_neighbors=kw.get("n_neighbors", 15), min_dist=kw.get("min_dist", 0.1),
                          metric="cosine", random_state=seed)
    else:
        raise ValueError(method)
    return model.fit_transform(embs), model


def plot_embeddings(coords, labels, hover, title: str, top_labels: int = 12, query_coords=None, query_hover=None):
    import plotly.express as px
    import plotly.graph_objects as go

    labels = pd.Series(labels).fillna("—").astype(str)
    keep = labels.value_counts().head(top_labels).index
    colour = labels.where(labels.isin(keep), "другие")
    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "city": colour.to_numpy(), "name": list(hover)})
    fig = px.scatter(df, x="x", y="y", color="city", hover_name="name", title=title, opacity=0.75,
                     color_discrete_map={"другие": "#c8c8c8"})
    fig.update_traces(marker={"size": 6})
    if query_coords is not None:
        fig.add_trace(go.Scatter(x=query_coords[:, 0], y=query_coords[:, 1], mode="markers", name="вопросы",
                                 text=[] if query_hover is None else list(query_hover), marker={"symbol": "x", "size": 9, "color": "black"}))
    fig.update_layout(legend_title_text="город", height=620)
    return fig
