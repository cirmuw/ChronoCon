from __future__ import annotations

from ra_utils.features.embeddings import embeddings_to_dataframe

from typing import Dict, Optional, Sequence, Union, Tuple
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



# Optional UMAP dependency is handled gracefully
try:
    import umap
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

from sklearn.manifold import TSNE


def _prepare_df_and_Z(
    df: Optional[pd.DataFrame],
    per_loader: Optional[Dict[str, Dict[str, Union[np.ndarray, torch.Tensor, Sequence[str]]]]],
    latent_cols: Optional[Sequence[str]],
) -> tuple[pd.DataFrame, np.ndarray, Sequence[str]]:
    if df is None and per_loader is None:
        raise ValueError("Provide either `df` or `per_loader`.")
    if df is None:
        df = embeddings_to_dataframe(per_loader)
    if df.empty:
        raise ValueError("No data to plot. The provided DataFrame is empty.")

    if latent_cols is None:
        latent_cols = [c for c in df.columns if c.startswith("z_")]
        if not latent_cols:
            raise ValueError("Could not infer latent columns. Pass `latent_cols` or ensure columns start with 'z_'.")

    Z = df[latent_cols].to_numpy(dtype=float)
    return df, Z, latent_cols

from typing import Literal

def _scatter_2d(
    fig: plt.Figure,
    ax: plt.Axes,
    proj_df: pd.DataFrame,
    color_by: str,
    style_by: Optional[str],
    point_size: float,
    alpha: float,
    equal_aspect: bool,
    title: str,
    loc : Literal["best", "out_right", "out_bottom"] = "best"
) -> None:
    if color_by not in proj_df.columns:
        raise ValueError(f"`color_by`='{color_by}' not found in DataFrame columns.")

    color_values = proj_df[color_by]
    is_categorical = (
        (color_values.dtype == "object")
        or pd.api.types.is_categorical_dtype(color_values)
        or (color_values.nunique() < 20 and not np.issubdtype(color_values.dtype, np.number))
    )

    # Marker styles
    markers = ("o", "s", "D", "v", "^", "<", ">", "P", "X", "*", "h", "H")
    if style_by is not None and style_by in proj_df.columns:
        marker_values = proj_df[style_by].astype(str)
        marker_categories = marker_values.unique().tolist()
        marker_map = {cat: markers[i % len(markers)] for i, cat in enumerate(marker_categories)}
    else:
        marker_values = pd.Series([None] * len(proj_df), index=proj_df.index)
        marker_map = {None: "o"}

    handles, labels = [], []
    for mcat, mdf in proj_df.groupby(marker_values):
        marker = marker_map[mcat]
        if is_categorical:
            for cat, cdf in mdf.groupby(color_values.loc[mdf.index]):
                h = ax.scatter(
                    cdf["X1"].to_numpy(),
                    cdf["X2"].to_numpy(),
                    s=point_size,
                    alpha=alpha,
                    marker=marker,
                    label=f"{style_by}={mcat} | {color_by}={cat}" if mcat is not None else f"{color_by}={cat}",
                )
                handles.append(h); labels.append(h.get_label())
        else:
            sc = ax.scatter(
                mdf["X1"].to_numpy(),
                mdf["X2"].to_numpy(),
                s=point_size,
                alpha=alpha,
                marker=marker,
                c=mdf[color_by].to_numpy(dtype=float),
            )
            if "._scatter_colorbar_done" not in ax.__dict__:
                cb = fig.colorbar(sc, ax=ax); cb.set_label(color_by)
                ax.__dict__["._scatter_colorbar_done"] = True
            if mcat is not None:
                handles.append(sc); labels.append(f"{style_by}={mcat}")

    if ((handles and is_categorical) or
        (handles and not is_categorical and style_by is not None and style_by in proj_df.columns)): 
        if loc == "best":
            ax.legend(handles=handles, labels=labels, frameon=False, loc="best")
        elif loc == "out_right":
            ax.legend(
            handles=handles,
            labels=labels,
            frameon=False,
            loc="upper left",        # anchor corner of legend
            bbox_to_anchor=(1.05, 1) # shift outside the axes
        )
        elif loc == "out_bottom":
            ax.legend(
            handles=handles,
            labels=labels,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1), # center below axes
            ncol=3                      # spread entries across columns
        )

    

    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title(title)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)



def pca_scatter_embeddings(
    *,
    df: Optional[pd.DataFrame] = None,
    per_loader: Optional[Dict[str, Dict[str, Union[np.ndarray, torch.Tensor, Sequence[str]]]]] = None,
    latent_cols: Optional[Sequence[str]] = None,
    color_by: str = "score",
    style_by: Optional[str] = "loader_name",   # marker per category, if present
    standardize: bool = True,
    n_components: int = 2,
    figsize_mm: tuple[int, int] = (120, 100),
    point_size: float = 10.0,
    alpha: float = 0.6,
    random_state: int = 42,
    equal_aspect: bool = False,
    return_data: bool = True,
    ax: Optional[plt.Axes] = None,
    legend_loc : Literal["best", "out_right", "out_bottom"] = "best"
):
    """
    PCA scatter plot for latent embeddings, implemented using the shared helpers:
    `_prepare_df_and_Z` and `_scatter_2d`.

    Parameters mirror the earlier version; `ax=None` creates a new figure, otherwise
    the provided axis is reused.
    """
    # --- prepare inputs and latent matrix (reuses the shared helper) ---
    df, Z, latent_cols = _prepare_df_and_Z(df, per_loader, latent_cols)

    # --- PCA fit / transform ---
    X = StandardScaler().fit_transform(Z) if standardize else Z
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    ev = pca.explained_variance_ratio_
    ev1 = float(ev[0]) if len(ev) > 0 else 0.0
    ev2 = float(ev[1]) if len(ev) > 1 else 0.0

    # --- figure / axis setup ---
    if ax is None:
        mm2in = 1 / 25.4
        fig, ax = plt.subplots(figsize=(figsize_mm[0] * mm2in, figsize_mm[1] * mm2in))
    else:
        fig = ax.figure

    # --- build projected DataFrame consistent with other embedders ---
    proj_df = df.copy()
    proj_df["X1"] = X_pca[:, 0]
    proj_df["X2"] = X_pca[:, 1]

    # --- reuse the generic scatter helper, then adjust labels with EV info ---
    _scatter_2d(
        fig=fig,
        ax=ax,
        proj_df=proj_df,
        color_by=color_by,
        style_by=style_by,
        point_size=point_size,
        alpha=alpha,
        equal_aspect=equal_aspect,
        title="PCA of Latent Space",
        loc=legend_loc
    )

    # Overwrite axis labels to include explained variance (keeps helper generic)
    ax.set_xlabel(f"PC1 ({ev1*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({ev2*100:.1f}% var)")

    out = {
        "pca": pca,
        "X_pca": X_pca,
        "explained_var": ev,
        "proj_df": proj_df[["X1", "X2"] + [c for c in df.columns if c not in latent_cols]],
    }
    return (fig, ax, out) if return_data else (fig, ax)




def umap_scatter_embeddings(
    *,
    df: Optional[pd.DataFrame] = None,
    per_loader: Optional[Dict[str, Dict[str, Union[np.ndarray, torch.Tensor, Sequence[str]]]]] = None,
    latent_cols: Optional[Sequence[str]] = None,
    color_by: str = "score",
    style_by: Optional[str] = "loader_name",
    standardize: bool = True,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
    figsize_mm: tuple[int, int] = (120, 100),
    point_size: float = 10.0,
    alpha: float = 0.6,
    equal_aspect: bool = False,
    return_data: bool = True,
    ax: Optional[plt.Axes] = None,
    legend_loc : Literal["best", "out_right", "out_bottom"] = "best"
):
    """
    UMAP scatter for latent embeddings (same feel as pca_scatter_embeddings).

    Requires umap-learn. If it's not installed, a clear error is raised.
    """
    if not _HAS_UMAP:
        raise ImportError("umap-learn is required for umap_scatter_embeddings. Install via `pip install umap-learn`.")

    df, Z, latent_cols = _prepare_df_and_Z(df, per_loader, latent_cols)

    X = StandardScaler().fit_transform(Z) if standardize else Z
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    X_umap = reducer.fit_transform(X)

    # Figure / axis
    if ax is None:
        mm2in = 1 / 25.4
        fig, ax = plt.subplots(figsize=(figsize_mm[0] * mm2in, figsize_mm[1] * mm2in))
    else:
        fig = ax.figure

    proj_df = df.copy()
    proj_df["X1"] = X_umap[:, 0]
    proj_df["X2"] = X_umap[:, 1]

    _scatter_2d(fig, ax, proj_df, color_by, style_by, point_size, alpha, equal_aspect, "UMAP of Latent Space", loc=legend_loc)

    out = {
        "umap": reducer,
        "X_embedded": X_umap,
        "proj_df": proj_df[["X1", "X2"] + [c for c in df.columns if c not in latent_cols]],
    }
    return (fig, ax, out) if return_data else (fig, ax)


def tsne_scatter_embeddings(
    *,
    df: Optional[pd.DataFrame] = None,
    per_loader: Optional[Dict[str, Dict[str, Union[np.ndarray, torch.Tensor, Sequence[str]]]]] = None,
    latent_cols: Optional[Sequence[str]] = None,
    color_by: str = "score",
    style_by: Optional[str] = "loader_name",
    standardize: bool = True,
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: Union[str, float] = "auto",
    early_exaggeration: float = 12.0,
    n_iter: int = 1000,
    init: str = "pca",
    metric: str = "euclidean",
    random_state: int = 42,
    figsize_mm: tuple[int, int] = (120, 100),
    point_size: float = 10.0,
    alpha: float = 0.6,
    equal_aspect: bool = False,
    return_data: bool = True,
    ax: Optional[plt.Axes] = None,
    legend_loc : Literal["best", "out_right", "out_bottom"] = "best"
):
    """
    t-SNE scatter for latent embeddings (same feel as pca_scatter_embeddings).

    Notes:
    - `perplexity` should be < N/3 (N = number of samples).
    - `standardize=True` usually helps if latent dims vary in scale.
    """
    df, Z, latent_cols = _prepare_df_and_Z(df, per_loader, latent_cols)

    X = StandardScaler().fit_transform(Z) if standardize else Z
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        early_exaggeration=early_exaggeration,
        n_iter=n_iter,
        init=init,
        metric=metric,
        random_state=random_state,
        verbose=0,
        n_jobs=None,  # uses single thread; change if you prefer multithreading (sklearn >=1.4)
    )
    X_tsne = tsne.fit_transform(X)

    # Figure / axis
    if ax is None:
        mm2in = 1 / 25.4
        fig, ax = plt.subplots(figsize=(figsize_mm[0] * mm2in, figsize_mm[1] * mm2in))
    else:
        fig = ax.figure

    proj_df = df.copy()
    proj_df["X1"] = X_tsne[:, 0]
    proj_df["X2"] = X_tsne[:, 1]

    _scatter_2d(fig, ax, proj_df, color_by, style_by, point_size, alpha, equal_aspect, "t-SNE of Latent Space",
                loc=legend_loc)

    out = {
        "tsne": tsne,
        "X_embedded": X_tsne,
        "proj_df": proj_df[["X1", "X2"] + [c for c in df.columns if c not in latent_cols]],
    }
    return (fig, ax, out) if return_data else (fig, ax)


