
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from ra_utils.training.scores_SHS.scores_SHS_training_lib import (
    calculate_some_classification_metrics
)

import seaborn as sns
import pandas as pd
# ------------------------------------------------------------
# generic joint histogram + scatter
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# def joint_hist_scatter(
#         df,
#         true_col: str,
#         pred_col: str,
#         *,
#         name: str = "ERO F",
#         bins: int = 30,
#         figsize=(8, 8), 
#         icc="ICC3",
#         calculate_and_add_metrics=True,
#         plot_histograms=True,
#         regplot=False,
#         metrics_loc: tuple[float, float] = (0.05, 0.95)  # axes‐fraction coords
#     ):
#     """
#     Draws either:
#       - a full‐figure scatter (if plot_histograms=False), or
#       - a (top‐hist, scatter, right‐hist) layout (if plot_histograms=True).

#     Returns
#     -------
#     (fig, (ax_histx, ax_scatter, ax_histy))
#       - If plot_histograms=False, ax_histx and ax_histy will be None.
#     """
#     x = df[true_col]
#     y = df[pred_col]

#     # optional metrics overlay
#     if calculate_and_add_metrics:
#         m = calculate_some_classification_metrics(
#             all_labels=np.array(x, dtype=np.float64),
#             all_preds=np.array(y, dtype=np.float64),
#             calc_ICC3=2,
#             add_classification_metrics=False,
#             add_spearman=True,
#             add_kappa=True,
#             calc_psych_ICC=2,
#             icc=icc
#         )
#         txt0 = (
#             fr"$\mathrm{{RMSE}}= {m['rmse']:2.2f}\;$"
#             "\n"
#             fr"$\rho= {m['spearman_corr']:2.2f}\;$"
#         )

#         txt_ICC_psych = (
#             "\n"
#             fr"$\mathrm{{ICC}}= {m['ICC_psych']:2.2f}$"
#             " "
#             fr"$ [{m['ICC_psych_lower']:2.2f}, {m['ICC_psych_upper']:2.2f}]$"
#             "\n"
#             fr"$\mathrm{{N}}= {m['ICC_n']}$"
#         )

#         txt_ICC_pingouin = (
#             "\n"
#             fr"$\mathrm{{ICC}}= {m['ICC']:2.2f}$"
#             " "
#             fr"$ [{m['ICC_CI95_lower']:2.2f}, {m['ICC_CI95_upper']:2.2f}]$"
#             "\n"
#             fr"$\mathrm{{N}}= {int((m['ICC_psych_df1'] + m['ICC_psych_df2'])/2) + 1 }$"
#         )
#         txt = (txt0 + #"\npingouin:" + txt_ICC_pingouin  
#                #+ "\nR psych:" + 
#                txt_ICC_psych)



#     if not plot_histograms:
#         # ---- only scatter ----
#         fig, ax_scatter = plt.subplots(figsize=figsize)
#         ax = ax_scatter
#         # scatter
#         ax.plot(x, y, ".", color="red", alpha=0.3, markersize=8)
#         lims = [min(x.min(), y.min()), max(x.max(), y.max())]
#         ax.plot(lims, lims, "-", color="gray", lw=1, alpha=0.7)
#         # labels
#         is_delta = "_delta" in true_col.lower()
#         prefix = "Δ summed" if is_delta else "summed"
#         ax.set_xlabel(f"Ground‐truth {prefix} scores ({name})")
#         ax.set_ylabel(f"Predicted {prefix} scores ({name})")
#         ax.grid()

#         # optional metrics
#         if calculate_and_add_metrics:
#             ax.text(
#                 *metrics_loc, txt,
#                 transform=ax.transAxes,
#                 ha="left", va="top",
#                 fontsize=9,
#                 bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=3)
#             )

#         return fig, (None, ax, None)

#     # ---- full joint with histograms ----
#     fig = plt.figure(figsize=figsize)
#     gs = gridspec.GridSpec(
#         nrows=2, ncols=2,
#         height_ratios=[1, 4],
#         width_ratios=[4, 1],
#         hspace=0.05, wspace=0.05
#     )

#     ax_histx  = fig.add_subplot(gs[0, 0])
#     ax_scatter = fig.add_subplot(gs[1, 0], sharex=ax_histx)
#     ax_histy  = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

#     # top histogram
#     ax_histx.hist(x, bins=bins, log=False)
#     ax_histx.set_ylabel("Count")
#     ax_histx.tick_params(axis="x", labelbottom=False)

#     # right histogram
#     ax_histy.hist(y, bins=bins, orientation="horizontal", log=False)
#     ax_histy.set_xlabel("Count")
#     ax_histy.tick_params(axis="y", labelleft=False)

#     # main scatter
#     ax_scatter.plot(x, y, ".", color="black", alpha=0.3, markersize=5)
#     lims = [min(x.min(), y.min()), max(x.max(), y.max())]
#     ax_scatter.plot(lims, lims, "--")

#     is_delta = "_delta" in true_col.lower()
#     prefix = "Δ summed" if is_delta else "summed"
#     ax_scatter.set_xlabel(f"Ground‐truth {prefix} scores ({name})")
#     ax_scatter.set_ylabel(f"Predicted {prefix} scores ({name})")
#     ax_scatter.grid()

#     # optional metrics overlay
#     if calculate_and_add_metrics:
#         ax_scatter.text(
#             *metrics_loc, txt,
#             transform=ax_scatter.transAxes,
#             ha="left", va="top",
#             fontsize=9,
#             bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=3)
#         )

#     return fig, (ax_histx, ax_scatter, ax_histy)


def joint_hist_scatter(
        df: pd.DataFrame,
        true_col: str,
        pred_col: str,
        *,
        name: str = "ERO F",
        bins: int = 30,
        figsize=(8, 8),  
        icc: str = "ICC3",
        calculate_and_add_metrics: bool = True,
        plot_histograms: bool = True,
        regplot: bool = False,
        metrics_loc: tuple[float, float] = (0.05, 0.95)
    ):
    """
    Draws either a simple scatter or a joint scatter+histogram plot.
    If regplot=True, overlays a regression line via seaborn.regplot.

    Returns
    -------
    (fig, (ax_histx, ax_scatter, ax_histy))
      - ax_histx or ax_histy may be None if plot_histograms=False.
    """
    # Ensure numeric and drop NaNs
    x_raw = df[true_col]
    y_raw = df[pred_col]
    x = pd.to_numeric(x_raw, errors='coerce')
    y = pd.to_numeric(y_raw, errors='coerce')
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]

    # Compute metrics text
    if calculate_and_add_metrics:
        m = calculate_some_classification_metrics(
            all_labels=x.values.astype(float),
            all_preds=y.values.astype(float),
            calc_ICC3=2,
            add_classification_metrics=False,
            add_spearman=True,
            add_kappa=True,
            calc_psych_ICC=2,
            icc=icc
        )
    # optional metrics overlay
        txt0 = (
            fr"$\mathrm{{RMSE}}= {m['rmse']:2.2f}\;$"
            "\n"
            fr"$\rho= {m['spearman_corr']:2.2f}\;$"
        )

        txt_ICC_psych = (
            "\n"
            fr"$\mathrm{{ICC}}= {m['ICC_psych']:2.2f}$"
            " "
            fr"$ [{m['ICC_psych_lower']:2.2f}, {m['ICC_psych_upper']:2.2f}]$"
            "\n"
            fr"$\mathrm{{N}}= {m['ICC_n']}$"
        )

        txt_ICC_pingouin = (
            "\n"
            fr"$\mathrm{{ICC}}= {m['ICC']:2.2f}$"
            " "
            fr"$ [{m['ICC_CI95_lower']:2.2f}, {m['ICC_CI95_upper']:2.2f}]$"
            "\n"
            fr"$\mathrm{{N}}= {int((m['ICC_psych_df1'] + m['ICC_psych_df2'])/2) + 1 }$"
        )
        txt = (txt0 +# "\npingouin:" + txt_ICC_pingouin  
             #  + "\nR psych:" + 
               txt_ICC_psych)

    # Only scatter if requested
    if not plot_histograms:
        fig, ax_scatter = plt.subplots(figsize=figsize)
        if regplot:
            ax_scatter = sns.regplot(x=x, y=y, ax=ax_scatter,
                        scatter_kws={'color': 'C0', 'alpha': 0.3, 's': 8},
                        line_kws={'color': 'blue', 'lw': 1},
                        truncate=False)
        else:
            ax_scatter.plot(x, y, '.', color='C0', alpha=0.3, markersize=8)
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax_scatter.plot(lims, lims, '-', color='gray', lw=1, alpha=0.7)
        ax_scatter.set_xlabel(f"Ground-truth scores ({name})")
        ax_scatter.set_ylabel(f"Predicted scores ({name})")
        ax_scatter.grid()
        if calculate_and_add_metrics:
            ax_scatter.text(*metrics_loc, txt, transform=ax_scatter.transAxes,
                             ha='left', va='top', fontsize=12,
                             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))
        return fig, (None, ax_scatter, None)

    # Joint histogram + scatter layout
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 4], width_ratios=[4, 1],
                           hspace=0.05, wspace=0.05)
    ax_histx = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[1, 0], sharex=ax_histx)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

    # Histograms
    ax_histx.hist(x, bins=bins)
    ax_histx.set_ylabel('Count')
    ax_histx.tick_params(axis='x', labelbottom=False)
    ax_histy.hist(y, bins=bins, orientation='horizontal')
    ax_histy.set_xlabel('Count')
    ax_histy.tick_params(axis='y', labelleft=False)

    # Scatter or regplot
    if regplot:
        ax_scatter = sns.regplot(x=x, y=y, ax=ax_scatter,
                    scatter_kws={'alpha':0.3, 's':5},
                    line_kws={'color':'black','lw':1},
                    truncate=False)
    else:
        ax_scatter.plot(x, y, '.', alpha=0.3, markersize=8)
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax_scatter.plot(lims, lims, '--')

    ax_scatter.set_xlabel(f"Ground-truth scores ({name})")
    ax_scatter.set_ylabel(f"Predicted scores ({name})")
    ax_scatter.grid()
    if calculate_and_add_metrics:
        ax_scatter.text(*metrics_loc, txt, transform=ax_scatter.transAxes,
                         ha='left', va='top', fontsize=12,
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))
    return fig, (ax_histx, ax_scatter, ax_histy)



# ------------------------------------------------------------
# convenience wrappers for your two tables
# ------------------------------------------------------------
def plot_SHS_deltas(df_delta, *, name: str = "ERO F", **kwargs):
    """Joint plot for df_delta."""
    return joint_hist_scatter(
        df_delta,
        true_col="labels_summed_extrapolated_delta",
        pred_col="preds_summed_extrapolated_delta",
        name=name,
        **kwargs
    )

def plot_SHS_sums(df_summed, name: str = "ERO F", regplot=False, **kwargs):
    """Joint plot for df_summed."""
    return joint_hist_scatter(
        df_summed,
        true_col="labels_summed_extrapolated",
        pred_col="preds_summed_extrapolated",
        regplot=regplot,
        name=name,

        **kwargs
    )
