
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


# def metrics_to_text(m):
#         txt0 = (f"$\\mathrm{{RMSE}}= {m['rmse']:2.2f}\;$"
#                 "\n"
#                 f"$\\rho= {m['spearman_corr']:2.2f}\;$")
#         txt_ICC_psych = (
#             "\n"
#             f"$\\mathrm{{ICC}}= {m['ICC_psych']:2.2f}$"
#             f" $ [{m['ICC_psych_lower']:2.2f}, {m['ICC_psych_upper']:2.2f}]$"
#             "\n"
#             f"$\\mathrm{{N}}= {m['ICC_n']}$"
#         )
#         txt = txt0 + txt_ICC_psych
#         return txt 

def metrics_to_text(m):
    # --- RMSE (+ CI if present)
    rmse_txt = f"$\\mathrm{{RMSE}}= {m['rmse']:2.2f}$"
    if "rmse_CI95_lower" in m and "rmse_CI95_upper" in m:
        rmse_txt += f" $ [{m['rmse_CI95_lower']:2.2f}, {m['rmse_CI95_upper']:2.2f}]$"
    rmse_txt += ""

    # --- Prefer Pearson r; fallback to Spearman rho
    corr_key, corr_label = None, None
    if "pearson_corr" in m:
        corr_key, corr_label = "pearson_corr", "\\rho   "  # "r"
    elif "spearman_corr" in m:
        corr_key, corr_label = "spearman_corr", "\\rho   "

    corr_txt = ""
    if corr_key is not None:
        corr_txt = f"${corr_label}= {m[corr_key]:2.2f}$"
        lo_key, hi_key = f"{corr_key}_CI95_lower", f"{corr_key}_CI95_upper"
        if lo_key in m and hi_key in m:
            corr_txt += f" $ [{m[lo_key]:2.2f}, {m[hi_key]:2.2f}]$"
        corr_txt += ""

    # txt0 = rmse_txt + "\n" + corr_txt

    # --- ICC (psych) + N (if present)
    txt_ICC_psych = ""
    if "ICC_psych" in m:
        txt_ICC_psych = (
            "\n"
            f"$\\mathrm{{ICC}} = {m['ICC_psych']:2.2f}$"
        )
        if "ICC_psych_lower" in m and "ICC_psych_upper" in m:
            txt_ICC_psych += f" $ [{m['ICC_psych_lower']:2.2f}, {m['ICC_psych_upper']:2.2f}]$"
        if "ICC_n" in m:
            txt_ICC_psych_N = f"$\\mathrm{{N}}= {int(m['ICC_n'])}$"



    return  rmse_txt +  txt_ICC_psych + "\n" + corr_txt + "\n" + txt_ICC_psych_N


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
        metrics_loc: tuple[float, float] = (0.05, 0.95),
        ax_scatter=None
    ):
    """
    Draws either a simple scatter or a joint scatter+histogram plot.
    If regplot=True, overlays a regression line via seaborn.regplot.

    Supports drawing directly into an existing scatter axis if `ax_scatter` is provided.

    Returns
    -------
    (fig, (ax_histx, ax_scatter, ax_histy), metrics)
      - ax_histx or ax_histy may be None if plot_histograms=False or ax_scatter is provided.
      - metrics: dict with computed metrics, or None if calculate_and_add_metrics=False
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
    m = None
    if calculate_and_add_metrics:
        m = calculate_some_classification_metrics(
            all_labels=x.values.astype(float),
            all_preds=y.values.astype(float),
            calc_ICC3=2,
            add_classification_metrics=False,
            add_spearman=False,
            add_pearson=True,
            add_kappa=True,
            calc_psych_ICC=2,
            icc=icc, 
            calculate_CI=True # 95% CI
        )
        txt = metrics_to_text(m)

    # If user provided an external scatter axis, draw only scatter there
    if ax_scatter is not None:
        if regplot:
            sns.regplot(
                x=x, y=y, ax=ax_scatter,
                scatter_kws={'alpha': 0.3, 's': 8},
                line_kws={'color': 'black', 'lw': 1},
                truncate=False
            )
        else:
            ax_scatter.plot(x, y, '.', alpha=0.3, markersize=8)
        # Diagonal reference line
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax_scatter.plot(lims, lims, '--', color='gray', lw=1, alpha=0.7)
        ax_scatter.set_xlabel(f"Ground-truth scores ({name})")
        ax_scatter.set_ylabel(f"Predicted scores ({name})")
        ax_scatter.grid()
        if calculate_and_add_metrics:
            ax_scatter.text(
                *metrics_loc, txt,
                transform=ax_scatter.transAxes,
                ha='left', va='top', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3)
            )
        # No new figure created
        return None, (None, ax_scatter, None), m

    # If no histograms requested, fall back to simple scatter layout
    if not plot_histograms:
        fig, ax = plt.subplots(figsize=figsize)
        if regplot:
            sns.regplot(
                x=x, y=y, ax=ax,
                scatter_kws={'color': 'C0', 'alpha': 0.3, 's': 8},
                line_kws={'color': 'blue', 'lw': 1},
                truncate=False
            )
        else:
            ax.plot(x, y, '.', color='C0', alpha=0.3, markersize=8)
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, '-', color='gray', lw=1, alpha=0.7)
        ax.set_xlabel(f"Ground-truth scores ({name})")
        ax.set_ylabel(f"Predicted scores ({name})")
        ax.grid()
        if calculate_and_add_metrics:
            ax.text(
                *metrics_loc, txt,
                transform=ax.transAxes,
                ha='left', va='top', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3)
            )
        return fig, (None, ax, None), m

    # Joint histogram + scatter layout
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        2, 2,
        height_ratios=[1, 4],
        width_ratios=[4, 1],
        hspace=0.05,
        wspace=0.05
    )
    ax_histx = fig.add_subplot(gs[0, 0])
    ax_sc = fig.add_subplot(gs[1, 0], sharex=ax_histx)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_sc)

    # Histograms
    ax_histx.hist(x, bins=bins)
    ax_histx.set_ylabel('Count')
    ax_histx.tick_params(axis='x', labelbottom=False)
    ax_histy.hist(y, bins=bins, orientation='horizontal')
    ax_histy.set_xlabel('Count')
    ax_histy.tick_params(axis='y', labelleft=False)

    # Scatter or regplot
    if regplot:
        sns.regplot(x=x, y=y, ax=ax_sc,
                    scatter_kws={'alpha': 0.3, 's': 5},
                    line_kws={'color': 'black', 'lw': 1},
                    truncate=False)
    else:
        ax_sc.plot(x, y, '.', alpha=0.3, markersize=8)
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax_sc.plot(lims, lims, '--', color='gray', lw=1, alpha=0.7)

    ax_sc.set_xlabel(f"Ground-truth scores ({name})")
    ax_sc.set_ylabel(f"Predicted scores ({name})")
    ax_sc.grid()
    if calculate_and_add_metrics:
        # ax_sc.text(
        #     *metrics_loc, txt,
        #     transform=ax_sc.transAxes,
        #     ha='left', va='top', fontsize=12,
        #     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3)
        # )
        with plt.rc_context({'font.family': 'monospace', 'mathtext.default': 'tt'}):
            ax_sc.text(
                *metrics_loc, txt.replace(r'\mathrm', r'\mathtt'),  # optional: steer math to tt
                transform=ax_sc.transAxes,
                ha='left', va='top', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3)
            )

    return fig, (ax_histx, ax_sc, ax_histy), m



# # ------------------------------------------------------------
# # convenience wrappers for your two tables
# # ------------------------------------------------------------
def plot_SHS_deltas(df_delta, *, name: str = "ERO F", **kwargs):
    """
    Joint plot for df_delta.
    
    Returns
    -------
    (fig, (ax_histx, ax_scatter, ax_histy), metrics)
    """
    return joint_hist_scatter(
        df_delta,
        true_col="labels_summed_extrapolated_delta",
        pred_col="preds_summed_extrapolated_delta",
        name=name,
        **kwargs
    )

def plot_SHS_sums(df_summed, name: str = "ERO F", regplot=False, **kwargs):
    """
    Joint plot for df_summed.
    
    Returns
    -------
    (fig, (ax_histx, ax_scatter, ax_histy), metrics)
    """
    return joint_hist_scatter(
        df_summed,
        true_col="labels_summed_extrapolated",
        pred_col="preds_summed_extrapolated",
        regplot=regplot,
        name=name,

        **kwargs
    )
