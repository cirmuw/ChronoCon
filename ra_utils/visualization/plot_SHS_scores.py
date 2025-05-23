
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from ra_utils.training.scores_SHS.scores_SHS_training_lib import (
    calculate_some_classification_metrics
)

# ------------------------------------------------------------
# generic joint histogram + scatter
# ------------------------------------------------------------
def joint_hist_scatter(
        df,
        true_col: str,
        pred_col: str,
        *,
        name: str = "ERO F",
        bins: int = 30,
        figsize=(8, 8), 
        calculate_and_add_metrics = True,
        metrics_loc: tuple[float, float] = (0.05, 0.95)  # axes-fraction coords
    ):
    """
    Draw a (top-hist, right-hist, scatter) plot for the given columns.

    Parameters
    ----------
    df        : pandas.DataFrame
    true_col  : column with ground-truth values
    pred_col  : column with predicted values
    name      : label inserted into axis texts (default "ERO F")
    bins      : histogram bin count
    figsize   : overall figure size

    Returns
    -------
    (fig, (ax_histx, ax_scatter, ax_histy))
    """

    x = df[true_col]
    y = df[pred_col]
    


    # ── figure layout ─────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(
            nrows=2, ncols=2,
            height_ratios=[1, 4],
            width_ratios=[4, 1],
            hspace=0.05, wspace=0.05
          )

    ax_histx  = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[1, 0], sharex=ax_histx)
    ax_histy  = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

    # ── top histogram ────────────────────────────────────────
    ax_histx.hist(x, bins=bins, log=False)
    ax_histx.set_ylabel("Count")
    ax_histx.tick_params(axis="x", labelbottom=False)

    # ── right histogram (horizontal) ─────────────────────────
    ax_histy.hist(y, bins=bins, orientation="horizontal", log=False)
    ax_histy.set_xlabel("Count")
    ax_histy.tick_params(axis="y", labelleft=False)

    # ── main scatter ─────────────────────────────────────────
    ax_scatter.plot(x, y, ".", color="black", alpha=0.3, markersize=5)
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax_scatter.plot(lims, lims, "--")

    # pretty axis labels based on whether we’re plotting deltas
    is_delta = "_delta" in true_col.lower()
    prefix   = "Δ summed" if is_delta else "summed"

    ax_scatter.set_xlabel(f"Ground-truth {prefix} scores ({name})")
    ax_scatter.set_ylabel(f"Predicted {prefix} scores ({name})")
    #ax_scatter.legend()
    
    # if calculate_and_add_metrics:
    #     m = calculate_some_classification_metrics(all_labels=x, 
    #                                     all_preds=y,
    #                                     calc_ICC3=2, 
    #                                     add_classification_metrics=False, 
    #                                     add_spearman=True
    #                                     )
    #     s = f"rmse = {m['rmse']:2.2f}; ρ = {m['spearman_corr']:2.2f}; ICC = {m['ICC3']:2.2f} [{m['ICC_CI95_lower']:2.2f},{m['ICC_CI95_upper']:2.2f}]  "

    # ── optional metrics overlay ─────────────────────────────
    if calculate_and_add_metrics:

        m = calculate_some_classification_metrics(
            all_labels=np.array(x, dtype=np.float64) ,
            all_preds=np.array(y, dtype=np.float64),
            calc_ICC3=2,
            add_classification_metrics=False,
            add_spearman=True,
        )
        rmse = m["rmse"]
        spearman_corr = m["spearman_corr"]
        ICC3 = m["ICC3"]
        ICC_CI95_lower = m["ICC_CI95_lower"]
        ICC_CI95_upper = m["ICC_CI95_upper"]
        ICC_n = m["ICC_n"]
        txt = (
            fr"$\mathrm{{RMSE}}= {rmse:2.2f}\;$"
            "\n"
            fr"$\rho= {spearman_corr:2.2f}\;$"
            "\n"
            fr"$\mathrm{{ICC}}= {ICC3:2.2f}$"
            " "
            fr"$ [{ICC_CI95_lower:2.2f}, {ICC_CI95_upper:2.2f}]$"
            "\n"
            fr"$\mathrm{{N}}= {ICC_n}$"
        )

        ax_scatter.text(
            *metrics_loc, txt,
            transform=ax_scatter.transAxes,
            ha="left", va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=3)
        )
            

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

def plot_SHS_sums(df_summed, *, name: str = "ERO F", **kwargs):
    """Joint plot for df_summed."""
    return joint_hist_scatter(
        df_summed,
        true_col="labels_summed_extrapolated",
        pred_col="preds_summed_extrapolated",
        name=name,
        **kwargs
    )
