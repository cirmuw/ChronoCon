import torch
import mlflow
import torch
import torch.nn as nn

from ra_utils.data.dataloader_CR_patches import (
    process_several_score_groups,
    dataset_and_loader_several,
    check_duplicates_in_dataloader
)

from ra_utils.training.scores_SHS.scores_SHS_training_lib_AE_v1 import (
    evaluate_and_log_testset_results_AE_v3,
    train_loop_AE_v3
)
from ra_utils.training.scores_SHS.scores_SHS_training_lib_AE_v4 import (
    evaluate_and_log_testset_results_AE_v4,
    train_loop_AE_v4
)

from ra_utils.networks.loss_function import get_score_loss_function, get_triplet_loss_fn
import torchvision.transforms.v2 as v2
from ra_utils.training.scores_SHS.model_builders import build_models_AE_v1_and2
import ra_utils.utils.utils_torch
from ra_utils.utils.verbosity_enums import *
import ra_utils.utils.utils

import ra_utils.utils.utils_torch
from pprint import pprint
import ra_utils.utils.config_parser

from ra_utils.utils.utils import datestr_to_years_since_2000


import numpy as np


from ra_utils.training.scores_SHS.run_training_main_lib import (
    check_config_consistency_and_partially_make_consistent,
    maybe_partially_init_model_from_state_dict,
)

import ra_utils.training.scores_SHS.run_training_main_lib

import ra_utils.loss.loss_fn_dict


import ra_utils.loss.loss_RnC_with_ranks_and_ids
import ra_utils.loss.loss_RnC

import ra_utils.loss.loss_RnCMono

import ra_utils.loss.online_mining_triplet_loss_MDP


import ra_utils.visualization.interactive.ra_model

from pathlib import Path

import ra_utils.visualization
import  ra_utils.visualization.trajectories

import pandas as pd
from collections import Counter

import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.cluster.hierarchy import linkage, leaves_list
import numpy as np
import matplotlib.pyplot as plt

from typing import List


def plot_similarity_matrix(npz_data, ROI="PIPIII", sim="negative_l2", axis=None,
                           intra_class_sort="cluster"):
    """
    Plot similarity matrix for one ROI. Within each score class, samples are ordered by similarity.

    intra_class_sort: "cluster" (hierarchical) | "mean" (similarity to class mean) | None
    """
    all_embeddings = npz_data["Z"]
    score_types = npz_data["score_type"]
    m = score_types == ROI
    scores_all = npz_data["score_gt"][m]
    emb_subset = all_embeddings[m]

    # ---- sort by score class first
    sorted_indices = np.argsort(scores_all)
    sorted_scores = scores_all[sorted_indices]
    emb_sorted = emb_subset[sorted_indices]

    # ---- optionally refine order within each class
    final_order = []
    unique_scores = np.unique(sorted_scores)

    for cls in unique_scores:
        idxs = np.where(sorted_scores == cls)[0]
        emb_cls = emb_sorted[idxs]

        if intra_class_sort == "cluster" and len(idxs) > 2:
            # compute small within-class sim and hierarchical linkage
            if sim == "cosine":
                sim_block = cosine_similarity(emb_cls)
                dist_block = 1 - sim_block  # convert to distance
            else:
                dist_block = euclidean_distances(emb_cls)

            # hierarchical clustering for 1D leaf order
            Z = linkage(dist_block, method="average")
            order_within = leaves_list(Z)
        elif intra_class_sort == "mean" and len(idxs) > 1:
            # sort by similarity to class mean
            cls_mean = emb_cls.mean(axis=0, keepdims=True)
            if sim == "cosine":
                sim_to_mean = cosine_similarity(emb_cls, cls_mean).ravel()
            else:
                sim_to_mean = -euclidean_distances(emb_cls, cls_mean).ravel()
            order_within = np.argsort(-sim_to_mean)  # most similar first
        else:
            order_within = np.arange(len(idxs))

        final_order.append(idxs[order_within])

    # Concatenate per-class orders
    final_order = np.concatenate(final_order)
    emb_final = emb_sorted[final_order]
    scores_final = sorted_scores[final_order]

    # ---- Compute full similarity matrix
    if sim == "cosine":
        sim_matrix = cosine_similarity(emb_final)
        sim_label = "Cosine Similarity"
    elif sim == "negative_l2":
        sim_matrix = -euclidean_distances(emb_final)
        sim_label = "Negative L2 Similarity"
    else:
        raise ValueError("sim must be 'negative_l2' or 'cosine'")

    # ---- Determine class boundaries for ticks
    score_changes = scores_final[1:] != scores_final[:-1]
    bounds = np.where(score_changes)[0] + 1
    bounds_with_origin = np.concatenate(([0], bounds))
    class_labels = scores_final[bounds_with_origin]

    # ---- Plot
    if axis is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        ax = axis
        fig = ax.get_figure()

    im = ax.imshow(sim_matrix, cmap="viridis")
    ax.set_xticks(bounds_with_origin)
    ax.set_xticklabels([f"{int(c)}" for c in class_labels], rotation=90)
    ax.set_yticks(bounds_with_origin)
    ax.set_yticklabels([f"{int(c)}" for c in class_labels])
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.12)
    ax.set_title(f"{sim_label} ({ROI})")
    fig.tight_layout(rect=[0, 0, 0.80, 1])

    return fig, ax, im, cbar

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np

from typing import Literal

def compute_similarity_data(npz_data, metric_name="negative_l2", 
                            sort_by: Literal["score", "date"] = "score",
                            sort_order: Literal["ascending", "descending"] = "ascending"):

    assert metric_name in ["cosine", "negative_l2"]

    all_embeddings = npz_data["Z"]
    score_types = sorted(list(set(npz_data["score_type"])))

    data = {}
    for score_type in score_types:
        row = {}
        row["score_type"] = score_type
        m = npz_data["score_type"] == score_type
        scores_all = npz_data["score_gt"][m]
        patient_scoretype_keys = npz_data["patient_scoretype_key"][m]
        scores_max = scores_all.max()
        if sort_by == "score":
            sorted_indices = np.argsort(scores_all)
            if sort_order == "descending":
                sorted_indices = sorted_indices[::-1]
        elif sort_by == "date":
            sorted_indices = np.argsort(npz_data["years_since_2000"][m])
            if sort_order == "descending":
                sorted_indices = sorted_indices[::-1]
        else:
            raise ValueError(f"Invalid sort_by: {sort_by}")
        emb_subset = all_embeddings[m][sorted_indices]

        if metric_name == "cosine":
            sim = cosine_similarity(emb_subset)
        elif metric_name == "negative_l2":
            l2_dist_matrix = euclidean_distances(emb_subset)
            sim = -l2_dist_matrix
        row["sim"] = sim

        row["scores_max"] = scores_max
        row["scores_min"] = scores_all.min()


        row["scores"] = scores_all[sorted_indices]
        row["patient_scoretype_keys"] = patient_scoretype_keys[sorted_indices]
        row["years_since_2000"] = npz_data["years_since_2000"][m][sorted_indices]
        row["date_str"] = npz_data["date_str"][m][sorted_indices]
        data[score_type] = row

    return data



def _compute_similarity_boxplot_data(scores_all, patient_scoretype_keys, sim):
    """
    Helper function to compute boxplot data for similarity vs score difference.
    Returns (bp_data, bp_positions, data, dmin, dmax, edges, x_ticks)
    """
    # Compute integer score differences robustly
    # Use rint to avoid off-by-one from float subtraction/astype(int) truncation
    Delta_ij = np.rint(scores_all[:, None] - scores_all[None, :]).astype(int)

    # Same-patient+scoretype pairs, exclude diagonal
    id_matches = patient_scoretype_keys[:, None] == patient_scoretype_keys[None, :]
    mask = id_matches & ~np.eye(len(scores_all), dtype=bool)

    # Only take one triangle to avoid counting both (i,j) and (j,i)
    # use upper triangle (excluding diagonal)
    tri = np.triu(np.ones_like(Delta_ij, dtype=bool), k=1)

    # Histogram prep (only masked, off-diagonal entries, upper triangle)
    data = Delta_ij[mask & tri]
    dmin = int(data.min())
    dmax = int(data.max())

    # Bin edges exactly align to integer-centered bars, no extra bin
    # Bins: [k-0.5, k+0.5] for integer k in [dmin, dmax]
    edges = np.arange(dmin - 0.5, dmax + 0.5 + 1e-9, 1.0)
    x_ticks = np.arange(dmin, dmax + 1, 1)

    # --- Boxplot of Similarity vs Difference in Scores ---
    bp_data = []
    bp_positions = []

    for k in range(dmin, dmax + 1):
        mask_k = (Delta_ij == k) & mask & tri 

        sim_k = sim[mask_k]
        if sim_k.size > 0:
            bp_data.append(sim_k)
        else:
            # Use NaN placeholder to keep positions aligned without drawing a box
            bp_data.append([np.nan])
        bp_positions.append(k)

    return bp_data, bp_positions, data, dmin, dmax, edges, x_ticks


def _plot_boxplot_and_hist(ax_top, ax_bottom, bp_data, bp_positions, data, dmin, dmax, edges, x_ticks,
                           ROI, plot_hist=True, hist_y_log=True, fontsize_top=None, fontsize_bottom=None):
    """
    Helper function to plot boxplot and histogram on given axes.
    Reused by both plot_single_roi and plot_all_rois_similarity_boxplots.
    """
    # Default font sizes
    if fontsize_top is None:
        fontsize_top = 13
    if fontsize_bottom is None:
        fontsize_bottom = 13

    # --- Boxplot of Similarity vs Difference in Scores ---
    ax_top.boxplot(
        bp_data,
        positions=bp_positions,
        widths=0.7,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,  # we'll set ticks ourselves on the shared axis
    )
    ax_top.set_xlim(dmin - 0.5, dmax + 0.5)
    if plot_hist:
        ax_top.set_xticks([])  # ticks only on histogram axis
    else:
        ax_top.set_xticks(x_ticks)
    ax_top.set_ylabel("Similarity", fontsize=fontsize_top)
    ax_top.set_title(f"{ROI}", fontsize=fontsize_top)
    ax_top.grid(True)
    if fontsize_top < 10:
        ax_top.tick_params(labelsize=fontsize_top-1)

    # --- Histogram of Δ_ij (off-diagonal, same id) ---
    if plot_hist:
        ax_bottom.hist(data, bins=edges, edgecolor="black", color="lightblue")
        ax_bottom.set_xlim(dmin - 0.5, dmax + 0.5)
        ax_bottom.set_xticks(x_ticks)
        ax_bottom.set_xlabel(r'Score difference (intra patient; time-ordered) $\Delta_{ij}$', fontsize=fontsize_bottom)
        ax_bottom.set_ylabel("Frequency", fontsize=fontsize_bottom)
        ax_bottom.tick_params(axis='both', direction='in', which='both', top=True, right=True, labelsize=fontsize_bottom)
        if hist_y_log:
            ax_bottom.set_yscale("log")
    else:
        ax_top.set_xticks(x_ticks)
        ax_top.set_xlabel(r'Score difference (intra patient; time-ordered) $\Delta_{ij}$', fontsize=fontsize_top)


def plot_single_roi(ROI, data_similarities, plot_hist=True, hist_y_log=True):
    import numpy as np
    import matplotlib.pyplot as plt

    scores_all = data_similarities[ROI]["scores"]
    patient_scoretype_keys = data_similarities[ROI]["patient_scoretype_keys"]
    sim = data_similarities[ROI]["sim"]

    # Compute boxplot data using shared helper
    bp_data, bp_positions, data, dmin, dmax, edges, x_ticks = _compute_similarity_boxplot_data(
        scores_all, patient_scoretype_keys, sim
    )

    if plot_hist:
        from matplotlib import gridspec
        fig = plt.figure(figsize=(7, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.25)
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1], sharex=ax0)
        axs = [ax0, ax1]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        axs = [ax]

    # Plot using shared helper
    if plot_hist:
        _plot_boxplot_and_hist(axs[0], axs[1], bp_data, bp_positions, data, dmin, dmax, edges, x_ticks,
                              ROI, plot_hist=True, hist_y_log=hist_y_log)
    else:
        _plot_boxplot_and_hist(axs[0], None, bp_data, bp_positions, data, dmin, dmax, edges, x_ticks,
                              ROI, plot_hist=False, hist_y_log=hist_y_log)
    
    # Set title with original format for single ROI plot
    axs[0].set_title(f"{ROI = }")

    return fig, axs


def _compute_boxplot_data_all_models(ROI, data_similarities_per_model):
    """
    Helper function to compute boxplot data for all models and find unified delta range.
    
    Parameters:
    -----------
    ROI : str
        The ROI name to extract data for
    data_similarities_per_model : dict
        Dictionary with model names as keys and data_similarities dicts as values
        (same structure as used in plot_single_roi)
    
    Returns:
    --------
    model_data : dict
        Dictionary with model names as keys and tuples (bp_data, bp_positions, data, dmin, dmax, edges, x_ticks) as values
    unified_dmin : int
        Minimum delta value across all models
    unified_dmax : int
        Maximum delta value across all models
    unified_x_ticks : ndarray
        Unified x-axis ticks covering all delta values
    """
    model_data = {}
    all_dmins = []
    all_dmaxs = []
    
    # Compute boxplot data for each model
    for model_name, data_similarities in data_similarities_per_model.items():
        scores_all = data_similarities[ROI]["scores"]
        patient_scoretype_keys = data_similarities[ROI]["patient_scoretype_keys"]
        sim = data_similarities[ROI]["sim"]
        
        bp_data, bp_positions, data, dmin, dmax, edges, x_ticks = _compute_similarity_boxplot_data(
            scores_all, patient_scoretype_keys, sim
        )
        
        model_data[model_name] = (bp_data, bp_positions, data, dmin, dmax, edges, x_ticks)
        all_dmins.append(dmin)
        all_dmaxs.append(dmax)
    
    # Find unified range
    unified_dmin = min(all_dmins)
    unified_dmax = max(all_dmaxs)
    unified_x_ticks = np.arange(unified_dmin, unified_dmax + 1, 1)
    
    return model_data, unified_dmin, unified_dmax, unified_x_ticks


def _plot_multiple_boxplots(ax, model_data, unified_dmin, unified_dmax, unified_x_ticks, 
                            model_names, colors=None, show_x_ticks=True, show_legend=True, 
                            legend_fontsize=10, ylabel_fontsize=13, legend_ncol=1, tick_fontsize=None):
    """
    Helper function to plot multiple boxplots side-by-side for each delta value.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    model_data : dict
        Dictionary with model names as keys and tuples (bp_data, bp_positions, data, dmin, dmax, edges, x_ticks) as values
    unified_dmin : int
        Minimum delta value across all models
    unified_dmax : int
        Maximum delta value across all models
    unified_x_ticks : ndarray
        Unified x-axis ticks
    model_names : list
        List of model names in the order to plot
    colors : list, optional
        List of colors for each model. If None, uses matplotlib default color cycle
    show_x_ticks : bool, default=True
        Whether to show x-axis ticks on this axis
    show_legend : bool, default=True
        Whether to show the legend
    legend_fontsize : int, default=10
        Font size for the legend
    ylabel_fontsize : int, default=13
        Font size for the y-axis label
    legend_ncol : int, default=1
        Number of columns in the legend
    tick_fontsize : int, optional
        Font size for tick labels. If None, uses matplotlib default.
    """
    n_models = len(model_names)
    
    # Set up colors
    if colors is None:
        # Use matplotlib's default color cycle
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = [prop_cycle.by_key()['color'][i % len(prop_cycle.by_key()['color'])] 
                 for i in range(n_models)]
    
    # Calculate box width and spacing
    # Total width for all boxes at one delta position should be reasonable (max 0.8)
    max_total_width = 0.8
    box_width = max_total_width / (n_models + 0.5)  # Slightly smaller to allow spacing
    spacing = box_width * 0.15  # Spacing between boxes
    
    # Plot boxplots for each delta value
    for delta_val in range(unified_dmin, unified_dmax + 1):
        # Collect data for all models at this delta value
        model_data_at_delta = []
        model_indices_at_delta = []
        
        for model_idx, model_name in enumerate(model_names):
            bp_data, bp_positions, _, dmin, dmax, _, _ = model_data[model_name]
            
            # Find the index of this delta value in the model's data
            if delta_val < dmin or delta_val > dmax:
                # This model doesn't have data for this delta value
                continue
            
            # Find the position in bp_data that corresponds to delta_val
            delta_idx = delta_val - dmin
            if delta_idx < len(bp_data):
                sim_data = bp_data[delta_idx]
                
                # Filter out NaN values
                if isinstance(sim_data, np.ndarray) and sim_data.size > 0:
                    sim_data = sim_data[~np.isnan(sim_data)]
                
                if len(sim_data) > 0:
                    model_data_at_delta.append((sim_data, model_idx))
                    model_indices_at_delta.append(model_idx)
        
        # If we have data for this delta value, plot all models side-by-side
        if model_data_at_delta:
            n_boxes = len(model_data_at_delta)
            total_width = n_boxes * box_width + (n_boxes - 1) * spacing
            start_pos = delta_val - total_width / 2 + box_width / 2
            
            for box_idx, (sim_data, model_idx) in enumerate(model_data_at_delta):
                # Calculate position for this boxplot
                pos = start_pos + box_idx * (box_width + spacing)
                
                # Create a single boxplot at this position
                bp = ax.boxplot(
                    [sim_data],
                    positions=[pos],
                    widths=box_width,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                )
                
                # Set color for this boxplot
                for patch in bp['boxes']:
                    patch.set_facecolor(colors[model_idx])
                    patch.set_alpha(0.7)
                
                # Set color for median line and other elements
                for median in bp['medians']:
                    median.set_color(colors[model_idx])
                    median.set_linewidth(1.5)
                
                # Set color for whiskers and caps
                for element in ['whiskers', 'caps']:
                    for item in bp[element]:
                        item.set_color(colors[model_idx])
                        item.set_linewidth(1.2)
    
    # Set axis properties
    ax.set_xlim(unified_dmin - 0.5, unified_dmax + 0.5)
    if show_x_ticks:
        ax.set_xticks(unified_x_ticks)
    else:
        ax.set_xticks([])  # Hide ticks when histogram is shown below
    ax.set_ylabel("Similarity", fontsize=ylabel_fontsize)
    if tick_fontsize is not None:
        ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.grid(True, alpha=0.3)
    
    # Create legend if requested
    if show_legend:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[i], alpha=0.7, label=model_name) 
                          for i, model_name in enumerate(model_names)]
        ax.legend(handles=legend_elements, loc='best', fontsize=legend_fontsize, ncol=legend_ncol)


def plot_single_roi_several_models(ROI, data_similarities_per_model, plot_hist=True, 
                                    hist_y_log=True, colors=None, hist_model=None, hide_hist_legend=True,
                                    show_legend=True):
    """
    Plot similarity boxplots for a single ROI comparing multiple models side-by-side.
    
    Parameters:
    -----------
    ROI : str
        The ROI name to plot
    data_similarities_per_model : dict
        Dictionary with model names as keys and data_similarities dicts as values.
        Each data_similarities dict should have the same structure as used in plot_single_roi,
        i.e., it should contain a key matching ROI with sub-dict containing:
        - "scores": array of scores
        - "patient_scoretype_keys": array of patient identifiers
        - "sim": similarity matrix
    plot_hist : bool, default=True
        Whether to plot the histogram below the boxplots
    hist_y_log : bool, default=True
        Whether to use log scale for histogram y-axis
    colors : list, optional
        List of colors for each model. If None, uses matplotlib default color cycle.
        Should have same length as number of models.
    hist_model : str, optional
        Model name to use for histogram data. If None, uses the first model.
    show_legend : bool, default=True
        Whether to show the legend on the plot.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    axs : list
        List of axes objects [ax_boxplot, ax_histogram] (or [ax_boxplot] if plot_hist=False)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    
    # Validate input
    if not data_similarities_per_model:
        raise ValueError("data_similarities_per_model cannot be empty")
    
    model_names = list(data_similarities_per_model.keys())
    
    # Validate ROI exists in all models
    for model_name in model_names:
        if ROI not in data_similarities_per_model[model_name]:
            raise ValueError(f"ROI '{ROI}' not found in model '{model_name}' data")
    
    # Compute boxplot data for all models
    model_data, unified_dmin, unified_dmax, unified_x_ticks = _compute_boxplot_data_all_models(
        ROI, data_similarities_per_model
    )
    
    # Set up figure and axes
    if plot_hist:
        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.25)
        ax_boxplot = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1], sharex=ax_boxplot)
        axs = [ax_boxplot, ax_hist]
    else:
        fig, ax_boxplot = plt.subplots(1, 1, figsize=(10, 4))
        axs = [ax_boxplot]
    
    # Plot multiple boxplots
    _plot_multiple_boxplots(ax_boxplot, model_data, unified_dmin, unified_dmax, unified_x_ticks,
                           model_names, colors=colors, show_x_ticks=not plot_hist, show_legend=show_legend)
    
    # Set title
    ax_boxplot.set_title(f"{ROI = }", fontsize=13)
    
    # Plot histogram if requested
    if plot_hist:
        # Determine which model to use for histogram
        if hist_model is None:
            hist_model = model_names[0]
        elif hist_model not in model_names:
            raise ValueError(f"hist_model '{hist_model}' not found in model names")
        
        # Get histogram data from selected model
        _, _, data, dmin, dmax, edges, _ = model_data[hist_model]
        
        # Plot histogram
        ax_hist.hist(data, bins=edges, edgecolor="black", color="lightblue", alpha=0.7)
        ax_hist.set_xlim(unified_dmin - 0.5, unified_dmax + 0.5)
        ax_hist.set_xticks(unified_x_ticks)
        ax_hist.set_xlabel(r'Score difference (intra patient; time-ordered) $\Delta_{ij}$', fontsize=13)
        ax_hist.set_ylabel("Frequency", fontsize=13)
        ax_hist.tick_params(axis='both', direction='in', which='both', top=True, right=True)
        if hist_y_log:
            ax_hist.set_yscale("log")
        
        # Add note about which model's histogram is shown
        if len(model_names) > 1 and not hide_hist_legend:
            ax_hist.text(0.98, 0.98, f"Histogram: {hist_model}", 
                        transform=ax_hist.transAxes, 
                        fontsize=9, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax_boxplot.set_xticks(unified_x_ticks)
        ax_boxplot.set_xlabel(r'Score difference (intra patient; time-ordered) $\Delta_{ij}$', fontsize=13)
    
    plt.tight_layout()
    
    return fig, axs





def plot_single_roi_several_models_LARGER_FONTS(ROI, data_similarities_per_model, plot_hist=True, 
                                    hist_y_log=True, colors=None, hist_model=None, hide_hist_legend=True,
                                    show_legend=True, title_fontsize=13, legend_fontsize=10, 
                                    ylabel_fontsize=13, xlabel_fontsize=13, ylabel_fontsize_hist=13,
                                    tick_fontsize=None):
    """
    Plot similarity boxplots for a single ROI comparing multiple models side-by-side.
    
    Parameters:
    -----------
    ROI : str
        The ROI name to plot
    data_similarities_per_model : dict
        Dictionary with model names as keys and data_similarities dicts as values.
        Each data_similarities dict should have the same structure as used in plot_single_roi,
        i.e., it should contain a key matching ROI with sub-dict containing:
        - "scores": array of scores
        - "patient_scoretype_keys": array of patient identifiers
        - "sim": similarity matrix
    plot_hist : bool, default=True
        Whether to plot the histogram below the boxplots
    hist_y_log : bool, default=True
        Whether to use log scale for histogram y-axis
    colors : list, optional
        List of colors for each model. If None, uses matplotlib default color cycle.
        Should have same length as number of models.
    hist_model : str, optional
        Model name to use for histogram data. If None, uses the first model.
    show_legend : bool, default=True
        Whether to show the legend on the plot.
    title_fontsize : int, default=13
        Font size for the title
    legend_fontsize : int, default=10
        Font size for the legend
    ylabel_fontsize : int, default=13
        Font size for the y-axis label of the boxplot
    xlabel_fontsize : int, default=13
        Font size for the x-axis label of the histogram
    ylabel_fontsize_hist : int, default=13
        Font size for the y-axis label of the histogram
    tick_fontsize : int, optional
        Font size for tick labels. If None, uses matplotlib default.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    axs : list
        List of axes objects [ax_boxplot, ax_histogram] (or [ax_boxplot] if plot_hist=False)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    
    # Validate input
    if not data_similarities_per_model:
        raise ValueError("data_similarities_per_model cannot be empty")
    
    model_names = list(data_similarities_per_model.keys())
    
    # Validate ROI exists in all models
    for model_name in model_names:
        if ROI not in data_similarities_per_model[model_name]:
            raise ValueError(f"ROI '{ROI}' not found in model '{model_name}' data")
    
    # Compute boxplot data for all models
    model_data, unified_dmin, unified_dmax, unified_x_ticks = _compute_boxplot_data_all_models(
        ROI, data_similarities_per_model
    )
    
    # Set up figure and axes
    if plot_hist:
        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0)
        ax_boxplot = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1], sharex=ax_boxplot)
        axs = [ax_boxplot, ax_hist]
    else:
        fig, ax_boxplot = plt.subplots(1, 1, figsize=(10, 4))
        axs = [ax_boxplot]
    
    # Plot multiple boxplots
    _plot_multiple_boxplots(ax_boxplot, model_data, unified_dmin, unified_dmax, unified_x_ticks,
                           model_names, colors=colors, show_x_ticks=not plot_hist, show_legend=show_legend,
                           legend_ncol=2, legend_fontsize=legend_fontsize, ylabel_fontsize=ylabel_fontsize,
                           tick_fontsize=tick_fontsize)
    
    # Set title
    ax_boxplot.set_title(f"{ROI = }", fontsize=title_fontsize)
    
    # Remove xticklabels from top plot when histogram is shown (they share the same axis)
    if plot_hist:
        ax_boxplot.tick_params(axis='x', labelbottom=False)  # Hide x-axis labels on top plot
    
    # Plot histogram if requested
    if plot_hist:
        # Determine which model to use for histogram
        if hist_model is None:
            hist_model = model_names[0]
        elif hist_model not in model_names:
            raise ValueError(f"hist_model '{hist_model}' not found in model names")
        
        # Get histogram data from selected model
        _, _, data, dmin, dmax, edges, _ = model_data[hist_model]
        
        # Plot histogram
        ax_hist.hist(data, bins=edges, edgecolor="black", color="lightblue", alpha=0.7)
        ax_hist.set_xlim(unified_dmin - 0.5, unified_dmax + 0.5)
        ax_hist.set_xticks(unified_x_ticks)
        # Explicitly show x-axis labels on bottom plot
        ax_hist.tick_params(axis='x', labelbottom=True)  # Ensure x-axis labels are visible on bottom plot
        ax_hist.set_xlabel(r'Score difference (intra patient; time-ordered) $\Delta_{ij}$', fontsize=xlabel_fontsize)
        ax_hist.set_ylabel("Frequency", fontsize=ylabel_fontsize_hist)
        ax_hist.tick_params(axis='both', direction='in', which='both', top=True, right=True)
        if tick_fontsize is not None:
            ax_hist.tick_params(axis='both', labelsize=tick_fontsize)
        if hist_y_log:
            ax_hist.set_yscale("log")
        
        # Add note about which model's histogram is shown
        if len(model_names) > 1 and not hide_hist_legend:
            ax_hist.text(0.98, 0.98, f"Histogram: {hist_model}", 
                        transform=ax_hist.transAxes, 
                        fontsize=9, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax_boxplot.set_xticks(unified_x_ticks)
        ax_boxplot.set_xlabel(r'Score difference (intra patient; time-ordered) $\Delta_{ij}$', fontsize=xlabel_fontsize)
        if tick_fontsize is not None:
            ax_boxplot.tick_params(axis='both', labelsize=tick_fontsize)
    
    plt.tight_layout()
    
    return fig, axs



### Plot for all joints (59 score types)
import math

def plot_all_rois_similarity_boxplots(data_similarities, plot_hist=True, axis=None, hist_y_log=True):
    """
    Plot similarity boxplots and histograms for all ROIs in data_similarities.
    If axis is None, creates a new figure and axes grid.
    Uses gridspec with height ratios (3:1) to make histograms smaller, similar to plot_single_roi.
    Returns (fig, axes), where axes is a 2D numpy array of subplot Axes objects, shape (2*n_rows, n_cols).
    """
    from matplotlib import gridspec
    
    all_rois = sorted(list(data_similarities.keys()))
    n_rois = len(all_rois)

    # Grid: 8x8 covers up to 64 ROIs
    n_cols = 8
    n_rows = math.ceil(n_rois / n_cols)

    # Prepare axes
    if axis is None:
        # Use gridspec to control height ratios (3:1 for boxplot:histogram)
        # Each ROI gets 2 rows in the grid, but we'll use gridspec within each cell
        fig = plt.figure(figsize=(3*n_cols, 2.5*2*n_rows))
        # Create a master gridspec for the overall layout
        # We'll create nested gridspecs for each ROI to control height ratios
        gs_master = gridspec.GridSpec(2*n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)
        created_fig = True
    else:
        # User provides their own axes, not a figure
        fig = None
        gs_master = None
        created_fig = False

    # Prepare array of axes handles
    axes = np.empty((2*n_rows, n_cols), dtype=object)

    for idx, ROI in enumerate(all_rois):
        row = idx // n_cols
        col = idx % n_cols

        scores_all = data_similarities[ROI]["scores"]
        patient_scoretype_keys = data_similarities[ROI]["patient_scoretype_keys"]
        sim = data_similarities[ROI]["sim"]

        # Use shared helper to compute boxplot data
        try:
            bp_data, bp_positions, data, dmin, dmax, edges, x_ticks = _compute_similarity_boxplot_data(
                scores_all, patient_scoretype_keys, sim
            )
        except (ValueError, IndexError):
            # Empty data case
            axes[2*row, col] = None
            axes[2*row+1, col] = None
            continue

        if len(data) == 0:
            # Leave empty plots
            axes[2*row, col] = None
            axes[2*row+1, col] = None
            continue

        # Create nested gridspec for this ROI to control height ratio (3:1)
        if axis is None:
            # Create a nested gridspec within the master grid for this ROI
            # This allows us to control the height ratio between boxplot and histogram
            # Increased hspace to prevent overlap between boxplot and histogram
            gs_roi = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=gs_master[2*row:2*row+2, col],
                height_ratios=[3, 1], hspace=0.25
            )
            ax_top = fig.add_subplot(gs_roi[0])
            if plot_hist:
                ax_bottom = fig.add_subplot(gs_roi[1], sharex=ax_top)
            else:
                ax_bottom = None
        else:
            ax_top = axis[2*row, col]
            if plot_hist:
                ax_bottom = axis[2*row+1, col]
            else:
                ax_bottom = None
        
        axes[2*row, col] = ax_top
        if plot_hist:
            axes[2*row+1, col] = ax_bottom
        else:
            axes[2*row+1, col] = None

        # Plot using shared helper function
        if ax_top is not None:
            _plot_boxplot_and_hist(
                ax_top, ax_bottom, bp_data, bp_positions, data, dmin, dmax, edges, x_ticks,
                ROI, plot_hist=plot_hist, hist_y_log=hist_y_log,
                fontsize_top=9, fontsize_bottom=8
            )
            # Additional styling for the multi-plot layout (override grid alpha)
            ax_top.grid(True, alpha=0.3)
            # Add padding to title to prevent overlap with histogram xlabel
            ax_top.set_title(f"{ROI}", fontsize=9, pad=4)
            if plot_hist and ax_bottom is not None:
                # Override xlabel and ylabel for more compact multi-plot layout
                ax_bottom.set_xlabel(r'$\Delta_{ij}$', fontsize=8, labelpad=3)
                ax_bottom.set_ylabel("Freq", fontsize=8)
                ax_bottom.xaxis.set_major_locator(plt.MultipleLocator(1))
                # Adjust subplot parameters to add space for xlabel
                ax_bottom.tick_params(axis='x', pad=3)  # Add padding to x-axis ticks

    if axis is None:
        # Use tight_layout with padding to prevent overlap of xlabels and titles
        plt.tight_layout(rect=[0, 0.02, 1, 0.98], pad=1.5)
        plt.show()
        return fig, axes
    else:
        return None, axes


def plot_all_rois_similarity_boxplots_several_models(data_similarities_per_model, plot_hist=True, 
                                                      hist_y_log=True, colors=None, hist_model=None, axis=None, 
                                                      n_cols = 8, 
                                                      legend_y_position = 0.02
                                                      ):
    """
    Plot similarity boxplots and histograms for all ROIs comparing multiple models side-by-side.
    Uses the same logic as plot_single_roi_several_models but for all ROIs in a grid layout.
    
    Parameters:
    -----------
    data_similarities_per_model : dict
        Dictionary with model names as keys and data_similarities dicts as values.
        Each data_similarities dict should have the same structure as used in plot_single_roi,
        i.e., it should contain ROI keys with sub-dicts containing:
        - "scores": array of scores
        - "patient_scoretype_keys": array of patient identifiers
        - "sim": similarity matrix
    plot_hist : bool, default=True
        Whether to plot the histogram below the boxplots
    hist_y_log : bool, default=True
        Whether to use log scale for histogram y-axis
    colors : list, optional
        List of colors for each model. If None, uses matplotlib default color cycle.
        Should have same length as number of models.
    hist_model : str, optional
        Model name to use for histogram data. If None, uses the first model.
    axis : array-like, optional
        If provided, should be a 2D array of axes with shape (2*n_rows, n_cols).
        If None, creates a new figure.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure or None
        The figure object (None if axis is provided)
    axes : numpy.ndarray
        2D numpy array of subplot Axes objects, shape (2*n_rows, n_cols)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import math
    
    # Validate input
    if not data_similarities_per_model:
        raise ValueError("data_similarities_per_model cannot be empty")
    
    model_names = list(data_similarities_per_model.keys())
    
    # Get all ROIs from the first model (assuming all models have the same ROIs)
    first_model_name = model_names[0]
    all_rois = sorted(list(data_similarities_per_model[first_model_name].keys()))
    n_rois = len(all_rois)
    
    # Validate ROI exists in all models
    for model_name in model_names:
        model_rois = set(data_similarities_per_model[model_name].keys())
        missing_rois = set(all_rois) - model_rois
        if missing_rois:
            raise ValueError(f"ROIs {missing_rois} not found in model '{model_name}' data")
    
    # Grid: 8x8 covers up to 64 ROIs

    n_rows = math.ceil(n_rois / n_cols)
    
    # Determine which model to use for histogram
    if hist_model is None:
        hist_model = model_names[0]
    elif hist_model not in model_names:
        raise ValueError(f"hist_model '{hist_model}' not found in model names")
    
    # Set up colors
    if colors is None:
        # Use matplotlib's default color cycle
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = [prop_cycle.by_key()['color'][i % len(prop_cycle.by_key()['color'])] 
                 for i in range(len(model_names))]
    
    # Prepare axes
    if axis is None:
        # Use gridspec to control height ratios (3:1 for boxplot:histogram)
        # Each ROI gets 2 rows in the grid, but we'll use gridspec within each cell
        fig = plt.figure(figsize=(3*n_cols, 2.5*2*n_rows))
        # Create a master gridspec for the overall layout
        # We'll create nested gridspecs for each ROI to control height ratios
        gs_master = gridspec.GridSpec(2*n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)
    else:
        # User provides their own axes, not a figure
        fig = None
        gs_master = None
    
    # Prepare array of axes handles
    axes = np.empty((2*n_rows, n_cols), dtype=object)
    
    # Process each ROI
    for idx, ROI in enumerate(all_rois):
        row = idx // n_cols
        col = idx % n_cols
        
        # Compute boxplot data for all models for this ROI
        try:
            model_data, unified_dmin, unified_dmax, unified_x_ticks = _compute_boxplot_data_all_models(
                ROI, data_similarities_per_model
            )
        except (ValueError, IndexError) as e:
            # Empty data case
            axes[2*row, col] = None
            axes[2*row+1, col] = None
            continue
        
        # Check if we have any data
        has_data = False
        for model_name in model_names:
            _, _, data, _, _, _, _ = model_data[model_name]
            if len(data) > 0:
                has_data = True
                break
        
        if not has_data:
            axes[2*row, col] = None
            axes[2*row+1, col] = None
            continue
        
        # Create nested gridspec for this ROI to control height ratio (3:1)
        if axis is None:
            # Create a nested gridspec within the master grid for this ROI
            # This allows us to control the height ratio between boxplot and histogram
            gs_roi = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=gs_master[2*row:2*row+2, col],
                height_ratios=[3, 1], hspace=0.25
            )
            ax_top = fig.add_subplot(gs_roi[0])
            if plot_hist:
                ax_bottom = fig.add_subplot(gs_roi[1], sharex=ax_top)
            else:
                ax_bottom = None
        else:
            ax_top = axis[2*row, col]
            if plot_hist:
                ax_bottom = axis[2*row+1, col]
            else:
                ax_bottom = None
        
        axes[2*row, col] = ax_top
        if plot_hist:
            axes[2*row+1, col] = ax_bottom
        else:
            axes[2*row+1, col] = None
        
        # Plot multiple boxplots using shared helper
        # Don't show legend in grid layout (too cluttered), user can use single ROI function for legend
        if ax_top is not None:
            _plot_multiple_boxplots(ax_top, model_data, unified_dmin, unified_dmax, unified_x_ticks,
                                   model_names, colors=colors, show_x_ticks=not plot_hist, 
                                   show_legend=False, ylabel_fontsize=8)
            
            # Additional styling for the multi-plot layout
            ax_top.set_title(f"{ROI}", fontsize=9, pad=4)
            ax_top.grid(True, alpha=0.3)
            ax_top.tick_params(labelsize=7)
            
            # Plot histogram if requested
            if plot_hist and ax_bottom is not None:
                # Get histogram data from selected model
                _, _, data, dmin, dmax, edges, _ = model_data[hist_model]
                
                # Plot histogram
                ax_bottom.hist(data, bins=edges, edgecolor='black', color='skyblue', alpha=0.7)
                ax_bottom.set_xlim(unified_dmin - 0.5, unified_dmax + 0.5)
                ax_bottom.set_xticks(unified_x_ticks)
                ax_bottom.set_xlabel(r'$\Delta_{ij}$', fontsize=8, labelpad=3)
                ax_bottom.set_ylabel("Freq", fontsize=8)
                if hist_y_log:
                    ax_bottom.set_yscale("log")
                ax_bottom.tick_params(axis='both', direction='in', which='both', top=True, right=True, labelsize=7)
                ax_bottom.xaxis.set_major_locator(plt.MultipleLocator(1))
                ax_bottom.tick_params(axis='x', pad=3)  # Add padding to x-axis ticks
    
    if axis is None:
        # Add figure-level legend for all models
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[i], alpha=0.7, label=model_name) 
                          for i, model_name in enumerate(model_names)]
        # Place legend at the lower right corner of the figure (where there's likely white space)
        fig.legend(handles=legend_elements, loc='lower right', 
                  bbox_to_anchor=(0.9, legend_y_position), ncol=1, 
                  fontsize=15, frameon=True, fancybox=True, shadow=True)
        
        # Use tight_layout with padding to prevent overlap of xlabels and titles
        plt.tight_layout(rect=[0, 0.02, 1, 0.98], pad=1.5)
        return fig, axes
    else:
        return None, axes




def save_all_rois_as_pdfs(data_similarities_per_model, output_folder, plot_hist=True, 
                           hist_y_log=True, colors=None, hist_model=None):
    """
    Save all ROI subplots as separate PDF files in the specified folder.
    Each ROI is saved as an individual PDF without a legend, suitable for combining later in Inkscape.
    
    Parameters:
    -----------
    data_similarities_per_model : dict
        Dictionary with model names as keys and data_similarities dicts as values.
        Each data_similarities dict should have the same structure as used in plot_single_roi,
        i.e., it should contain ROI keys with sub-dicts containing:
        - "scores": array of scores
        - "patient_scoretype_keys": array of patient identifiers
        - "sim": similarity matrix
    output_folder : str or Path
        Path to the folder where PDF files will be saved. Will be created if it doesn't exist.
    plot_hist : bool, default=True
        Whether to plot the histogram below the boxplots
    hist_y_log : bool, default=True
        Whether to use log scale for histogram y-axis
    colors : list, optional
        List of colors for each model. If None, uses matplotlib default color cycle.
        Should have same length as number of models.
    hist_model : str, optional
        Model name to use for histogram data. If None, uses the first model.
    
    Returns:
    --------
    saved_files : list
        List of paths to the saved PDF files
    """
    from pathlib import Path
    
    # Convert to Path object and create directory if needed
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get all ROIs from the first model (assuming all models have the same ROIs)
    model_names = list(data_similarities_per_model.keys())
    if not model_names:
        raise ValueError("data_similarities_per_model cannot be empty")
    
    first_model_name = model_names[0]
    all_rois = sorted(list(data_similarities_per_model[first_model_name].keys()))
    
    saved_files = []
    
    # Loop through each ROI and save as PDF
    for ROI in all_rois:
        try:
            # Create plot for this ROI without legend
            fig, axs = plot_single_roi_several_models(
                ROI, data_similarities_per_model, 
                plot_hist=plot_hist, 
                hist_y_log=hist_y_log, 
                colors=colors, 
                hist_model=hist_model,
                show_legend=False  # No legend for individual PDFs
            )
            
            # Create filename (sanitize ROI name for filesystem)
            safe_roi_name = ROI.replace('/', '_').replace('\\', '_').replace(' ', '_')
            pdf_path = output_folder / f"{safe_roi_name}.pdf"
            
            # Save as PDF
            fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
            saved_files.append(pdf_path)
            
            # Close figure to free memory
            plt.close(fig)
            
        except Exception as e:
            print(f"Warning: Failed to save plot for ROI '{ROI}': {e}")
            continue
    
    print(f"Saved {len(saved_files)} PDF files to {output_folder}")
    return saved_files 
