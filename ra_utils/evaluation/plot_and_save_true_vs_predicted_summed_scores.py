import os
import sys

from torchvision import transforms
from torch.utils.data import DataLoader
import torch


import os
import numpy as np
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import random
from pathlib import Path 


# from ra_utils.autoscora.autoscorRA_Pipeline.scoring.src.io_scoring_method import io_scoring
# from ra_utils.autoscora.autoscorRA_Pipeline.scoring.src.run_utils import (
#     paths_list_scores_list_from_score_types,
#     restructure_paths_and_scores,
#     restructure_paths_and_scores_v2
# )

import pandas as pd
from typing import List 

import ra_utils
import ra_utils.utils.config_parser
import ra_utils.networks.loss_function


import ra_utils.data.dataloader_CR_patches
from ra_utils.data.dataloader_CR_patches import (
    load_img_SHS_patch_data,
    df_scores_to_dct_list,
)

import ra_utils.data.dataloader_CR_patches
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose
)
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader


from ra_utils.data.dataloader_CR_patches import (
    load_img_SHS_patch_data,
    dataset_and_loader,
    dataset_and_loader_several,
    df_scores_to_dct_list,
    make_paths_dataframe,
    restructure_paths_and_scores,
    restructure_paths_and_scores_v2,
    exclude_ROIS_according_surgery_status,
    split_training_val_test__on_patient_level,
    process_several_score_groups,
    process_single_score_group,
    load_img_SHS_patch_data
)

import yaml
from importlib import resources

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.collections import Collection
from matplotlib.lines import Line2D


import ra_utils.networks.architecture
from ra_utils.networks.architecture import (
    ResNet18Encoder,
    ResNet34Encoder,
    ResNet50Encoder,
    make_mlp,
    EncoderClassifierNetwork,
    MultiModalImageScoreTypeNetwork,
    ROI_type_encoder,
    model_interface_forward
)
import numpy as np
from ra_utils.autoscora.autoscorRA_Pipeline.scoring.src.network import Custom_VGG
from ra_utils.utils.utils_SHS_scoring import get_classes
import ra_utils.utils.utils
import torch.nn as nn


import ra_utils.mtan.im2im_pred.model_resnet_mtan.resnet_recon_mtan

from ra_utils.mtan.im2im_pred.model_resnet_mtan.resnet_recon_mtan import (
    #MTANResNetRecon
    MTANReconCls,
    build_mtan_recon_cls
)


from ra_utils.data.data_utils import (
    extract_extras_from_filename
)


from ra_utils.training.scores_SHS.model_builders import build_models_AE
from tqdm.notebook import tqdm

from ra_utils.progressionlearning.models.builder import (
    build_MTANAE
)
from ra_utils.progressionlearning.models.MTANUNet import (
    MTANRecUnet, 
    MTANRecUnet_v2,
    MTANRecUnet_v3
)
import monai
from monai.networks.nets import BasicUNet, UNet


from ra_utils.progressionlearning.models.builder import (
    build_MTANAE, 
    build_MTANAE_v2
)
from ra_utils.progressionlearning.models.MTANUNet import (
    MTANRecUnet,
    MTANRecUnet_v2
)
import monai
import monai.networks.nets
from monai.networks.nets import BasicUNet, UNet
from typing import Dict, List

import ra_utils.mtan.im2im_pred.model_resnet_mtan.resnet_mtan
from ra_utils.training.scores_SHS.model_builders import build_models_AE_v2, build_models_AE_v1_and2


# wrap model: 

from ra_utils.networks.architecture import (
    MultiModalImageScoreTypeNetworkAE,
    ROI_type_encoder
)



# get all scores types / roi types

import pingouin as pg


from ra_utils.data.shap_sums import (
    add_JSN_ERO_sums, 
    double_scoring_make_merge_id,
    limit_treatment_number
)


from ra_utils.data.icc import compute_icc3

from ra_utils.training.scores_SHS.scores_SHS_training_lib import (
    calculate_some_classification_metrics
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
)


import os
import mlflow
from mlflow.tracking import MlflowClient
import json
import yaml
from sklearn.metrics import balanced_accuracy_score

from pprint import pprint

# CODE: 
from ra_utils.evaluation.single_SHS import (
    combine_predictions, 
    get_main_metrics
)


from ra_utils.data.shap_sums import (
   #max_possible_score, 
    sum_and_extrapolate_scores_df,
    generate_score_differences,
    sum_and_extrapolate_scores_df_ERO_H,
    sum_and_extrapolate_scores_df_ERO_F,
    sum_and_extrapolate_scores_df_JSN_F,
    sum_and_extrapolate_scores_df_JSN_H
)



import ra_utils.visualization.plot_SHS_scores
from ra_utils.visualization.plot_SHS_scores import (
    plot_SHS_deltas, 
    plot_SHS_sums
)
import glob
import os


import argparse

with resources.files("ra_utils.resources.scores_metadata").joinpath("roi_scores_matching.csv") as f:
    df_scores_meta = pd.read_csv(f)
H_ERO_scores = sorted(df_scores_meta[(df_scores_meta["ERO_or_JSN"] == "ERO") & (df_scores_meta["region"] == "H")]["score_name"].unique())
F_ERO_scores = sorted(df_scores_meta[(df_scores_meta["ERO_or_JSN"] == "ERO") & (df_scores_meta["region"] == "F")]["score_name"].unique())
H_JSN_scores = sorted(df_scores_meta[(df_scores_meta["ERO_or_JSN"] == "JSN") & (df_scores_meta["region"] == "H")]["score_name"].unique())
F_JSN_scores = sorted(df_scores_meta[(df_scores_meta["ERO_or_JSN"] == "JSN") & (df_scores_meta["region"] == "F")]["score_name"].unique())



from sklearn.metrics import cohen_kappa_score
from rpy2.robjects import DataFrame, FloatVector, IntVector
from rpy2.robjects.packages import importr
r_icc = importr("ICC")
r_irr = importr("irr")
r_iccp = importr("psych")

print("Done loading")

# /home/cwatzenboeck/code/RA/ra_utils/ra_utils/data/shap_sums.py
import ra_utils.data.shap_sums


def metrics_to_text_no_CI(m):
    """
    Generate metrics text without confidence intervals and without N.
    
    Parameters
    ----------
    m : dict
        Metrics dictionary from calculate_some_classification_metrics
        
    Returns
    -------
    str
        Formatted metrics text
    """
    # --- RMSE (no CI)
    rmse_txt = f"$\\mathrm{{RMSE}}= {m['rmse']:2.2f}$"
    
    # --- Prefer Pearson r; fallback to Spearman rho (no CI)
    corr_key, corr_label = None, None
    if "pearson_corr" in m:
        corr_key, corr_label = "pearson_corr", "\\rho   "
    elif "spearman_corr" in m:
        corr_key, corr_label = "spearman_corr", "\\rho   "
    
    corr_txt = ""
    if corr_key is not None:
        corr_txt = f"${corr_label}= {m[corr_key]:2.2f}$"
    
    # --- ICC (psych) (no CI, no N)
    txt_ICC_psych = ""
    if "ICC_psych" in m:
        txt_ICC_psych = f"\n$\\mathrm{{ICC}} = {m['ICC_psych']:2.2f}$"
    
    return rmse_txt + txt_ICC_psych + "\n" + corr_txt


def plot_comparison_two_models(
    df_model1: pd.DataFrame,
    df_model2: pd.DataFrame,
    plot_type: str = "sums",
    *,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
    plot_name: str = "Comparison",
    figsize=(5, 5),
    regplot: bool = True,
    icc: str = "ICC3",
    metrics_loc: tuple[float, float] = (0.05, 0.95),
    model1_color: str = 'C0',
    model2_color: str = 'C1',
    ax=None,
    show_legend: bool = True,
    show_CI: bool = False,
    font_size: int = 10,
    text_box_permutation: list = [0, 1]
):
    """
    Plot scatter plots for two models on the same axes with different colors.

    Parameters
    ----------
    df_model1 : pd.DataFrame
        First model's dataframe (should have columns for true and predicted values)
    df_model2 : pd.DataFrame
        Second model's dataframe (should have columns for true and predicted values)
    plot_type : str, default "sums"
        Type of plot: "sums" or "deltas"
        - "sums": uses "labels_summed_extrapolated" and "preds_summed_extrapolated"
        - "deltas": uses "labels_summed_extrapolated_delta" and "preds_summed_extrapolated_delta"
    model1_name : str, default "Model 1"
        Name for first model (used in legend and metrics)
    model2_name : str, default "Model 2"
        Name for second model (used in legend and metrics)
    plot_name : str, default "Comparison"
        Name for the plot (used in axis labels)
    figsize : tuple, default (5, 5)
        Figure size
    regplot : bool, default True
        Whether to overlay regression lines
    icc : str, default "ICC3"
        ICC type to compute
    metrics_loc : tuple, default (0.05, 0.95)
        Location for metrics text (x, y) in axes coordinates
    model1_color : str, default 'C0'
        Color for model 1
    model2_color : str, default 'C1'
        Color for model 2
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new figure.
    show_legend : bool, default True
        Whether to display the legend.
    show_CI : bool, default False
        Whether to show confidence intervals in metrics.
    font_size : int, default 10
        Font size for axis labels and metrics text.
    text_box_permutation : list, default [0, 1]
        Order in which to display metrics text boxes for the two models. Permute [0, 1] as desired.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object (None if ax was provided)
    ax : matplotlib.axes.Axes
        Axes object
    metrics_dict : dict
        Dictionary with keys 'model1' and 'model2', each containing metrics dict
    """
    import seaborn as sns
    from ra_utils.training.scores_SHS.scores_SHS_training_lib import (
        calculate_some_classification_metrics
    )
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Determine column names based on plot type
    if plot_type == "sums":
        true_col = "labels_summed_extrapolated"
        pred_col = "preds_summed_extrapolated"
    elif plot_type == "deltas":
        true_col = "labels_summed_extrapolated_delta"
        pred_col = "preds_summed_extrapolated_delta"
    else:
        raise ValueError(f"plot_type must be 'sums' or 'deltas', got '{plot_type}'")

    # Prepare data for both models
    def prepare_data(df):
        x_raw = df[true_col]
        y_raw = df[pred_col]
        x = pd.to_numeric(x_raw, errors='coerce')
        y = pd.to_numeric(y_raw, errors='coerce')
        mask = x.notna() & y.notna()
        return x[mask], y[mask]

    x1, y1 = prepare_data(df_model1)
    x2, y2 = prepare_data(df_model2)

    # Compute metrics for both models
    metrics1 = calculate_some_classification_metrics(
        all_labels=x1.values.astype(float),
        all_preds=y1.values.astype(float),
        calc_ICC3=2,
        add_classification_metrics=False,
        add_spearman=False,
        add_pearson=True,
        add_kappa=True,
        calc_psych_ICC=2,
        icc=icc,
        calculate_CI=show_CI  # Control CI via kwarg
    )

    metrics2 = calculate_some_classification_metrics(
        all_labels=x2.values.astype(float),
        all_preds=y2.values.astype(float),
        calc_ICC3=2,
        add_classification_metrics=False,
        add_spearman=False,
        add_pearson=True,
        add_kappa=True,
        calc_psych_ICC=2,
        icc=icc,
        calculate_CI=show_CI
    )

    # Create or use existing axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    # Plot model 1
    if regplot:
        sns.regplot(
            x=x1, y=y1, ax=ax,
            scatter_kws={'color': model1_color, 'alpha': 0.3, 's': 8},
            line_kws={'color': model1_color, 'lw': 1},
            truncate=False
        )
        # Manually add legend entry for model 1
        line1 = Line2D([0], [0], color=model1_color, lw=1, label=model1_name)
    else:
        ax.scatter(x1, y1, c=model1_color, alpha=0.3, s=8, label=model1_name)
        line1 = None

    # Plot model 2
    if regplot:
        sns.regplot(
            x=x2, y=y2, ax=ax,
            scatter_kws={'color': model2_color, 'alpha': 0.3, 's': 8},
            line_kws={'color': model2_color, 'lw': 1},
            truncate=False
        )
        # Manually add legend entry for model 2
        line2 = Line2D([0], [0], color=model2_color, lw=1, label=model2_name)
    else:
        ax.scatter(x2, y2, c=model2_color, alpha=0.3, s=8, label=model2_name)
        line2 = None

    # Diagonal reference line
    all_x = pd.concat([x1, x2])
    all_y = pd.concat([y1, y2])
    D=30
    lims = [min(all_x.min()-D, all_y.min())-D, max(all_x.max()+D, all_y.max())+D]
    ax.plot(lims, lims, '--', color='gray', lw=1, alpha=0.7)

    # Set labels with custom font size
    ax.set_xlabel(f"Ground-truth scores ({plot_name})", fontsize=font_size)
    ax.set_ylabel(f"Predicted scores ({plot_name})", fontsize=font_size)
    ax.tick_params(axis='both', labelsize=font_size)
    ax.grid()

    # Add legend if requested
    if show_legend:
        if regplot and line1 is not None and line2 is not None:
            ax.legend(handles=[line1, line2], loc='lower right', fontsize=font_size)
        else:
            ax.legend(loc='lower right', fontsize=font_size)
    else:
        ax.legend_.remove() if ax.get_legend() is not None else None

    # Add metrics text in two separate vertically-aligned boxes, second one shifted right of the first
    if show_CI:
        txt1 = metrics_to_text_no_CI(metrics1)  # Replace with metrics_to_text_with_CI if available
        txt2 = metrics_to_text_no_CI(metrics2)
    else:
        txt1 = metrics_to_text_no_CI(metrics1)
        txt2 = metrics_to_text_no_CI(metrics2)
        
    box_props = dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3)

    x0, y0 = metrics_loc

    # Default right shift in axis coordinates, adjust for font size
    x_shift = 0.32 * (font_size / 10)  # This should work well for two 2-3 line metrics boxes at fontsize=10

    # Prepare model data for permutation
    models_data = [
        (model1_name, txt1),
        (model2_name, txt2)
    ]

    # Validate text_box_permutation
    if set(text_box_permutation) != {0, 1}:
        raise ValueError(f"text_box_permutation must be a permutation of [0, 1], got {text_box_permutation}")
    if len(text_box_permutation) != 2:
        raise ValueError(f"text_box_permutation must have length 2, got {len(text_box_permutation)}")

    # Place metrics boxes according to permutation order
    # The x-position is based on the position in the permutation, not the model index
    for i, model_idx in enumerate(text_box_permutation):
        model_name, model_txt = models_data[model_idx]
        x_pos = x0 + i * x_shift
        if x_pos > 0.98:
            x_pos = 0.98
        ax.text(
            x_pos, y0,
            f"{model_name}:\n{model_txt}",
            transform=ax.transAxes,
            ha='left', va='top', fontsize=font_size,
            bbox=box_props
        )

    metrics_dict = {
        'model1': metrics1,
        'model2': metrics2
    }

    return fig, ax, metrics_dict


def plot_comparison_three_models(
    df_model1: pd.DataFrame,
    df_model2: pd.DataFrame,
    df_model3: pd.DataFrame,
    plot_type: str = "sums",
    *,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
    model3_name: str = "Model 3",
    plot_name: str = "Comparison",
    figsize=(5, 5),
    regplot: bool = True,
    icc: str = "ICC3",
    metrics_loc: tuple[float, float] = (0.05, 0.95),
    model1_color: str = 'C0',
    model2_color: str = 'C1',
    model3_color: str = 'C2',
    ax=None,
    show_legend: bool = True,
    show_CI: bool = False,
    font_size: int = 10,
    text_box_permutation: list = [0, 1, 2]
):
    """
    Plot scatter plots for three models on the same axes with different colors.

    Parameters
    ----------
    df_model1 : pd.DataFrame
        First model's dataframe (should have columns for true and predicted values)
    df_model2 : pd.DataFrame
        Second model's dataframe (should have columns for true and predicted values)
    df_model3 : pd.DataFrame
        Third model's dataframe (should have columns for true and predicted values)
    plot_type : str, default "sums"
        Type of plot: "sums" or "deltas"
        - "sums": uses "labels_summed_extrapolated" and "preds_summed_extrapolated"
        - "deltas": uses "labels_summed_extrapolated_delta" and "preds_summed_extrapolated_delta"
    model1_name : str, default "Model 1"
        Name for first model (used in legend and metrics)
    model2_name : str, default "Model 2"
        Name for second model (used in legend and metrics)
    model3_name : str, default "Model 3"
        Name for third model (used in legend and metrics)
    plot_name : str, default "Comparison"
        Name for the plot (used in axis labels)
    figsize : tuple, default (5, 5)
        Figure size
    regplot : bool, default True
        Whether to overlay regression lines
    icc : str, default "ICC3"
        ICC type to compute
    metrics_loc : tuple, default (0.05, 0.95)
        Location for metrics text (x, y) in axes coordinates (top-left corner of first box)
    model1_color : str, default 'C0'
        Color for model 1
    model2_color : str, default 'C1'
        Color for model 2
    model3_color : str, default 'C2'
        Color for model 3
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new figure.
    show_legend : bool, default True
        Whether to display the legend.
    show_CI : bool, default False
        Whether to show confidence intervals in metrics.
    font_size : int, default 10
        Font size for axis labels and metrics text.
    text_box_permutation : list, default [0, 1, 2]
        Permutation of [0, 1, 2] to reorder the metrics boxes.
        Indices refer to model1 (0), model2 (1), model3 (2).
        Example: [2, 0, 1] displays model3, model1, model2 from left to right.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object (None if ax was provided)
    ax : matplotlib.axes.Axes
        Axes object
    metrics_dict : dict
        Dictionary with keys 'model1', 'model2', and 'model3', each containing metrics dict
    """
    import seaborn as sns
    from ra_utils.training.scores_SHS.scores_SHS_training_lib import (
        calculate_some_classification_metrics
    )
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Validate text_box_permutation
    if set(text_box_permutation) != {0, 1, 2}:
        raise ValueError(f"text_box_permutation must be a permutation of [0, 1, 2], got {text_box_permutation}")
    if len(text_box_permutation) != 3:
        raise ValueError(f"text_box_permutation must have length 3, got {len(text_box_permutation)}")

    # Determine column names based on plot type
    if plot_type == "sums":
        true_col = "labels_summed_extrapolated"
        pred_col = "preds_summed_extrapolated"
    elif plot_type == "deltas":
        true_col = "labels_summed_extrapolated_delta"
        pred_col = "preds_summed_extrapolated_delta"
    else:
        raise ValueError(f"plot_type must be 'sums' or 'deltas', got '{plot_type}'")

    # Prepare data for all three models
    def prepare_data(df):
        x_raw = df[true_col]
        y_raw = df[pred_col]
        x = pd.to_numeric(x_raw, errors='coerce')
        y = pd.to_numeric(y_raw, errors='coerce')
        mask = x.notna() & y.notna()
        return x[mask], y[mask]

    x1, y1 = prepare_data(df_model1)
    x2, y2 = prepare_data(df_model2)
    x3, y3 = prepare_data(df_model3)

    # Compute metrics for all three models
    metrics1 = calculate_some_classification_metrics(
        all_labels=x1.values.astype(float),
        all_preds=y1.values.astype(float),
        calc_ICC3=2,
        add_classification_metrics=False,
        add_spearman=False,
        add_pearson=True,
        add_kappa=True,
        calc_psych_ICC=2,
        icc=icc,
        calculate_CI=show_CI
    )

    metrics2 = calculate_some_classification_metrics(
        all_labels=x2.values.astype(float),
        all_preds=y2.values.astype(float),
        calc_ICC3=2,
        add_classification_metrics=False,
        add_spearman=False,
        add_pearson=True,
        add_kappa=True,
        calc_psych_ICC=2,
        icc=icc,
        calculate_CI=show_CI
    )

    metrics3 = calculate_some_classification_metrics(
        all_labels=x3.values.astype(float),
        all_preds=y3.values.astype(float),
        calc_ICC3=2,
        add_classification_metrics=False,
        add_spearman=False,
        add_pearson=True,
        add_kappa=True,
        calc_psych_ICC=2,
        icc=icc,
        calculate_CI=show_CI
    )

    # Create or use existing axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    # Plot model 1
    if regplot:
        sns.regplot(
            x=x1, y=y1, ax=ax,
            scatter_kws={'color': model1_color, 'alpha': 0.3, 's': 8},
            line_kws={'color': model1_color, 'lw': 1},
            truncate=False
        )
        line1 = Line2D([0], [0], color=model1_color, lw=1, label=model1_name)
    else:
        ax.scatter(x1, y1, c=model1_color, alpha=0.3, s=8, label=model1_name)
        line1 = None

    # Plot model 2
    if regplot:
        sns.regplot(
            x=x2, y=y2, ax=ax,
            scatter_kws={'color': model2_color, 'alpha': 0.3, 's': 8},
            line_kws={'color': model2_color, 'lw': 1},
            truncate=False
        )
        line2 = Line2D([0], [0], color=model2_color, lw=1, label=model2_name)
    else:
        ax.scatter(x2, y2, c=model2_color, alpha=0.3, s=8, label=model2_name)
        line2 = None

    # Plot model 3
    if regplot:
        sns.regplot(
            x=x3, y=y3, ax=ax,
            scatter_kws={'color': model3_color, 'alpha': 0.3, 's': 8},
            line_kws={'color': model3_color, 'lw': 1},
            truncate=False
        )
        line3 = Line2D([0], [0], color=model3_color, lw=1, label=model3_name)
    else:
        ax.scatter(x3, y3, c=model3_color, alpha=0.3, s=8, label=model3_name)
        line3 = None

    # Diagonal reference line
    all_x = pd.concat([x1, x2, x3])
    all_y = pd.concat([y1, y2, y3])
    lims = [min(all_x.min(), all_y.min()), max(all_x.max(), all_y.max())]
    ax.plot(lims, lims, '--', color='gray', lw=1, alpha=0.7)

    # Set labels with custom font size
    ax.set_xlabel(f"Ground-truth scores ({plot_name})", fontsize=font_size)
    ax.set_ylabel(f"Predicted scores ({plot_name})", fontsize=font_size)
    ax.tick_params(axis='both', labelsize=font_size)
    ax.grid()

    # Add legend if requested
    if show_legend:
        if regplot and line1 is not None and line2 is not None and line3 is not None:
            ax.legend(handles=[line1, line2, line3], loc='lower right', fontsize=font_size)
        else:
            ax.legend(loc='lower right', fontsize=font_size)
    else:
        if ax.get_legend() is not None:
            ax.legend_.remove()

    # Add metrics text in three separate boxes, arranged horizontally
    if show_CI:
        txt1 = metrics_to_text_no_CI(metrics1)  # Replace with metrics_to_text_with_CI if available
        txt2 = metrics_to_text_no_CI(metrics2)
        txt3 = metrics_to_text_no_CI(metrics3)
    else:
        txt1 = metrics_to_text_no_CI(metrics1)
        txt2 = metrics_to_text_no_CI(metrics2)
        txt3 = metrics_to_text_no_CI(metrics3)
        
    # Prepare model data for permutation
    models_data = [
        (model1_name, txt1),
        (model2_name, txt2),
        (model3_name, txt3)
    ]
    
    box_props = dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3)

    x0, y0 = metrics_loc

    # Calculate spacing for three boxes horizontally, adjust for font size
    x_shift = 0.28 * (font_size / 10)  # Slightly tighter spacing for three boxes

    # Place metrics boxes according to permutation
    for i, model_idx in enumerate(text_box_permutation):
        model_name, model_txt = models_data[model_idx]
        x_pos = x0 + i * x_shift
        if x_pos > 0.98:
            x_pos = 0.98
        ax.text(
            x_pos, y0,
            f"{model_name}:\n{model_txt}",
            transform=ax.transAxes,
            ha='left', va='top', fontsize=font_size,
            bbox=box_props
        )

    metrics_dict = {
        'model1': metrics1,
        'model2': metrics2,
        'model3': metrics3
    }

    return fig, ax, metrics_dict


def main():
    FRACTION_REQUIRED_VALID_SCORES = 0.75

    parser = argparse.ArgumentParser(description='Plot and save true vs predicted summed scores')
    parser.add_argument(
        '--run_path', '-r', type=str, 
        required=False,  # <<<<
        default="/msc/home/cwatze93/data/mlflow/mlflow_RA/844424495910332051/07e62b6853554a9a911d916113bcd168",
        help='Path to the MLflow run directory'
    )
    # soft_prediction becomes a simple flag: absent=False, present=True
    parser.add_argument(
        '--no-soft-prediction', dest='no_soft_prediction',
        action='store_true', 
        default=False,
        help='Use "preds" instead of "preds_float" (used to be for classifcation sort of sum Σ_c p̂_c * c label) (default: False)'
    )
    # strict supports --strict / --no-strict
    parser.add_argument(
        '--strict', action=argparse.BooleanOptionalAction, 
        default=False, 
        help='Assert output folder does not exist (default: True). Use --no-strict to allow overwrite.'
    )
    parser.add_argument(
        '--limit_treatment_ED', type=str, default="cap E_D sum to 5",
        help='How to handle ED score pairs for hands. Options: "cap E_D sum to 5" or "E_D_mean  default: "cap E_D sum to 5" '
    )
    
    parser.add_argument('--data_partition', type=str, default="valFinal",
                        help='Data partition to use. Options: "valFinal" or "test" (default: "valFinal")')
    
    #--data_partition  valFinal  test
 
    args = parser.parse_args()
    run_path = args.run_path
    soft_prediction = not args.no_soft_prediction
    limit_treatment_ED = args.limit_treatment_ED
    strict = args.strict
    data_partition = args.data_partition

    

    args = parser.parse_args()
    print("ARGS::")
    print(args)
    
    run_path = args.run_path

    


    # Locate predictions
    preds = f"{run_path}/artifacts/predictions/{data_partition}_/*.npz"
    npz_files = glob.glob(preds)
    assert len(npz_files) == 1, f"Expected exactly one npz file, found {len(npz_files)}"
    npz_file = npz_files[0]


    # Create the output folder
    output_folder = f"{run_path}/artifacts/plots/00_summed_scores/{data_partition}/"
    if strict: 
        assert not os.path.exists(output_folder), f"Output folder already exists: {output_folder}"
    os.makedirs(output_folder, exist_ok=True)



    # Load predictions
    predictions_path = npz_file
    src = predictions_path

    df = combine_predictions([src], keys_for_df = ['labels', 'preds', 'file_name', 'score_type', 'JSN_or_ERO', 'extremity', 'patient_id', "preds_float"])
    df["patientId_date_HF_LR"] = df["file_name"].apply(lambda x: "_".join(x.split("_")[:4]))
    df["patientId_date_HF"] = df["file_name"].apply(lambda x: "_".join(x.split("_")[:3]))
    df["patientId_date"] = df["file_name"].apply(lambda x: "_".join(x.split("_")[:2]))


    # For now correct predictions... Class 6 can not occur anyhow...
    # df[df["JSN_or_ERO"]== "JSN"]["labels"].value_counts()  ## They are already filtered (surgery was exluded, ... )
    #df["labels"] = df["labels"].apply(lambda x: limit_treatment_number(x, limit=5, limit_treatment="over_limit_to_NA"))
    #df["preds"] = df["preds"].apply(lambda x: limit_treatment_number(x, limit=5, limit_treatment="over_limit_to_NA"))

    df = df.dropna()
    df_ALL  = df
    if soft_prediction: 
        df["preds"] = df["preds_float"]
        dst_score_progression = f"{output_folder}/score_progression_EV.png"
        dst_sums_plot = f"{output_folder}/score_sums_true_vs_predicted_EV.png"
    else: 
        dst_score_progression = f"{output_folder}/score_progression.png"
        dst_sums_plot = f"{output_folder}/score_sums_true_vs_predicted.png"

    ## We only have subset of scores in this run... 
    m = (df["extremity"] == "F") & (df["JSN_or_ERO"] == "ERO")
    #max_summed_score_F_ERO_ = len(set(df[m]["score_type"])) * 5 * 2  # score 5 is max but both sides
    max_summed_score_F_ERO = ra_utils.data.shap_sums.max_possible_score(
        list(set(df[m]["score_type"])),
        collapse_ED_EP_pairs=False
    )

    m = (df["extremity"] == "F") & (df["JSN_or_ERO"] == "JSN")
    #max_summed_score_F_JSN_ = len(set(df[m]["score_type"])) * 4 * 2  # score 4; both sides
    max_summed_score_F_JSN = ra_utils.data.shap_sums.max_possible_score(
        list(set(df[m]["score_type"])),
        collapse_ED_EP_pairs=True
    )

    m = (df["extremity"] == "H") & (df["JSN_or_ERO"] == "ERO")
    max_summed_score_H_ERO = ra_utils.data.shap_sums.max_possible_score(
        list(set(df[m]["score_type"])),
        collapse_ED_EP_pairs=True
    )

    m = (df["extremity"] == "H") & (df["JSN_or_ERO"] == "JSN")
    #max_summed_score_H_JSN_ = len(set(df[m]["score_type"])) * 4 * 2  # score 4; both sides
    max_summed_score_H_JSN = ra_utils.data.shap_sums.max_possible_score(
        list(set(df[m]["score_type"])),
        collapse_ED_EP_pairs=True
    )


    ######################################
    #  ERO H
    df = df_ALL
    df_summed_H = sum_and_extrapolate_scores_df_ERO_H(df, fraction_required_valid_scores=FRACTION_REQUIRED_VALID_SCORES, limit_treatment_ED=limit_treatment_ED, max_total=max_summed_score_H_ERO)
    df_delta_H = generate_score_differences(df_summed_H)

    # plot_SHS_sums(df_summed_H, figsize=(5,5))
    # plt.show()

    assert max_summed_score_H_ERO <= 160

    m_wrong = (df_summed_H["limit_summed"] > max_summed_score_H_ERO)
    if m_wrong.sum() > 0:
        print("There are patients with wrong summed scores! Please check the data!")
        df_summed_H["limit_summed"].hist(bins=50)

    # plot_SHS_deltas(df_delta_H)
    # plt.show()




    ######################################
    #  ERO F
    df = df_ALL
    df_summed_F = sum_and_extrapolate_scores_df_ERO_F(df, fraction_required_valid_scores=FRACTION_REQUIRED_VALID_SCORES, max_total=max_summed_score_F_ERO)

    df_delta_F = generate_score_differences(df_summed_F)
    # plot_SHS_sums(df_summed_F, figsize=(5,5))
    # plt.show()

    assert max_summed_score_F_ERO <= 120

    m_wrong = (df_summed_F["limit_summed"] > max_summed_score_F_ERO)
    if m_wrong.sum() > 0:
        print("There are patients with wrong summed scores! Please check the data!")
        df_summed_F["limit_summed"].hist(bins=50)



    ######################################
    #  ERO H+F
    # Plot for ERO model vs Gabi
    df_summed_ERO_H_F = (
        df_summed_H
        .set_index('patientId_date')
        .add(df_summed_F.set_index('patientId_date'), fill_value=None)
        .reset_index()
    )
    # fig__ERO_H_F, (_, ax_scatter__ERO_H_F, _) =  plot_SHS_sums(df_summed_ERO_H_F.dropna(), name="ERO H+F", figsize=(5,5), plot_histograms=False, regplot=True)
    # plt.show()

    m_wrong = (df_summed_ERO_H_F["limit_summed"] > (max_summed_score_F_ERO + max_summed_score_H_ERO))
    if m_wrong.sum() > 0:
        print("There are patients with wrong summed scores! Please check the data!")
        df_summed_ERO_H_F["limit_summed"].hist(bins=50)





    ######################################
    #  JSN H
    # Plot for JSN H model vs Gabi
    df = df_ALL
    df_summed_JSN_H = sum_and_extrapolate_scores_df_JSN_H(df, fraction_required_valid_scores=FRACTION_REQUIRED_VALID_SCORES, max_total=max_summed_score_H_JSN)
    df_delta_JSN_H = generate_score_differences(df_summed_JSN_H)

    # plot_SHS_sums(df_summed_JSN_H, figsize=(5,5), name="JSN H", plot_histograms=False)
    # plt.show()
    assert max_summed_score_H_JSN <= 120

    m_wrong = (df_summed_JSN_H["limit_summed"] > max_summed_score_H_JSN)
    if m_wrong.sum() > 0:
        print("There are patients with wrong summed scores! Please check the data!")
        df_summed_JSN_H["limit_summed"].hist(bins=50)


    # plot_SHS_deltas(df_delta_JSN_H)
    # plt.show()



    ######################################
    #  JSN F
    # Plot for JSN F model vs Gabi
    df = df_ALL
    df_summed_JSN_F = sum_and_extrapolate_scores_df_JSN_F(df, fraction_required_valid_scores=FRACTION_REQUIRED_VALID_SCORES, max_total=max_summed_score_F_JSN)
    df_delta_JSN_F = generate_score_differences(df_summed_JSN_F)



    # plot_SHS_sums(df_summed_JSN_F, figsize=(5,5), name="JSN F", plot_histograms=False, regplot=True);
    # # plt.show()
    assert max_summed_score_F_JSN <=48

    m_wrong = (df_summed_JSN_F["limit_summed"] > 48)
    if m_wrong.sum() > 0:
        print("There are patients with wrong summed scores! Please check the data!")
        df_summed_JSN_F["limit_summed"].hist(bins=50)

    # plot_SHS_deltas(df_delta_JSN_F)
    # plt.show()


    ######################################
    #  JSN H+F


    # Plot for JSN model vs Gabi
    df_summed_H_F_JSN = (
        df_summed_JSN_H
        .set_index('patientId_date')
        .add(df_summed_JSN_F.set_index('patientId_date'), fill_value=None)
        .reset_index()
    )

    # fig__JSN_H_F, (_, ax_scatter__JSN_H_F, _) =   plot_SHS_sums(df_summed_H_F_JSN.dropna(), figsize=(5,5), name="JSN H+F", plot_histograms=False, regplot=True)
    # plt.show()



    ######################################
    # SUMMED and delta: 
    # Plot for SvH model vs Gabi
    df_summed_SvH = (
        df_summed_H_F_JSN
        .set_index('patientId_date')
        .add(df_summed_ERO_H_F.set_index('patientId_date'), fill_value=None)
        .reset_index()
    )

    # fig__SvH_H_F, (_, ax_scatter__SvH_H_F, _) = plot_SHS_sums(df_summed_SvH.dropna(), figsize=(5,5), name="SvH H+F", plot_histograms=False, regplot=True)
    # plt.show()

    df_delta_SvH = generate_score_differences(df_summed_SvH)
    
    # Save dataframes
    df_delta_SvH_path = os.path.join(output_folder, 'df_delta_SvH.csv')
    df_summed_SvH_path = os.path.join(output_folder, 'df_summed_SvH.csv')
    df_delta_SvH.to_csv(df_delta_SvH_path, index=False)
    df_summed_SvH.to_csv(df_summed_SvH_path, index=False)
    print(f" Saved df_delta_SvH to {df_delta_SvH_path}")
    print(f" Saved df_summed_SvH to {df_summed_SvH_path}")
    
    fig, axis_tuple, metrics_delta_SvH = plot_SHS_deltas(df_delta_SvH.dropna(), figsize=(5,5), name="ΔSvH H+F", regplot=True)

    # Save metrics for delta SvH
    if metrics_delta_SvH is not None:
        metrics_delta_path = os.path.join(os.path.dirname(dst_score_progression), 'metrics_delta_SvH.yaml')
        with open(metrics_delta_path, 'w') as f:
            yaml.dump(metrics_delta_SvH, f, default_flow_style=False, sort_keys=False)
        print(f" Saved metrics to {metrics_delta_path}")

    fig.savefig(dst_score_progression)
    print(f" Saved figure to {dst_score_progression}")

    #plt.show()




    # Combined plot
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # # Recreate plots directly into subplots
    # _ = plot_SHS_sums(df_summed_SvH.dropna(),    ax_scatter=axs[0], name="SvH H+F", plot_histograms=False, regplot=True)
    # _ = plot_SHS_sums(df_summed_H_F_JSN.dropna(), ax_scatter=axs[1], name="JSN H+F", plot_histograms=False, regplot=True)
    # _ = plot_SHS_sums(df_summed_ERO_H_F.dropna(), ax_scatter=axs[2], name="ERO H+F", plot_histograms=False, regplot=True)
    # fig.tight_layout()

    fig.savefig(dst_sums_plot)
    print(f" Saved figure to {dst_sums_plot}")

    # Save individual 1x1 plots
    fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    _, _, metrics_SvH = plot_SHS_sums(df_summed_SvH.dropna(), ax_scatter=ax1, name="SvH H+F", plot_histograms=False, regplot=True)
    fig1.tight_layout()
    suffix = "_EV_SvH_H+F.png" if soft_prediction else "_SvH_H+F.png"
    dst_svH_individual = dst_sums_plot.replace('.png', suffix)
    fig1.savefig(dst_svH_individual)
    print(f" Saved figure to {dst_svH_individual}")
    # Save metrics for SvH
    if metrics_SvH is not None:
        metrics_svH_path = os.path.join(os.path.dirname(dst_sums_plot), 'metrics_SvH_H+F.yaml')
        with open(metrics_svH_path, 'w') as f:
            yaml.dump(metrics_SvH, f, default_flow_style=False, sort_keys=False)
        print(f" Saved metrics to {metrics_svH_path}")

    fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5))
    _, _, metrics_JSN = plot_SHS_sums(df_summed_H_F_JSN.dropna(), ax_scatter=ax2, name="JSN H+F", plot_histograms=False, regplot=True)
    fig2.tight_layout()
    suffix = "_EV_JSN_H+F.png" if soft_prediction else "_JSN_H+F.png"
    dst_jsn_individual = dst_sums_plot.replace('.png', suffix)
    fig2.savefig(dst_jsn_individual)
    print(f" Saved figure to {dst_jsn_individual}")
    # Save metrics for JSN
    if metrics_JSN is not None:
        metrics_jsn_path = os.path.join(os.path.dirname(dst_sums_plot), 'metrics_JSN_H+F.yaml')
        with open(metrics_jsn_path, 'w') as f:
            yaml.dump(metrics_JSN, f, default_flow_style=False, sort_keys=False)
        print(f" Saved metrics to {metrics_jsn_path}")

    fig3, ax3 = plt.subplots(1, 1, figsize=(5, 5))
    _, _, metrics_ERO = plot_SHS_sums(df_summed_ERO_H_F.dropna(), ax_scatter=ax3, name="ERO H+F", plot_histograms=False, regplot=True)
    fig3.tight_layout()
    suffix = "_EV_ERO_H+F.png" if soft_prediction else "_ERO_H+F.png"
    dst_ero_individual = dst_sums_plot.replace('.png', suffix)
    fig3.savefig(dst_ero_individual)
    print(f" Saved figure to {dst_ero_individual}")
    # Save metrics for ERO
    if metrics_ERO is not None:
        metrics_ero_path = os.path.join(os.path.dirname(dst_sums_plot), 'metrics_ERO_H+F.yaml')
        with open(metrics_ero_path, 'w') as f:
            yaml.dump(metrics_ERO, f, default_flow_style=False, sort_keys=False)
        print(f" Saved metrics to {metrics_ero_path}")


    #plt.show()
    print("DONE!!!")

if __name__ == "__main__":
    main()
