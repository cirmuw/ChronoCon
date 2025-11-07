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



# wrap model: 

from ra_utils.networks.architecture import (
    MultiModalImageScoreTypeNetworkAE,
    ROI_type_encoder
)



# get all scores types / roi types
import ra_utils.utils

import ra_utils.utils.config_parser
import torchvision.transforms.v2 as v2

import ra_utils.networks.loss_function
from ra_utils.networks.loss_function import get_score_loss_function, get_triplet_loss_fn

import torch.nn.functional as F


from datetime import datetime
import seaborn as sns
from torchmetrics.functional import structural_similarity_index_measure as ssim
from ra_utils.visualization.structural_similarity_index import (
    ssim_matrix_torchmetrics, 
    compute_and_plot_ssi
)
from ra_utils.visualization.plot_image_series import (
    plot_patch_series
)

from ra_utils.loss.online_mining_delta_loss import batch_all_score_differences_loss
from collections import Counter


import math
import numpy as np
import matplotlib.pyplot as plt

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def _to_numpy_image(x):
    """
    Accepts a tensor/ndarray in shape [C,H,W] / [H,W] / [H,W,C].
    Returns a numpy image in [H,W] (grayscale) or [H,W,3] (RGB).
    """
    try:
        import torch
        is_tensor = torch.is_tensor(x)
    except Exception:
        is_tensor = False

    if is_tensor:
        x = x.detach().cpu().numpy()

    # Move channel last if needed
    if x.ndim == 3:
        # [C,H,W] -> [H,W,C]
        if x.shape[0] in (1, 3):
            x = np.transpose(x, (1, 2, 0))
        # else assume already [H,W,C]

        # If single-channel, squeeze to [H,W]
        if x.shape[-1] == 1:
            x = x[..., 0]

        # If more than 3 channels, take first 3
        if x.ndim == 3 and x.shape[-1] > 3:
            x = x[..., :3]

    elif x.ndim == 2:
        pass  # [H,W] ok
    else:
        raise ValueError(f"Unsupported image shape: {x.shape}")

    # If floating with values outside [0,1], try min-max normalize per image
    if np.issubdtype(x.dtype, np.floating):
        x_min, x_max = np.nanmin(x), np.nanmax(x)
        if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
            x = (x - x_min) / (x_max - x_min)
        # else leave as-is

    return x


def plot_image_series(
    ds_instance,
    image_key = "img",
    text_to_add=("score", "date_str", "score_type"),
    axis=None,
    cols=None,
    figsize_per_image=(3.2, 3.2),
    tight_layout=True,
    cmap="gray",
    hide_axes_ticks=True,
    suptitle=None,
    ):
    """
    Plot a temporal (or ordered) series of images from a dataset of dict-like samples.

    Parameters
    ----------
    ds_instance : torch.utils.data.Dataset or Sequence[dict]
        Each item should be a dict with at least an 'image' key and optional metadata
        (e.g., 'score', 'date_str', 'score_type', 'left_or_right', etc.).
    text_to_add : Sequence[str]
        Keys from each sample to concatenate as multiline caption under each image.
    axis : list[matplotlib.axes.Axes] or None
        If provided, draw onto these axes (length must match number of images).
        If None, a new figure and axes are created.
    cols : int or None
        Number of columns in the grid. Defaults to len(ds_instance) (i.e., one row).
    figsize_per_image : tuple(float, float)
        Size (inches) per image tile, multiplied by cols/rows for overall figsize.
    tight_layout : bool
        Whether to call tight_layout() at the end.
    cmap : str or None
        Colormap for grayscale images. Ignored for RGB.
    hide_axes_ticks : bool
        If True, hides ticks and frames.
    suptitle : str or None
        Global title for the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of Axes
    """
    n = len(ds_instance)
    if n == 0:
        raise ValueError("Dataset instance is empty – nothing to plot.")

    # Determine grid: default to single row timeline
    if cols is None or cols <= 0:
        cols = n
    rows = math.ceil(n / cols)

    # Prepare axes
    created_new_fig = False
    if axis is None:
        w = max(cols, 1) * figsize_per_image[0]
        h = max(rows, 1) * figsize_per_image[1]
        fig, axes = plt.subplots(rows, cols, figsize=(w, h))
        created_new_fig = True
    else:
        # User-supplied axes
        axes = np.array(axis)
        # Flatten and ensure enough axes
        if axes.size < n:
            raise ValueError(f"Provided axis has {axes.size} axes but {n} images to plot.")
        # Create a dummy fig handle
        fig = axes.flat[0].get_figure()

    axes = np.array(axes).reshape(rows, cols)

    # Iterate samples
    for i in range(n):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        sample = ds_instance[i]  # expects dict-like: sample["image"], sample["score"], etc.

        img = _to_numpy_image(sample[image_key])
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap)
        else:  # RGB
            ax.imshow(img)

        if hide_axes_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

        # Build caption from requested metadata keys
        lines = []
        for k in text_to_add:
            if k in sample and sample[k] is not None:
                v = sample[k]
                # Numpy scalars -> Python types
                if hasattr(v, "item"):
                    try:
                        v = v.item()
                    except Exception:
                        pass
                lines.append(f"{k}: {v}")
        if lines:
            ax.set_title("\n".join(lines), fontsize=9)

    # Hide any unused axes (when n < rows*cols)
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        ax = axes[r, c]
        ax.axis("off")

    if suptitle and created_new_fig:
        fig.suptitle(suptitle, y=0.98, fontsize=12)

    if tight_layout and created_new_fig:
        plt.tight_layout()

    return fig, axes




#---------------------







def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "date_dt" not in df.columns:
        if "date_str" in df.columns:
            d = df.copy()
            d["date_dt"] = pd.to_datetime(d["date_str"], errors="coerce")
            return d
        else:
            raise KeyError("Neither 'date_dt' nor 'date_str' found in df.")
    return df

def _select_first_n_series(df: pd.DataFrame, n: int = 5):
    group_cols = ["patient_id", "left_or_right", "chosen_score"]
    # Ensure the group columns exist
    for c in group_cols:
        if c not in df.columns:
            raise KeyError(f"Required column '{c}' not found in df.")
    # Get unique combinations in stable order
    unique_series = df[group_cols].drop_duplicates().head(n)
    # Return as list of tuples
    return list(unique_series.itertuples(index=False, name=None))


def plot_some_score_series(df, n=5, score_modifier = None):
    df = _ensure_datetime(df)
    series_keys = _select_first_n_series(df, n=n)

    # 1) Individual line charts: score over time for each series
    for key in series_keys:
        pid, lor, score_name = key
        sub = df[
            (df["patient_id"] == pid) &
            (df["left_or_right"] == lor) &
            (df["chosen_score"] == score_name)
        ].sort_values("date_dt")

        # Guard: drop entries without dates or scores
        sub = sub.loc[sub["date_dt"].notna() & sub["score"].notna()]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(sub["date_dt"], sub["score"], marker="o")
        ax.set_title(f"Score over time — patient {pid}, {lor}, {score_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Score")
        fig.autofmt_xdate()  # rotate dates
        plt.show()

    # 2) Grouped boxplot of scores for the same 5 series
    box_data = []
    labels = []
    for key in series_keys:
        pid, lor, score_name = key
        sub = df[
            (df["patient_id"] == pid) &
            (df["left_or_right"] == lor) &
            (df["chosen_score"] == score_name)
        ]
        vals = sub["score"].dropna().values
        if vals.size > 0:
            box_data.append(vals)
            labels.append(f"{pid}-{lor}-{score_name}")
        else:
            # ensure alignment if empty, skip
            pass

    if box_data:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.boxplot(box_data, showfliers=True)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("Score distribution for first 5 series")
        plt.show()
    else:
        print("No valid score data to plot in the selected series.")

        return fig, ax
    
