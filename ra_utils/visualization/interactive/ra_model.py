### My model and datapipeline imports: 


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

import ra_utils.mtan.im2im_pred.model_resnet_mtan.resnet_mtan
from ra_utils.training.scores_SHS.model_builders import build_models_AE_v2, build_models_AE_v1_and2
from ra_utils.training.scores_SHS.run_training_main_lib import (
    maybe_partially_init_model_from_state_dict,
    check_config_consistency_and_partially_make_consistent
    )

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
    plot_patch_series, plot_triplet_dataloader_examples
)

from ra_utils.loss.online_mining_delta_loss import batch_all_score_differences_loss
from collections import Counter


import ra_utils.loss.loss_fn_dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence
from torch import Tensor
#  import online_triplet_loss.losses   # now added to ra_utils and modified
import ra_utils.loss.online_mining_triplet_loss
import ra_utils.loss.online_mining_triplet_loss_with_scores
import ra_utils.loss.consistency_regularizer_loss
from typing import Optional

import numpy as np

import ra_utils.loss.loss_utils


import ra_utils.features.embeddings
import ra_utils.visualization.embeddings


import umap



# Interactive embeddings imports
import fiftyone as fo
import fiftyone.core.labels as fol
from fiftyone import brain as fob
import numpy as np


import os, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import fiftyone as fo
import fiftyone.core.labels as fol
from fiftyone import brain as fob
from typing import Literal

# import torch.multiprocessing as mp
# mp.set_sharing_strategy("file_system")  # avoids /dev/shm exhaustion issues on Linux


# ---------- image saving ----------
def _to_uint8_image(arr: np.ndarray, robust=True):
    """
    arr: (H,W) or (C,H,W) float or int; returns uint8 image array for saving
    - robust=True does percentile-based min/max to avoid outlier washout
    """
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[0] in (1, 3):       # (C,H,W) -> (H,W,C)
        a = np.transpose(a, (1, 2, 0))
    if a.ndim == 3 and a.shape[-1] not in (1, 3):  # weird channel count -> take first
        a = a[..., 0]

    a = a.astype(np.float32)

    if robust:
        lo = np.percentile(a, 0.5)
        hi = np.percentile(a, 99.5)
        if hi <= lo:
            lo, hi = float(a.min()), float(a.max())
    else:
        lo, hi = float(a.min()), float(a.max())

    denom = max(hi - lo, 1e-8)
    a = (np.clip(a, lo, hi) - lo) / denom
    a = (a * 255.0).round().astype(np.uint8)

    # enforce 3 channels for nicer previews
    if a.ndim == 2:
        a = a  # grayscale OK
    elif a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    elif a.ndim == 3 and a.shape[-1] >= 3:
        a = a[..., :3]

    return a

def _hashed_suffix(p: str) -> str:
    return hashlib.md5(p.encode()).hexdigest()[:10]

def ensure_png_for_sample(tensor_img: torch.Tensor,
                          src_path: str,
                          out_root: str) -> str:
    """
    Save a post-transform tensor to PNG under out_root, mirroring the original
    filename with a short hash to avoid collisions. Returns png path.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    src = Path(str(src_path))
    stem = src.stem
    h = _hashed_suffix(str(src))
    png_name = f"{stem}_{h}.png"
    # Optional: keep a bit of folder context (e.g., ROI folder)
    subdir = src.parent.name  # one level of context; adjust as you like
    out_dir = out_root / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / png_name

    if not out_path.exists():
        arr = tensor_img.detach().cpu().numpy()
        img = _to_uint8_image(arr, robust=True)
        Image.fromarray(img).save(out_path)

    return str(out_path)

# ---------- embeddings + pack ----------
# @torch.no_grad()
# def extract_embeddings_and_pngs(model_AE, model_c, loader, device, score_estimator, 
#                                 #is_regression: bool,
#                                 task_type_y: Literal["classification", "regression", "classification_regression_mix"], 
                                
#                                 png_root: str):
#     model_AE.eval()
#     model_c.eval()

#     Z, png_paths, y_true_all, y_pred_all, conf_all= [], [], [], [], []
#     score_types, roi_names, patient_ids, extremities, lr_sides = [], [], [], [], []
#     score_difference_prev_visit_all, score_difference_next_visit_all  = [], []

#     for batch in loader:
#         X = batch["img"].to(device)
#         s_type = batch["score_type"]               # list[str]
#         y = batch["score"]
#         if torch.is_tensor(y):
#             y = y.detach().cpu().numpy()

#         # forward -> embedding + logits
#         _, z = model_AE(X, s_type)                 # (B, D)
#         logits = model_c(z, s_type)                # (B, C) or (B, 1)

#         # predictions
#         if task_type_y == "regression":
#             score_estimate = score_estimator(logits)
#             yhat = score_estimate.detach().cpu().numpy()
#             conf = np.full_like(yhat, np.nan)
#         elif task_type_y == "classification":
#             probs = torch.softmax(logits, dim=1)
#             yhat_idx = probs.argmax(1)
#             yhat = yhat_idx.detach().cpu().numpy()
#             conf = probs[torch.arange(len(yhat_idx)), yhat_idx].detach().cpu().numpy()
#         elif task_type_y == "classification_regression_mix":
#             probs = torch.softmax(logits[:,1:], dim=1)
#             yhat_idx = probs.argmax(1)
#             #yhat = yhat_idx.detach().cpu().numpy()
#             conf = probs[torch.arange(len(yhat_idx)), yhat_idx].detach().cpu().numpy()
#             score_estimate = score_estimator(logits).cpu().numpy()
#             yhat = score_estimate.detach().cpu().numpy()

#         # ensure a PNG for each sample *after* transforms
#         # we use your new 'image_path' field as the source-id
#         src_paths = batch.get("image_path", None)
#         if src_paths is None:
#             raise KeyError("Batch missing 'image_path'. Please add it in your Dataset dicts.")

#         for i in range(X.shape[0]):
#             png_p = ensure_png_for_sample(X[i], src_paths[i], png_root)
#             png_paths.append(png_p)

#         # meta
#         Z.append(z.detach().cpu().float().numpy())
#         y_true_all.extend(list(y))
#         y_pred_all.extend(list(yhat))
#         conf_all.extend(list(conf))
#         score_types.extend(list(s_type))
#         score_difference_prev_visit_all.extend(list(batch["score_difference_prev_visit"]))
#         score_difference_next_visit_all.extend(list(batch["score_difference_next_visit"]))


#         roi_names.extend(batch.get("roi_name", [""] * X.shape[0]))
#         patient_ids.extend(batch.get("patient_id", [""] * X.shape[0]))
#         extremities.extend(batch.get("extremity", [""] * X.shape[0]))
#         lr_sides.extend(batch.get("left_or_right", [""] * X.shape[0]))


#     Z = np.concatenate(Z, axis=0)

#     return dict(
#         Z=Z,
#         display_paths=np.array(png_paths),
#         y_true=np.array(y_true_all),
#         y_pred=np.array(y_pred_all),
#         conf=np.array(conf_all),
#         score_type=np.array(score_types),
#         roi_name=np.array(roi_names),
#         patient_id=np.array(patient_ids),
#         extremity=np.array(extremities),
#         left_or_right=np.array(lr_sides),
#         score_difference_prev_visit=np.array(score_difference_prev_visit_all),
#         score_difference_next_visit=np.array(score_difference_next_visit_all),
#     )

from ra_utils.utils.utils import datestr_to_years_since_2000
from ra_utils.utils.utils import datestr_to_years_since_2000

@torch.no_grad()
def extract_embeddings_and_pngs(
    model_AE,
    model_c,
    loader,
    device,
    score_estimator,
    task_type_y: Literal["classification", "regression", "classification_regression_mix"],
    png_root: str,
    max_class_per_type: dict[str, int] | None = None,
):
    model_AE.eval()
    model_c.eval()

    Z_list, png_paths = [], []
    score_gt, score_pred_all = [], []
    class_gt_all, class_pred_all, class_conf_all = [], [], []
    head_name_all = []

    score_types, roi_names, patient_ids, extremities, lr_sides = [], [], [], [], []
    d_prev_all, d_next_all = [], []

    # identity + time
    ps_key_all, years_all, date_str_all = [], [], []

    for batch in loader:
        X = batch["img"].to(device)
        s_type: list[str] = batch["score_type"]
        y = batch["score"]
        y = y.detach().cpu().float().numpy() if torch.is_tensor(y) else np.asarray(y, dtype=float)
        B = X.size(0)

        # identity + time
        ps_keys = batch["patient_scoretype_key"]
        years_np = datestr_to_years_since_2000(batch["date_str"])
        ps_key_all.extend(ps_keys)
        years_all.extend(years_np.tolist())
        date_str_all.extend(list(batch["date_str"]))

        # forward
        _, z = model_AE(X, s_type)              # (B, D)

        # multi-head call (mirror training)
        try:
            out_dict = model_c(z, s_type, return_dict=True)
        except TypeError:
            # fallback: single-head legacy path
            logits = model_c(z, s_type)         # (B, C) or (B,1) or (B,1+C)
            out_dict = {"__single__": (torch.arange(B, device=z.device), logits)}

        # merge per-head predictions into per-sample arrays
        score_pred, class_pred, class_conf, head_name_per_sample, has_logits = _merge_multihead_outputs(
            out_dict, task_type_y, score_estimator, B, device=device
        )

        # class ground-truth:
        if task_type_y == "classification":
            cls_gt = y.astype(int)
        else:
            # regression or mix: create class proxy from ground-truth score
            cls_gt = []
            for yi, sti in zip(y.astype(float).tolist(), s_type):
                mx = max_class_per_type.get(sti) if max_class_per_type else None
                cls_gt.append(round_to_class(float(yi), 0, mx))
            cls_gt = np.array(cls_gt, dtype=int)

        # scalar ground-truth
        score_gt.extend(y.tolist())
        score_pred_all.extend(score_pred.tolist())
        class_gt_all.extend(cls_gt.tolist())
        class_pred_all.extend(class_pred.tolist())
        class_conf_all.extend(class_conf.tolist())
        head_name_all.extend(head_name_per_sample)

        # pngs after transforms
        src_paths = batch.get("image_path", None)
        if src_paths is None:
            raise KeyError("Batch missing 'image_path'. Please add it in your Dataset dicts.")
        for i in range(B):
            png_paths.append(ensure_png_for_sample(X[i], src_paths[i], png_root))

        # meta
        Z_list.append(z.detach().cpu().float().numpy())
        score_types.extend(s_type)
        d_prev_all.extend(list(batch.get("score_difference_prev_visit", [None] * B)))
        d_next_all.extend(list(batch.get("score_difference_next_visit", [None] * B)))
        roi_names.extend(batch.get("roi_name", [""] * B))
        patient_ids.extend(batch.get("patient_id", [""] * B))
        extremities.extend(batch.get("extremity", [""] * B))
        lr_sides.extend(batch.get("left_or_right", [""] * B))

    Z = np.concatenate(Z_list, axis=0)

    return dict(
        Z=Z,
        display_paths=np.array(png_paths),

        # scalar view
        score_gt=np.array(score_gt, dtype=float),
        score_pred=np.array(score_pred_all, dtype=float),

        # classification view
        class_gt=np.array(class_gt_all, dtype=int),
        class_pred=np.array(class_pred_all, dtype=int),
        class_conf=np.array(class_conf_all, dtype=float),

        # which head produced the prediction for this sample
        head_name=np.array(head_name_all),

        # meta
        score_type=np.array(score_types),
        roi_name=np.array(roi_names),
        patient_id=np.array(patient_ids),
        extremity=np.array(extremities),
        left_or_right=np.array(lr_sides),
        score_difference_prev_visit=np.array(d_prev_all, dtype=object),
        score_difference_next_visit=np.array(d_next_all, dtype=object),

        # time + identity
        patient_scoretype_key=np.array(ps_key_all),
        years_since_2000=np.array(years_all, dtype=float),
        date_str=np.array(date_str_all),
    )


# ---------- FiftyOne dataset ----------
# def build_fiftyone_dataset(name, pack, task_type_y: Literal["classification", "regression", "classification_regression_mix"], 
#                            class_names=None):
#     if  class_names is None: # not is_regression and
#         uniq = sorted(set(pack["y_true"].tolist()))
#         #uniq = sorted(set(pack["y_true"].tolist()) | set(pack["y_pred"].tolist()))
#         class_names = {int(i): str(i) for i in uniq}

#     ds = fo.Dataset(name)
#     samples = []
#     for i in range(len(pack["display_paths"])):
#         s = fo.Sample(filepath=str(pack["display_paths"][i]))

#         if (task_type_y == "regression") or (task_type_y == "classification_regression_mix"):
#             s["ground_truth"] = float(pack["y_true"][i])
#             s["prediction"]   = float(pack["y_pred"][i])
#             s["wrong"] = None
#             s["confidence"] = None
#         else:
#             gt = class_names.get(int(pack["y_true"][i]), str(pack["y_true"][i]))
#             pr = class_names.get(int(pack["y_pred"][i]), str(pack["y_pred"][i]))
#             s["ground_truth"] = fol.Classification(label=gt)
#             s["prediction"]   = fol.Classification(
#                 label=pr,
#                 confidence=None if np.isnan(pack["conf"][i]) else float(pack["conf"][i])
#             )
#             s["wrong"] = (str(pr) != str(gt))
#             s["confidence"] = None if np.isnan(pack["conf"][i]) else float(pack["conf"][i])

#         s["score_type"]    = str(pack["score_type"][i])
#         s["roi_name"]      = str(pack["roi_name"][i]) if len(pack["roi_name"]) else ""
#         s["patient_id"]    = str(pack["patient_id"][i]) if len(pack["patient_id"]) else ""
#         s["extremity"]     = str(pack["extremity"][i]) if len(pack["extremity"]) else ""
#         s["left_or_right"] = str(pack["left_or_right"][i]) if len(pack["left_or_right"]) else ""
#         # But score_difference_prev_visit, score_difference_next_visit this might contain None entries. Not sure if this will be an issue... 
#         s["score_difference_prev_visit"] = str(pack["score_difference_prev_visit"][i]) if len(pack["score_difference_prev_visit"]) else ""
#         s["score_difference_next_visit"] = str(pack["score_difference_next_visit"][i]) if len(pack["score_difference_next_visit"]) else ""


#         s["embedding"]     = np.asarray(pack["Z"][i], dtype=np.float32)

#         samples.append(s)

#     ds.add_samples(samples)
#     return ds

def build_fiftyone_dataset(
    name: str,
    pack: dict,
    task_type_y: Literal["classification", "regression", "classification_regression_mix"],
    class_names: dict[int, str] | None = None,
):
    # make default class names if not provided
    if class_names is None:
        uniq = sorted(set(pack["class_gt"].tolist()) | set(pack["class_pred"].tolist()))
        class_names = {int(i): str(i) for i in uniq}

    ds = fo.Dataset(name)
    samples = []
    for i in range(len(pack["display_paths"])):
        s = fo.Sample(filepath=str(pack["display_paths"][i]))

        # always store scalar view
        s["score_gt"]   = float(pack["score_gt"][i])
        s["score_pred"] = float(pack["score_pred"][i])

        # always store classification view (using your rounding/mix)
        gt_label = class_names.get(int(pack["class_gt"][i]), str(int(pack["class_gt"][i])))
        pr_label = class_names.get(int(pack["class_pred"][i]), str(int(pack["class_pred"][i])))
        conf_val = pack["class_conf"][i]
        conf_val = None if (np.isnan(conf_val) or np.isinf(conf_val)) else float(conf_val)

        s["class_gt"] = fol.Classification(label=gt_label)
        s["class_pred"] = fol.Classification(label=pr_label, confidence=conf_val)

        # convenience flags
        s["wrong_cls"] = (gt_label != pr_label)
        # absolute error on scalar scores
        s["abs_err_score"] = float(abs(pack["score_pred"][i] - pack["score_gt"][i]))

        # meta
        s["score_type"]    = str(pack["score_type"][i])
        s["roi_name"]      = str(pack["roi_name"][i]) if len(pack["roi_name"]) else ""
        s["patient_id"]    = str(pack["patient_id"][i]) if len(pack["patient_id"]) else ""
        s["extremity"]     = str(pack["extremity"][i]) if len(pack["extremity"]) else ""
        s["left_or_right"] = str(pack["left_or_right"][i]) if len(pack["left_or_right"]) else ""

        s["patient_scoretype_key"] = str(pack["patient_scoretype_key"][i])
        s["years_since_2000"]      = float(pack["years_since_2000"][i])
        s["date_str"]              = str(pack["date_str"][i])

        # store None as None (not strings) for numeric filters
        prev_v = pack["score_difference_prev_visit"][i]
        next_v = pack["score_difference_next_visit"][i]
        s["score_difference_prev_visit"] = None if prev_v is None else float(prev_v)
        s["score_difference_next_visit"] = None if next_v is None else float(next_v)

        # embedding
        s["embedding"] = np.asarray(pack["Z"][i], dtype=np.float32)

        samples.append(s)

    ds.add_samples(samples)
    return ds

from sklearn.decomposition import PCA
import umap as umap_pkg

def compute_projection(Z: np.ndarray, method: str = "umap", seed: int = 42) -> np.ndarray:
    """
    Returns (N,2) projection for 'umap' or 'pca'.
    """
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(Z)
    elif method == "umap":
        reducer = umap_pkg.UMAP(n_components=2, random_state=seed)
        return reducer.fit_transform(Z)
    else:
        raise ValueError(f"Unknown method: {method}")

import plotly.graph_objects as go
from collections import defaultdict

def plot_trajectories_html(
    coords_2d: np.ndarray,           # (N,2)
    years: np.ndarray,               # (N,)
    keys: np.ndarray,                # (N,) patient_scoretype_key
    hover_text: list[str],           # per-point hover (e.g., score, date, score_type, etc.)
    color_by: np.ndarray | None,     # optional numeric to color points (e.g., score_pred)
    title: str,
    out_html: str,
):
    """
    Creates an interactive Plotly figure:
      - points are samples
      - line segments connect consecutive time points within each key
    """
    x, y = coords_2d[:, 0], coords_2d[:, 1]

    # group indices by key and order by time
    groups = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)
    for k in groups:
        groups[k].sort(key=lambda i: years[i])

    fig = go.Figure()

    # scatter points
    marker_kwargs = dict(size=7)
    if color_by is not None:
        marker_kwargs.update(dict(color=color_by, showscale=True))

    fig.add_trace(go.Scattergl(
        x=x, y=y,
        mode="markers",
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        marker=marker_kwargs,
        name="samples",
    ))

    # add line segments per key
    for k, idxs in groups.items():
        if len(idxs) < 2:
            continue
        # Build polyline via NaNs to break between groups
        xs, ys = [], []
        for j in range(len(idxs)):
            i = idxs[j]
            xs.append(x[i]); ys.append(y[i])
        # Add a trace per trajectory (keeps it performant and selectable)
        fig.add_trace(go.Scattergl(
            x=xs, y=ys,
            mode="lines+markers",
            line=dict(width=1),
            marker=dict(size=0),
            name=f"traj:{k}",
            hoverinfo="skip",
            opacity=0.4,
            showlegend=False,
        ))

    fig.update_layout(
        title=title,
        xaxis_title="dim-1",
        yaxis_title="dim-2",
        width=1000,
        height=800,
    )
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"Saved trajectory plot to: {out_html}")



def fo_visualize_multi(ds, methods=("umap", "pca"), seed=42):
    for m in methods:
        fob.compute_visualization(ds, embeddings="embedding", brain_key=f"{m}2d", method=m, seed=seed)

    if "sim_index" not in ds.list_brain_runs():
        fob.compute_similarity(ds, embeddings="embedding", brain_key="sim_index", metric="cosine")

    session = fo.launch_app(ds)
    print("Opened FiftyOne. In the Embeddings panel, pick 'umap2d' or 'pca2d'.")
    return session


from typing import Literal


def make_score_estimator(config, device="cuda"):
    """
    Returns a function that maps a per-head 'logits' tensor to a scalar score estimate.
    For 'classification_regression_mix', it uses the same mse_weight as in training.
    """
    task_type_y = config.get("task_type_y", "classification")

    # For the mix case we need the same weight used in training
    w_mse = 0.5
    try:
        loss_fn_dict = ra_utils.loss.loss_fn_dict.get_loss_fn_dict(config, device=device)
        w_mse = float(getattr(loss_fn_dict["y"], "mse_weight", w_mse))
    except Exception:
        pass  # fall back to 0.5 if loss dict cannot be constructed here

    if task_type_y == "regression":
        def score_estimator(logits: torch.Tensor) -> torch.Tensor:
            # logits: (N,1) -> (N,)
            return logits.squeeze(-1)

    elif task_type_y == "classification":
        def score_estimator(logits: torch.Tensor) -> torch.Tensor:
            # expectation over classes 0..C-1
            return ra_utils.networks.score_estimator.estimate_scalar_score_from_logits(
                logits, mode="expectation_value"
            )

    elif task_type_y == "classification_regression_mix":
        def score_estimator(logits: torch.Tensor) -> torch.Tensor:
            # logits: (N, 1 + C)  -> weighted combination (reg + class expectation)
            score_reg = logits[..., 0]
            score_cls = ra_utils.networks.score_estimator.estimate_scalar_score_from_logits(
                logits[..., 1:], mode="expectation_value"
            )
            return w_mse * score_reg + (1.0 - w_mse) * score_cls
    else:
        raise ValueError(f"Unknown task_type_y: {task_type_y}")

    return score_estimator

def _merge_multihead_outputs(
    head_out_dict: dict,                # {head_name: (idx, logits)}
    task_type_y: Literal["classification", "regression", "classification_regression_mix"],
    score_estimator,                    # function from make_score_estimator
    B: int,
    device: str = "cuda",
):
    """
    Returns per-sample tensors aligned to the full batch of size B:
        score_pred: (B,) float
        class_pred: (B,) int (proxy for reg-only heads)
        class_conf: (B,) float (NaN when not available)
        head_name:  list[str] length B ('' if none)
        raw_class_logits_available: bool
    """
    score_pred = torch.full((B,), float("nan"), device="cpu")
    class_pred = torch.full((B,), -1, dtype=torch.long, device="cpu")
    class_conf = torch.full((B,), float("nan"), device="cpu")
    head_name_per_sample = [""] * B

    raw_class_logits_available = False

    for hname, (idx, logits) in head_out_dict.items():
        if logits is None or idx.numel() == 0:
            continue
        # indices in this head's mini-batch -> global batch positions
        idx_cpu = idx.detach().cpu().long()
        logits = logits  # (n_h, C) or (n_h,1) or (n_h,1+C)

        # 1) scalar score per sample
        sc = score_estimator(logits).detach().cpu().float()  # (n_h,)
        score_pred[idx_cpu] = sc

        # 2) class view per sample
        if task_type_y == "classification":
            probs = torch.softmax(logits, dim=1)                             # (n_h, C)
            pred_idx = probs.argmax(1).detach().cpu().long()
            conf = probs[torch.arange(probs.size(0)), pred_idx].detach().cpu().float()
            class_pred[idx_cpu] = pred_idx
            class_conf[idx_cpu] = conf
            raw_class_logits_available = True

        elif task_type_y == "classification_regression_mix":
            probs = torch.softmax(logits[:, 1:], dim=1)                      # (n_h, C)
            pred_idx = probs.argmax(1).detach().cpu().long()
            conf = probs[torch.arange(probs.size(0)), pred_idx].detach().cpu().float()
            class_pred[idx_cpu] = pred_idx
            class_conf[idx_cpu] = conf
            raw_class_logits_available = True

        else:  # regression only: make a rounded proxy, no confidence
            # round the scalar score we just computed
            class_pred[idx_cpu] = torch.rint(sc).long()
            # leave class_conf as NaN

        for j in idx_cpu.tolist():
            head_name_per_sample[j] = hname

    return score_pred.numpy(), class_pred.numpy(), class_conf.numpy(), head_name_per_sample, raw_class_logits_available


def round_to_class(x: float, min_c: int = 0, max_c: Optional[int] = None) -> int:
    xi = int(np.rint(x))
    if max_c is not None:
        xi = int(np.clip(xi, min_c, max_c))
    else:
        xi = max(min_c, xi)
    return xi

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = ra_utils.utils.config_parser.load_config(
        # default_config="/home/cwatzenboeck/code/RA/ra_utils/runs/config_scoring/34_AHM/WRS_v02p00.yml", 
        default_config="/home/cwatzenboeck/code/RA/ra_utils/runs/config_scoring/36_all_but_wrist/ResNet_MSECESoft_MSE0.0_CE1.0_tLSR0.0_r01_DEBUGGING.yml", 
        debugging_in_jupyter_nb=False, silencium=True)

    # Load tables with paths and scores (+ split)
    data_tables = process_several_score_groups(config["data"])

    # # Make dataset and dataloaders
    data = dataset_and_loader_several(data_tables, config)



    # Load/ make model
    classifier_head_infos, attention_paths_dct, config = check_config_consistency_and_partially_make_consistent(config)
    model_name = config["model_name"]
    model_AE, model_c = build_models_AE_v1_and2(
                                        model_name, config, 
                                        classifier_head_infos = classifier_head_infos, 
                                        attention_paths_dct = attention_paths_dct
                                        )
    task_type_y = config.get("task_type_y", "classification")
    score_estimator = make_score_estimator(config, device)
   



    maybe_partially_init_model_from_state_dict(config, model_AE, model_c, 
                                               verbose=config.get("model_initialization", {}).get("verbosity_level", 3))
    model_AE.to(device)
    model_c.to(device)


    val_loaders = {k: data[k]["val_loader"] for k in data.keys()}
    train_loaders = {k: data[k]["train_loader"] for k in data.keys()}

    dl_key = config.get("plot_settings", {}).get("val_key", "ALL_wo_wrist")
    dl = val_loaders[dl_key]


    # Decide task type
    task_type_y = config.get("task_type_y", "classification")
    #classifier_name = config["model"].get("classifier", {}).get("name", "LogReg")
    #is_regression = (classifier_name == "Reg")

    # Where to store the post-transform PNGs
    png_root = config.get("plot_settings", {}).get("png_root", "/home/cwatzenboeck/data/fo_png_cache/val")

    # Extract embeddings + write PNGs
    max_class_per_type = None

    pack = extract_embeddings_and_pngs(
        model_AE=model_AE,
        model_c=model_c,
        loader=dl,
        device=device,
        score_estimator=score_estimator,
        task_type_y=task_type_y,
        png_root=png_root,
        max_class_per_type=max_class_per_type,
    )


    if False:  # Deactivate for now .... There are bigger issues... 
        # after you built `pack`:
        # 2D coords for BOTH UMAP + PCA (optional)
        coords_umap = compute_projection(pack["Z"], method="umap", seed=42)
        coords_pca  = compute_projection(pack["Z"], method="pca",  seed=42)

        # Make nice hover text
        hover = [
            f"key={k}<br>year={yr:.2f}<br>date={ds}<br>"
            f"score_pred={sp:.2f}<br>score_gt={sg:.2f}<br>type={st}"
            for k, yr, ds, sp, sg, st in zip(
                pack["patient_scoretype_key"],
                pack["years_since_2000"],
                pack["date_str"],
                pack["score_pred"],
                pack["score_gt"],
                pack["score_type"],
            )
        ]

        # Color points by score error or by year, up to you:
        color_by = np.abs(pack["score_pred"] - pack["score_gt"])

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_trajectories_html(
            coords_umap,
            years=pack["years_since_2000"],
            keys=pack["patient_scoretype_key"],
            hover_text=hover,
            color_by=color_by,
            title="UMAP trajectories (color: |pred-gt|)",
            out_html=f"/home/cwatzenboeck/data/fo_png_cache/plots/umap_trajs_{ts}.html",
        )
        plot_trajectories_html(
            coords_pca,
            years=pack["years_since_2000"],
            keys=pack["patient_scoretype_key"],
            hover_text=hover,
            color_by=pack["years_since_2000"],
            title="PCA trajectories (color: years since 2000)",
            out_html=f"/home/cwatzenboeck/data/fo_png_cache/plots/pca_trajs_{ts}.html",
        )

    # Build FO dataset + visualize
    ds_name = f"ra_val_embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ds = build_fiftyone_dataset(ds_name, pack, task_type_y)

    methods = tuple(config.get("plot_settings", {}).get("dimension_reduction_techniques", ["umap", "pca"]))
    fo_visualize_multi(ds, methods=methods)



if __name__ == "__main__":
    main()
