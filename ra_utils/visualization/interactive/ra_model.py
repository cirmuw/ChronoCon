# ==================== Standard Library ====================
import hashlib
import os
import random
import sys
from collections import Counter
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence

import yaml

# ==================== Third-Party Libraries ====================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader as TorchDataLoader, Dataset as TorchDataset, WeightedRandomSampler
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchvision import transforms
import torchvision.transforms.v2 as v2

import monai
import monai.networks.nets
from monai.data import DataLoader as MonaiDataLoader, Dataset as MonaiDataset
from monai.networks.nets import BasicUNet, UNet

from tqdm.notebook import tqdm
import umap

# Interactive embeddings (FiftyOne)
import fiftyone as fo
import fiftyone.core.labels as fol
from fiftyone import brain as fob

# ==================== Project / Local Imports ====================
import ra_utils
import ra_utils.utils.config_parser
import ra_utils.utils.utils

# Data loaders & helpers
import ra_utils.data.dataloader_CR_patches
from ra_utils.data.dataloader_CR_patches import (
    dataset_and_loader,
    dataset_and_loader_several,
    df_scores_to_dct_list,
    exclude_ROIS_according_surgery_status,
    load_img_SHS_patch_data,
    make_paths_dataframe,
    process_several_score_groups,
    process_single_score_group,
    restructure_paths_and_scores,
    restructure_paths_and_scores_v2,
    split_training_val_test__on_patient_level,
)

from ra_utils.data.data_utils import extract_extras_from_filename

# Networks / architectures
import ra_utils.networks.architecture
from ra_utils.networks.architecture import (
    EncoderClassifierNetwork,
    MultiModalImageScoreTypeNetwork,
    MultiModalImageScoreTypeNetworkAE,
    ResNet18Encoder,
    ResNet34Encoder,
    ResNet50Encoder,
    ROI_type_encoder,
    make_mlp,
    model_interface_forward,
)

from ra_utils.autoscora.autoscorRA_Pipeline.scoring.src.network import Custom_VGG
from ra_utils.utils.utils_SHS_scoring import get_classes

# MTAN / AE models


from ra_utils.training.scores_SHS.run_training_main_lib import (
    #check_config_consistency_and_partially_make_consistent,
    maybe_partially_init_model_from_state_dict,
)

# Losses
import ra_utils.networks.loss_function
from ra_utils.networks.loss_function import get_score_loss_function, get_triplet_loss_fn

import ra_utils.loss.consistency_regularizer_loss
import ra_utils.loss.loss_fn_dict
import ra_utils.loss.loss_utils
import ra_utils.loss.online_mining_triplet_loss
import ra_utils.loss.online_mining_triplet_loss_with_scores
from ra_utils.loss.online_mining_delta_loss import batch_all_score_differences_loss
import ra_utils.utils.multiprocessing

# Viz / analysis
import ra_utils.features.embeddings
import ra_utils.visualization.embeddings
from ra_utils.visualization.plot_image_series import (
    plot_patch_series,
    plot_triplet_dataloader_examples,
)
from ra_utils.visualization.structural_similarity_index import (
    compute_and_plot_ssi,
    ssim_matrix_torchmetrics,
)
from sklearn.decomposition import PCA
import umap as umap_pkg
import plotly.graph_objects as go
from collections import defaultdict
from sklearn.manifold import TSNE
from ra_utils.visualization.trajectories import (
    compute_2d_projection as traj_compute_2d_projection,
    save_projection as traj_save_projection,
    load_projection as traj_load_projection,
)

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


from ra_utils.utils.utils import datestr_to_years_since_2000
from ra_utils.utils.utils import datestr_to_years_since_2000


def compute_relative_time(years: np.ndarray, patient_scoretype_keys: np.ndarray) -> np.ndarray:
    """
    Compute relative time for each sample: (t-t_min)/(t_max - t_min + 1.0e-10)
    Grouped by patient_scoretype_key.
    
    Args:
        years: Array of years since 2000 for each sample
        patient_scoretype_keys: Array of patient_scoretype_key for each sample
        
    Returns:
        Array of relative times (0.0 to 1.0) for each sample
    """
    t_rel = np.zeros_like(years, dtype=float)
    
    # Group by patient_scoretype_key
    unique_keys = np.unique(patient_scoretype_keys)
    
    for key in unique_keys:
        mask = patient_scoretype_keys == key
        key_years = years[mask]
        
        if len(key_years) > 0:
            t_min = np.min(key_years)
            t_max = np.max(key_years)
            t_range = t_max - t_min + 1.0e-10  # Add small epsilon to avoid division by zero
            
            t_rel[mask] = (key_years - t_min) / t_range
    
    return t_rel

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
    if model_c is not None:
        model_c.eval()

    Z_list, png_paths = [], []
    score_gt, score_pred_all = [], []
    class_gt_all, class_pred_all, class_conf_all, max_score_difference_all = [], [], [], []
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
        s_type = batch["score_type"]             # list[str]
        s_type_np = np.array(s_type)

        # identity + time
        ps_keys = batch["patient_scoretype_key"]
        years_np = datestr_to_years_since_2000(batch["date_str"])
        ps_key_all.extend(ps_keys)
        years_all.extend(years_np.tolist())
        date_str_all.extend(list(batch["date_str"]))

        # forward
        _, z = model_AE(X, s_type)              # (B, D)

        # multi-head call (mirror training) - only if model_c is not None
        if model_c is not None:
            try:
                out_dict = model_c(z, s_type, return_dict=True)
            except TypeError:
                # fallback: single-head legacy path
                logits = model_c(z, s_type)         # (B, C) or (B,1) or (B,1+C)
                out_dict = {"__single__": (torch.arange(B, device=z.device), logits)}

            # merge per-head predictions into per-sample arrays
            score_pred, class_pred, class_conf, head_name_per_sample, has_logits = _merge_multihead_outputs(
                out_dict, task_type_y, score_estimator, B, device=device, s_type_np=s_type_np
            )
        else:
            # When model_c is None, skip predictions - no predictions to make
            head_name_per_sample = [""] * B

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
        if model_c is not None:
            score_pred_all.extend(score_pred.tolist())
            class_pred_all.extend(class_pred.tolist())
            class_conf_all.extend(class_conf.tolist())
        class_gt_all.extend(cls_gt.tolist())
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
        max_score_difference_all.extend(list(batch.get("score maxdiff", [None]*B)))
        roi_names.extend(batch.get("roi_name", [""] * B))
        patient_ids.extend(batch.get("patient_id", [""] * B))
        extremities.extend(batch.get("extremity", [""] * B))
        lr_sides.extend(batch.get("left_or_right", [""] * B))


    Z = np.concatenate(Z_list, axis=0)

    # Compute relative time
    years_array = np.array(years_all, dtype=float)
    ps_key_array = np.array(ps_key_all)
    t_rel = compute_relative_time(years_array, ps_key_array)

    result_dict = dict(
        Z=Z,
        display_paths=np.array(png_paths),

        # scalar view
        score_gt=np.array(score_gt, dtype=float),

        # classification view - ground truth is always available
        class_gt=np.array(class_gt_all, dtype=int),

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
        max_score_difference=np.array(max_score_difference_all),
        # time + identity
        patient_scoretype_key=np.array(ps_key_all),
        years_since_2000=np.array(years_all, dtype=float),
        date_str=np.array(date_str_all),
        t_rel=t_rel,
    )
    
    # Only include predictions if model_c is not None
    if model_c is not None:
        result_dict["score_pred"] = np.array(score_pred_all, dtype=float)
        result_dict["class_pred"] = np.array(class_pred_all, dtype=int)
        result_dict["class_conf"] = np.array(class_conf_all, dtype=float)
    
    return result_dict


def build_fiftyone_dataset(
    name: str,
    pack: dict,
    task_type_y: Literal["classification", "regression", "classification_regression_mix"],
    class_names: dict[int, str] | None = None,
):
    # make default class names if not provided
    if class_names is None:
        uniq = set(pack["class_gt"].tolist())
        if "class_pred" in pack:
            uniq = uniq | set(pack["class_pred"].tolist())
        class_names = {int(i): str(i) for i in sorted(uniq)}

    ds = fo.Dataset(name)
    samples = []
    for i in range(len(pack["display_paths"])):
        s = fo.Sample(filepath=str(pack["display_paths"][i]))

        # always store scalar view
        s["score_gt"]   = float(pack["score_gt"][i])
        # only store score_pred if it exists (i.e., model_c was not None)
        if "score_pred" in pack:
            s["score_pred"] = float(pack["score_pred"][i])

        # always store classification ground truth
        gt_label = class_names.get(int(pack["class_gt"][i]), str(int(pack["class_gt"][i])))
        s["class_gt"] = fol.Classification(label=gt_label)
        
        # only store classification predictions if they exist (i.e., model_c was not None)
        if "class_pred" in pack and "class_conf" in pack:
            pr_label = class_names.get(int(pack["class_pred"][i]), str(int(pack["class_pred"][i])))
            conf_val = pack["class_conf"][i]
            conf_val = None if (np.isnan(conf_val) or np.isinf(conf_val)) else np.round(float(conf_val), 2)

            s["class_pred"] = fol.Classification(label=pr_label, confidence=conf_val)
            if conf_val != None: 
                s["pred_confidence"] = str(conf_val)

            # convenience flags - only if predictions exist
            s["correct_cls"] = (gt_label == pr_label)
        
        # absolute error on scalar scores - only if score_pred exists
        if "score_pred" in pack:
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
        s["t_rel"]                 = float(pack["t_rel"][i])

        # store None as None (not strings) for numeric filters
        prev_v = pack["score_difference_prev_visit"][i]
        next_v = pack["score_difference_next_visit"][i]
        diff_v = pack["max_score_difference"][i]
        s["score_difference_prev_visit"] = None if prev_v is None else float(prev_v)
        s["score_difference_next_visit"] = None if next_v is None else float(next_v)
        #s["score_will_increase"] = None if next_v is None else int(next_v>0)
        #s["score_did_increase"] = None if next_v is None else int(next_v>0)
        s["max_score_difference"] = None if diff_v is None else float(diff_v)
        # embedding
        s["embedding"] = np.asarray(pack["Z"][i], dtype=np.float32)

        samples.append(s)

    ds.add_samples(samples)
    return ds



def compute_projection(Z: np.ndarray, method: str = "umap", seed: int = 42) -> np.ndarray:
    """
    Returns (N,2) projection for 'umap', 'pca', or 'tsne'.
    """
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(Z)
    elif method == "umap":
        reducer = umap_pkg.UMAP(n_components=2, random_state=seed)
        return reducer.fit_transform(Z)
    elif method == "tsne":
        return TSNE(n_components=2, random_state=seed, perplexity=min(30, max(5, Z.shape[0] // 10))).fit_transform(Z)
    else:
        raise ValueError(f"Unknown method: {method}")



def plot_trajectories_html(
    coords_2d: np.ndarray,           # (N,2)
    years: np.ndarray,               # (N,)
    keys: np.ndarray,                # (N,) patient_scoretype_key
    hover_text: list[str],           # per-point hover (e.g., score, date, score_type, etc.)
    color_by: np.ndarray | None,     # optional numeric to color points (e.g., score_pred)
    title: str,
    out_html: str,
    t_rel: np.ndarray | None = None, # optional relative time for coloring
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
    elif t_rel is not None:
        marker_kwargs.update(dict(color=t_rel, showscale=True, colorscale="Viridis"))

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



def fo_visualize_multi(ds, methods=("umap", "pca"), seed=42, port=None):
    for m in methods:
        fob.compute_visualization(ds, embeddings="embedding", brain_key=f"{m}2d", method=m, seed=seed)

    if "sim_index" not in ds.list_brain_runs():
        fob.compute_similarity(ds, embeddings="embedding", brain_key="sim_index", metric="cosine")

    session = fo.launch_app(ds, port=port)   # returns a Session
    #session.open_tab()            # optional: pop a browser tab
    print("Opened FiftyOne. In the Embeddings panel, pick 'umap2d' or 'pca2d'.")
    session.wait()                # <-- block so the session stays alive
    return session





def make_score_estimator(config, model_score_estimator=None, device="cuda"):
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
        def score_estimator(logits: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            # logits: (N,1) -> (N,)
            return logits.squeeze(-1)

    elif task_type_y == "classification":
        def score_estimator(logits: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            # expectation over classes 0..C-1
            return ra_utils.networks.score_estimator.estimate_scalar_score_from_logits(
                logits, mode="expectation_value"
            )

    elif task_type_y == "classification_regression_mix":
        if model_score_estimator is not None:
            model_score_estimator.eval().to(device)
            score_estimator = model_score_estimator 
        else: 
            def score_estimator(logits: torch.Tensor, *args, **kwargs) -> torch.Tensor:
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
    s_type_np = None, 
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

    # s_type = batch["score_type"]             # list[str]
    # s_type_np = np.array(s_type)

    raw_class_logits_available = False

    for hname, (idx, logits) in head_out_dict.items():
        if logits is None or idx.numel() == 0:
            continue
        # indices in this head's mini-batch -> global batch positions
        idx_cpu = idx.detach().cpu().long()
        logits = logits  # (n_h, C) or (n_h,1) or (n_h,1+C)

        s_type_head = s_type_np[idx.numpy(force=True)]

        # 1) scalar score per sample
        sc = score_estimator(logits, s_type_head).detach().cpu().float()  # (n_h,)
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
            class_pred[idx_cpu] = sc.round().long()
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

######################################################################
### Functions for reloading embedding

def save_pack(pack: dict, out_dir: str, png_root: str | None = None, save_projections: bool = True):
    """
    Saves:
      - pack_arrays.npz  (all np arrays from `pack`)
      - summary.parquet  (handy for quick peeks)
      - png_root.txt     (where your PNGs were written)
      - umap2d.npy / pca2d.npy (optional, precomputed 2D coords)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Save all arrays as .npz (keeps types, compact)
    np.savez_compressed(out / "pack_arrays.npz", **{
        k: v for k, v in pack.items()
        if isinstance(v, np.ndarray)
    })

    # 2) Save a small tabular summary (great for filtering in pandas)
    df_dict = {
        "display_paths": pack["display_paths"],
        "score_type": pack["score_type"],
        "roi_name": pack["roi_name"],
        "patient_id": pack["patient_id"],
        "extremity": pack["extremity"],
        "left_or_right": pack["left_or_right"],
        "score_gt": pack["score_gt"],
        "class_gt": pack["class_gt"],
        "patient_scoretype_key": pack["patient_scoretype_key"],
        "years_since_2000": pack["years_since_2000"],
        "date_str": pack["date_str"],
        "t_rel": pack["t_rel"],
    }
    # Only include predictions if they exist (i.e., model_c was not None)
    if "score_pred" in pack:
        df_dict["score_pred"] = pack["score_pred"]
    if "class_pred" in pack:
        df_dict["class_pred"] = pack["class_pred"]
    if "class_conf" in pack:
        df_dict["class_conf"] = pack["class_conf"]
    df = pd.DataFrame(df_dict)
    df.to_parquet(out / "summary.parquet", index=False)

    # 3) Remember PNG root (helps when moving between machines)
    if png_root is not None:
        (out / "png_root.txt").write_text(str(png_root))

    # 4) (Optional) precompute & store 2D projections so you can skip UMAP/PCA later
    if save_projections and "Z" in pack:
        try:
            coords_umap = compute_projection(pack["Z"], method="umap", seed=42)
            coords_pca  = compute_projection(pack["Z"], method="pca",  seed=42)
            np.save(out / "umap2d.npy", coords_umap)
            np.save(out / "pca2d.npy",  coords_pca)

            # Optionally write tsne2d.npy iff provided in the pack (avoid expensive compute here)
            if "tsne2d" in pack:
                np.save(out / "tsne2d.npy", np.asarray(pack["tsne2d"]))
        except Exception as e:
            print(f"[save_pack] Skipping projections: {e}")

    print(f"[save_pack] Wrote cache to: {out.resolve()}")


def save_or_compute_projection(pack: dict,
                               out_dir: str,
                               method: str = "umap",
                               seed: int = 42,
                               save_reducer: bool = False):
    """
    Convenience helper: load cached projection if present; otherwise compute and save.
    Returns (coords_2d, reducer_or_None).
    """
    try:
        coords, reducer = traj_load_projection(out_dir, method, load_reducer=save_reducer)
        print(f"[save_or_compute_projection] Loaded {method} from cache: {out_dir}")
        return coords, reducer
    except FileNotFoundError:
        pass

    coords, reducer = traj_compute_2d_projection(pack, reduction_method=method, seed=seed)
    traj_save_projection(coords, reducer, out_dir, method, save_reducer=save_reducer)
    print(f"[save_or_compute_projection] Computed and cached {method} to: {out_dir}")
    return coords, reducer


def load_pack(in_dir: str) -> dict:
    """
    Loads arrays from pack_arrays.npz and returns a `pack` dict
    with the same keys you used downstream.
    """
    p = Path(in_dir) / "pack_arrays.npz"
    data = np.load(p, allow_pickle=True)

    # Rebuild dict with explicit dtypes where needed
    pack = {k: data[k] for k in data.files}
    # Ensure a few expected dtypes
    for k in ("score_gt", "score_pred", "class_conf", "years_since_2000", "t_rel"):
        if k in pack: pack[k] = pack[k].astype(float)
    for k in ("class_gt", "class_pred"):
        if k in pack: pack[k] = pack[k].astype(int)
    # Strings stay strings; embeddings Z remains float
    return pack

def rebuild_fo_dataset_from_pack(
    name: str,
    pack: dict,
    task_type_y: Literal["classification", "regression", "classification_regression_mix"] = "classification",
    class_names: dict[int, str] | None = None,
    persistent: bool = True,
):
    if class_names is None:
        uniq = set(pack["class_gt"].tolist())
        if "class_pred" in pack:
            uniq = uniq | set(pack["class_pred"].tolist())
        class_names = {int(i): str(i) for i in sorted(uniq)}

    # If dataset exists, delete or load fresh
    if name in fo.list_datasets():
        fo.delete_dataset(name)

    ds = fo.Dataset(name, persistent=persistent)

    samples = []
    N = len(pack["display_paths"])
    for i in range(N):
        s = fo.Sample(filepath=str(pack["display_paths"][i]))

        # scalar view
        s["score_gt"]   = float(pack["score_gt"][i])
        # only store score_pred if it exists (i.e., model_c was not None)
        if "score_pred" in pack:
            s["score_pred"] = float(pack["score_pred"][i])

        # always store classification ground truth
        gt_label = class_names.get(int(pack["class_gt"][i]), str(int(pack["class_gt"][i])))
        s["class_gt"]  = fol.Classification(label=gt_label)
        
        # only store classification predictions if they exist (i.e., model_c was not None)
        if "class_pred" in pack and "class_conf" in pack:
            pr_label = class_names.get(int(pack["class_pred"][i]), str(int(pack["class_pred"][i])))
            conf = pack["class_conf"][i]
            conf = None if (np.isnan(conf) or np.isinf(conf)) else float(conf)

            s["class_pred"] = fol.Classification(label=pr_label, confidence=conf)
            if conf is not None:
                s["pred_confidence"] = conf

        # meta
        for fld in ("score_type","roi_name","patient_id","extremity","left_or_right"):
            s[fld] = str(pack[fld][i]) if len(pack[fld]) else ""

        s["patient_scoretype_key"] = str(pack["patient_scoretype_key"][i])
        s["years_since_2000"]      = float(pack["years_since_2000"][i])
        s["date_str"]              = str(pack["date_str"][i])
        s["t_rel"]                 = float(pack["t_rel"][i])

        # numeric deltas (may be None)
        pv = pack["score_difference_prev_visit"][i] if "score_difference_prev_visit" in pack else None
        nv = pack["score_difference_next_visit"][i] if "score_difference_next_visit" in pack else None
        dv = pack["max_score_difference"][i] if "max_score_difference" in pack else None
        s["score_difference_prev_visit"] = None if pv is None else float(pv)
        s["score_difference_next_visit"] = None if nv is None else float(nv)
        s["max_score_difference"] = None if dv is None else float(dv)

        # store the embedding
        s["embedding"] = np.asarray(pack["Z"][i], dtype=np.float32)

        samples.append(s)

    ds.add_samples(samples)
    return ds



import ra_utils.training.scores_SHS.run_training_main_lib 

def main():
    from pathlib import Path
    import numpy as np
    from fiftyone import brain as fob
    import fiftyone as fo
    ra_utils.utils.multiprocessing.set_multiprocessing_strategy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------- 0) Load config
    config = ra_utils.utils.config_parser.load_config(
        # default_config="/home/cwatzenboeck/code/RA/ra_utils/runs/config_scoring/development_inputs/training_confing.yml",
        # default_config="/msc/home/cwatze93/data/mlflow/mlflow_RA/976870858386409169/49f1b8541acc4565a227c82568d8bcb1/artifacts/original_config_latentspace_visualization_dev.yml",
        default_config="/msc/home/cwatze93/data/mlflow/mlflow_RAv2/377224905299293446/b4b349f4e5ba46658b78775246c39e84/artifacts/original_config_plotting.yml",
        debugging_in_jupyter_nb=False,
        silencium=True
    )

    ps = config.get("plot_settings", {}) or {}

    loader_type  = ps.get("loader_type", "val").lower()  # "val" or "train"
    split_key    = ps.get("val_key", "ALL_wo_wrist")     # used for train/val
    png_root     = ps.get("png_root", "/home/cwatzenboeck/data/fo_png_cache/val")

    cache_dir_save   = ps.get("cache_dir_save", None)
    cache_dir_reload = ps.get("cache_dir_reload", None)
    SAVE             = bool(ps.get("SAVE_EMBEDDINGS", False))
    RELOAD           = bool(ps.get("RELOAD_EMBEDDINGS", False))
    PORT             = ps.get("PORT", None)

    methods = tuple(ps.get("dimension_reduction_techniques", ["umap", "pca"]))

    # -------------------- 1) Either reload pack or compute it
    pack = None
    used_cache_dir = None

    if RELOAD:
        if not cache_dir_reload:
            raise ValueError("RELOAD_EMBEDDINGS=True but 'plot_settings.cache_dir_reload' is not set")
        used_cache_dir = Path(str(cache_dir_reload))
        print(f"[main] Reloading pack from: {used_cache_dir}")
        pack = load_pack(str(used_cache_dir))

    else:
        # --- Compute fresh embeddings (uses model + dataloader)
        # 1. Data tables + loaders
        data_tables = process_several_score_groups(config["data"])
        data = dataset_and_loader_several(data_tables, config)

        # 2. Choose split/loader according to config
        val_loaders   = {k: data[k]["val_loader"]   for k in data.keys() if "val_loader"   in data[k]}
        train_loaders = {k: data[k]["train_loader"] for k in data.keys() if "train_loader" in data[k]}

        if loader_type == "train":
            if split_key not in train_loaders:
                # fallback to any available train loader
                if len(train_loaders) == 0:
                    raise KeyError("No train loaders available")
                print(f"[main] split_key '{split_key}' not in train_loaders, using first available")
                dl = next(iter(train_loaders.values()))
            else:
                dl = train_loaders[split_key]
        else:  # "val"
            if split_key not in val_loaders:
                if len(val_loaders) == 0:
                    raise KeyError("No val loaders available")
                print(f"[main] split_key '{split_key}' not in val_loaders, using first available")
                dl = next(iter(val_loaders.values()))
            else:
                dl = val_loaders[split_key]

        # 3. Build models
        models, config = ra_utils.training.scores_SHS.run_training_main_lib.build_models_v3(config)
        model_AE, model_c,  model_score_estimator = models["model_AE"], models["model_c"], models["model_score_estimator"]

        if config.get("REMOVE_REGRESSION_OR_CLASSIFICATION_MODEL", False):
            print("Removing regression or classification model")
            model_c = None
            model_score_estimator = None

        task_type_y = config.get("task_type_y", "classification")
        score_estimator = make_score_estimator(config, model_score_estimator=model_score_estimator, device=device)

        maybe_partially_init_model_from_state_dict(
            config = config, 
            model_AE = model_AE, 
            model_c = model_c, 
            model_score_estimator=model_score_estimator,
            verbose=config.get("model_initialization", {}).get("verbosity_level", 3), 
            strict=True
        )
        model_AE.to(device).eval()
        if model_c is not None:
            model_c.to(device).eval()

        # 4. Extract embeddings (+ write PNGs)
        pack = extract_embeddings_and_pngs(
            model_AE=model_AE,
            model_c=model_c,
            loader=dl,
            device=device,
            score_estimator=score_estimator,
            task_type_y=task_type_y,
            png_root=png_root,
            max_class_per_type=None,
        )

        # 5. Save pack if requested
        if SAVE:
            if not cache_dir_save:
                print("[main] SAVE_EMBEDDINGS=True but 'plot_settings.cache_dir_save' is not set → skipping save")
            else:
                used_cache_dir = Path(str(cache_dir_save))
                print(f"[main] Saving pack to: {used_cache_dir}")
                save_pack(pack, str(used_cache_dir), png_root=png_root, save_projections=True)

    # -------------------- 2) Build FiftyOne dataset
    task_type_y = config.get("task_type_y", "classification")
    ds_suffix = f"{loader_type}_{split_key}"
    ds_name = f"ra_embeddings_{ds_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ds = build_fiftyone_dataset(ds_name, pack, task_type_y)

    fo_visualize_multi(ds, methods=methods, port=PORT)


if __name__ == "__main__":
    main()
