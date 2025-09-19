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
from ra_utils.training.scores_SHS.run_training_main_lib import maybe_partially_init_model_from_state_dict

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
@torch.no_grad()
def extract_embeddings_and_pngs(model_AE, model_c, loader, device,
                                is_regression: bool,
                                png_root: str):
    model_AE.eval()
    model_c.eval()

    Z, png_paths, y_true_all, y_pred_all, conf_all = [], [], [], [], []
    score_types, roi_names, patient_ids, extremities, lr_sides = [], [], [], [], []

    for batch in loader:
        X = batch["img"].to(device)
        s_type = batch["score_type"]               # list[str]
        y = batch["score"]
        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()

        # forward -> embedding + logits
        _, z = model_AE(X, s_type)                 # (B, D)
        logits = model_c(z, s_type)                # (B, C) or (B, 1)

        # predictions
        if is_regression:
            yhat = logits.detach().cpu().view(-1).numpy()
            conf = np.full_like(yhat, np.nan)
        else:
            probs = torch.softmax(logits, dim=1)
            yhat_idx = probs.argmax(1)
            yhat = yhat_idx.detach().cpu().numpy()
            conf = probs[torch.arange(len(yhat_idx)), yhat_idx].detach().cpu().numpy()

        # ensure a PNG for each sample *after* transforms
        # we use your new 'image_path' field as the source-id
        src_paths = batch.get("image_path", None)
        if src_paths is None:
            raise KeyError("Batch missing 'image_path'. Please add it in your Dataset dicts.")

        for i in range(X.shape[0]):
            png_p = ensure_png_for_sample(X[i], src_paths[i], png_root)
            png_paths.append(png_p)

        # meta
        Z.append(z.detach().cpu().float().numpy())
        y_true_all.extend(list(y))
        y_pred_all.extend(list(yhat))
        conf_all.extend(list(conf))
        score_types.extend(list(s_type))

        roi_names.extend(batch.get("roi_name", [""] * X.shape[0]))
        patient_ids.extend(batch.get("patient_id", [""] * X.shape[0]))
        extremities.extend(batch.get("extremity", [""] * X.shape[0]))
        lr_sides.extend(batch.get("left_or_right", [""] * X.shape[0]))

    Z = np.concatenate(Z, axis=0)

    return dict(
        Z=Z,
        display_paths=np.array(png_paths),
        y_true=np.array(y_true_all),
        y_pred=np.array(y_pred_all),
        conf=np.array(conf_all),
        score_type=np.array(score_types),
        roi_name=np.array(roi_names),
        patient_id=np.array(patient_ids),
        extremity=np.array(extremities),
        left_or_right=np.array(lr_sides),
    )

# ---------- FiftyOne dataset ----------
def build_fiftyone_dataset(name, pack, is_regression, class_names=None):
    if not is_regression and class_names is None:
        uniq = sorted(set(pack["y_true"].tolist()) | set(pack["y_pred"].tolist()))
        class_names = {int(i): str(i) for i in uniq}

    ds = fo.Dataset(name)
    samples = []
    for i in range(len(pack["display_paths"])):
        s = fo.Sample(filepath=str(pack["display_paths"][i]))

        if is_regression:
            s["ground_truth"] = float(pack["y_true"][i])
            s["prediction"]   = float(pack["y_pred"][i])
            s["wrong"] = None
            s["confidence"] = None
        else:
            gt = class_names.get(int(pack["y_true"][i]), str(pack["y_true"][i]))
            pr = class_names.get(int(pack["y_pred"][i]), str(pack["y_pred"][i]))
            s["ground_truth"] = fol.Classification(label=gt)
            s["prediction"]   = fol.Classification(
                label=pr,
                confidence=None if np.isnan(pack["conf"][i]) else float(pack["conf"][i])
            )
            s["wrong"] = (str(pr) != str(gt))
            s["confidence"] = None if np.isnan(pack["conf"][i]) else float(pack["conf"][i])

        s["score_type"]    = str(pack["score_type"][i])
        s["roi_name"]      = str(pack["roi_name"][i]) if len(pack["roi_name"]) else ""
        s["patient_id"]    = str(pack["patient_id"][i]) if len(pack["patient_id"]) else ""
        s["extremity"]     = str(pack["extremity"][i]) if len(pack["extremity"]) else ""
        s["left_or_right"] = str(pack["left_or_right"][i]) if len(pack["left_or_right"]) else ""
        s["embedding"]     = np.asarray(pack["Z"][i], dtype=np.float32)

        samples.append(s)

    ds.add_samples(samples)
    return ds

def fo_visualize(ds, brain_key="umap2d", method="umap"):
    fob.compute_visualization(ds, embeddings="embedding", brain_key=brain_key, method=method, seed=42)
    fob.compute_similarity(ds, embeddings="embedding", brain_key="sim_index", metric="cosine")
    session = fo.launch_app(ds)
    print("Opened FiftyOne. Use the Embeddings panel and select your run.")
    return session





def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = ra_utils.utils.config_parser.load_config(
        default_config="/home/cwatzenboeck/code/RA/ra_utils/runs/config_scoring/34_AHM/WRS_v02p00.yml", 
        debugging_in_jupyter_nb=True, silencium=True)

    # Load tables with paths and scores (+ split)
    data_tables = process_several_score_groups(config["data"])

    # # Make dataset and dataloaders
    data = dataset_and_loader_several(data_tables, config)



    # Load/ make model
    attention_paths_dct = config["data"].get("network_score_groups")
    if attention_paths_dct is None:
        print("'network_score_groups' not found in config file. Using default attention paths: 'score_groups'")
        attention_paths_dct = config["data"]["score_groups"]

    classifier_head_infos = config["data"]["classifier_head_infos"]
    if config["data"].get("how_to_deal_with_surgery")  == "keep: map over limit to limit plus one":
        print("Adding one to the output dimension of the classifier heads! (surgery class)")
        for k in classifier_head_infos.keys():
            v = classifier_head_infos[k]["out_dim"]
            classifier_head_infos[k]["out_dim"] = v + 1

    # If pure regression -> output-dim has to be 1
    classifier_name = config["model"].get("classifier", {}).get("name", "LogReg")
    print(f"{classifier_name = }")
    if classifier_name == "Reg":
        for k in classifier_head_infos.keys():
            classifier_head_infos[k]["out_dim"] = 1

    model_name = config["model_name"]
    model_AE, model_c = build_models_AE_v1_and2(
                                        model_name, config, 
                                        classifier_head_infos = classifier_head_infos, 
                                        attention_paths_dct = attention_paths_dct
                                        )
    model_c.to(device)
    model_AE.to(device)
    maybe_partially_init_model_from_state_dict(config, model_AE, model_c, verbose=3)



    val_loaders = {k: data[k]["val_loader"] for k in data.keys()}
    train_loaders = {k: data[k]["train_loader"] for k in data.keys()}


    dl = val_loaders["H_JSN_PIPII"]


    # Decide task type
    classifier_name = config["model"].get("classifier", {}).get("name", "LogReg")
    is_regression = (classifier_name == "Reg")

    # Where to store the post-transform PNGs
    png_root = "/home/cwatzenboeck/data/fo_png_cache/val"   # choose your folder

    # Extract embeddings + write PNGs
    pack = extract_embeddings_and_pngs(
        model_AE=model_AE,
        model_c=model_c,
        loader=dl,              # e.g., val_loaders["H_JSN_PIPII"]
        device=device,
        is_regression=is_regression,
        png_root=png_root,
    )

    # Build FO dataset + visualize
    ds_name = f"ra_val_embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ds = build_fiftyone_dataset(ds_name, pack, is_regression)
    fo_visualize(ds, brain_key="umap2d", method="umap")




if __name__ == "__main__":
    main()
