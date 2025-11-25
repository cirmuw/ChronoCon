"""
#### Goal: Calculate progression from predictions and ground truths

0) Input:  
artifact_path
set_type # test, valFinal, ... 



1) Load predictions/ labels for individual scores

3) sort by the time (years_since_2000) should be in column
4) calculate score difference. No self-difference and no double counting. 

types_to_consider # list of strings e.g. all hand scores  

id = ... "<score_type>_<patient_id>_<side>" # np array of strings of this type
diff_gt = y[:, None] - y[None, :]
diff_pred = y_pred[:, None] - y_pred[None, :]

mask_ids = (id[:, None] == id[None, :])  # same side, patient, joint 
mask_no_dc = np.tri(d=1)(np.ones_like(mask_ids))  # No self-difference and no double counting. (Note that are already sorted)

mask_type = np.array(df["score_type].isin(types_to_consider))
mask = (mask_ids & mask_no_dc & mask_type[:, None])


diff_gt = diff_gt[mask]
diff_pred = diff_pred[mask]



5) quantize_progression(dx: np.array, threshold=0.5): 
        return  not (((- threshold)  < dx ) &  (dx < threshold))

6) Calculate PPV, NPV, Sensitivity, Specivity


7) Optional save to artifacts/predictions




"""



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

import ra_utils.data.shap_sums
from pathlib import Path 
from ra_utils.utils.utils import datestr_to_years_since_2000


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


def read_npz_preds(run_path, data_partition="test") -> pd.DataFrame:    
    preds = f"{run_path}/artifacts/predictions/{data_partition}_/*.npz"
    npz_files = glob.glob(preds)
    assert len(npz_files) == 1, f"Expected exactly one npz file, found {len(npz_files)}"
    npz_file = npz_files[0]
    npz_file = np.load(npz_file, allow_pickle=True)

    
    left_or_right = npz_file["left_or_right"]
    patient_id = npz_file["patient_id"]
    score_type = npz_file["score_type"]
    y_true = npz_file["labels"]
    y_pred = npz_file["preds_float"]
        
    npz_file["date_str"]
    years_since_2000 = datestr_to_years_since_2000(npz_file["date_str"])    

    
    d = dict(
        left_or_right = left_or_right, 
        patient_id = patient_id, 
        score_type = score_type, 
        y_true = y_true, 
        y_pred = y_pred, 
        date_str = npz_file["date_str"], 
        years_since_2000 = years_since_2000,
        #ids = ids
        )
    return pd.DataFrame(d)

def make_id(patient_id, left_or_right, score_type) -> np.array:
    # Convert pandas Series to numpy arrays if necessary
    if hasattr(patient_id, 'to_numpy'):
        patient_id = patient_id.to_numpy()
    if hasattr(left_or_right, 'to_numpy'):
        left_or_right = left_or_right.to_numpy()
    if hasattr(score_type, 'to_numpy'):
        score_type = score_type.to_numpy()
    # Assert length match
    assert len(patient_id) == len(left_or_right) == len(score_type), (
        f"Mismatched lengths: patient_id={len(patient_id)}, left_or_right={len(left_or_right)}, score_type={len(score_type)}"
    )
    # Create ids as "<score_type>_<patient_id>_<side>"
    ids = np.core.defchararray.add(
        np.core.defchararray.add(score_type.astype(str), "_"),
        np.core.defchararray.add(patient_id.astype(str), "_")
    )
    ids = np.core.defchararray.add(ids, left_or_right.astype(str))
    return ids

def get_diff(y, ids):
    """
    Compute all pairwise differences y_i - y_j for samples that
    share the same id, without self-differences and without double counting.

    Parameters
    ----------
    y : array-like or pandas Series
        Target values (shape: [N])
    ids : array-like or pandas Series
        Identity labels (shape: [N])

    Returns
    -------
    diff_flat : np.ndarray
        1D array containing y_i - y_j for all pairs (i, j)
        with same id and j > i.
    """
    # Convert pandas Series or similar to numpy arrays
    if hasattr(y, "to_numpy"):
        y = y.to_numpy()
    else:
        y = np.asarray(y)

    if hasattr(ids, "to_numpy"):
        ids = ids.to_numpy()
    else:
        ids = np.asarray(ids)

    # Sanity check
    assert y.shape[0] == ids.shape[0], "y and ids must have same length"
    n = y.shape[0]

    # All pairwise differences
    diff = y[:, None] - y[None, :]   # shape (n, n)

    # Same-id mask
    mask_ids = (ids[:, None] == ids[None, :])  # shape (n, n)

    # Upper-triangular mask without diagonal: j > i
    # This avoids self-differences and double counting
    mask_no_dc = np.triu(np.ones((n, n), dtype=bool), k=1)

    # Combine: same id AND j > i
    mask = mask_ids & mask_no_dc

    # Return flattened differences for valid pairs
    diff = diff[mask]
    return diff

def quantize_progression(dx: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    dx = np.asarray(dx)
    #stable = ((-threshold) < dx) & (dx < threshold)
    #return ~stable
    return dx > threshold

def confusion_counts(y_true_bin, y_pred_bin):
    """
    Return TP, FP, TN, FN counts for binary arrays.
    """
    y_true_bin = np.asarray(y_true_bin).astype(bool)
    y_pred_bin = np.asarray(y_pred_bin).astype(bool)

    TP = np.sum((y_true_bin == True)  & (y_pred_bin == True))
    TN = np.sum((y_true_bin == False) & (y_pred_bin == False))
    FP = np.sum((y_true_bin == False) & (y_pred_bin == True))
    FN = np.sum((y_true_bin == True)  & (y_pred_bin == False))
    return TP, FP, TN, FN

def compute_progression_classification_metrics(df, threshold=0.5):
    results = {}
    overall_TP = overall_FP = overall_TN = overall_FN = 0

    for s_type in sorted(df["score_type"].unique()):
        m = df["score_type"] == s_type

        dy_true = get_diff(df[m]["y_true"], df[m]["id"])
        dy_pred = get_diff(df[m]["y_pred"], df[m]["id"])

        # Quantize
        yT = quantize_progression(dy_true, threshold=threshold)
        yP = quantize_progression(dy_pred, threshold=threshold)

        # Get counts
        TP, FP, TN, FN = confusion_counts(yT, yP)

        # Store metrics
        results[s_type] = compute_metrics_from_counts(TP, FP, TN, FN)

        # Accumulate for overall totals
        overall_TP += TP
        overall_FP += FP
        overall_TN += TN
        overall_FN += FN

    # Compute global (not macro averaged — pooled)
    results["overall"] = compute_metrics_from_counts(
        overall_TP, overall_FP, overall_TN, overall_FN
    )    

    return results


def compute_metrics_from_counts(TP, FP, TN, FN):
    """
    Compute PPV, NPV, Sensitivity, Specificity, and F1 score.
    Handles division-by-zero by returning np.nan.
    """

    PPV = TP / (TP + FP) if TP + FP > 0 else np.nan     # Precision
    NPV = TN / (TN + FN) if TN + FN > 0 else np.nan
    Sens = TP / (TP + FN) if TP + FN > 0 else np.nan     # Recall
    Spec = TN / (TN + FP) if TN + FP > 0 else np.nan

    # F1 = harmonic mean of precision & recall
    denom = (2 * TP + FP + FN)
    F1 = (2 * TP) / denom if denom > 0 else np.nan

    return dict(
        PPV=PPV,
        NPV=NPV,
        Sensitivity=Sens,
        Specificity=Spec,
        F1=F1,
        N_samples=TP + FP + TN + FN,
        P=TP + FN,
        N=TN + FP,
        TP=TP,
        FP=FP,
        TN=TN,
        FN=FN,
        Acc = (TP + TN) / (TP + TN + FP + FN),
        BalancedAcc = (Sens + Spec) / 2,
    )

def to_python_scalars(obj):
    """
    Recursively convert NumPy scalars/arrays to Python native types
    so the object becomes JSON-serializable.
    """
    if isinstance(obj, dict):
        return {k: to_python_scalars(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_python_scalars(x) for x in obj]

    if isinstance(obj, (np.generic,)):  # all numpy scalars
        return obj.item()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj

def main():
    parser = argparse.ArgumentParser(description='Plot and save true vs predicted summed scores')
    parser.add_argument(
        '--run_path', '-r', type=str, 
        required=False,  # <<<<
        default="/msc/home/cwatze93/data/mlflow/mlflow_RAv2/351704134836737587/8ee50b5d85944141ba4193c0a4c0a95d/",
        help='Path to the MLflow run directory'
    )

    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for quantizing progression. Options: 0.5 (default) or 0.25 or 0.75')

    parser.add_argument('--skip_ERO_ED_combination', action='store_true', default=False,
                        help='Skip ERO_ED combination. Options: True or False (default: False)')
    

    

    parser.add_argument('--data_partition', type=str, default="valFinal",
                        help='Data partition to use. Options: "valFinal" or "test" (default: "valFinal")')
    
    #--data_partition  valFinal  test
 
    args = parser.parse_args()
    run_path = args.run_path
    data_partition = args.data_partition
    threshold = args.threshold
    skip_ERO_ED_combination = args.skip_ERO_ED_combination is not None

    args = parser.parse_args()
    print("ARGS::")
    print(args)
    
    run_path = args.run_path

    # 1 Load data
    df = read_npz_preds(run_path, data_partition = data_partition)
     
     
    # 2 Combine ERO_H ED parts 
    if not skip_ERO_ED_combination:
        df = ra_utils.data.shap_sums.combine_ERO_ED(df)
    df = df.dropna().reset_index(drop=True)

    # 3 sort 
    df = df.sort_values(by="years_since_2000", ascending=False).reset_index(drop=True)

    df["id"] = make_id(df["patient_id"], df["left_or_right"], df["score_type"])
    
    results = compute_progression_classification_metrics(df, threshold=threshold)

    #print(results)
    print(results["overall"])

    # save results as json file
    import json
    os.makedirs(f"{run_path}/artifacts/metrics/{data_partition}", exist_ok=True)
        
    results_json = to_python_scalars(results)

    with open(f"{run_path}/artifacts/metrics/{data_partition}/progression_classification_metrics.json", "w") as f:
        json.dump(results_json, f, indent=2)
        
    print(f"Results saved to: \n    {run_path}/artifacts/metrics/{data_partition}/progression_classification_metrics.json")
    print("\n\n")

if __name__ == "__main__":
    main()
