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


def main():
    FRACTION_REQUIRED_VALID_SCORES = 0.75

    parser = argparse.ArgumentParser(description='Plot and save true vs predicted summed scores')
    parser.add_argument(
        '--run_path', type=str, 
        required=True,
        default="/msc/home/cwatze93/data/mlflow/mlflow_RA/844424495910332051/07e62b6853554a9a911d916113bcd168",
        help='Path to the MLflow run directory'
    )
    # soft_prediction becomes a simple flag: absent=False, present=True
    parser.add_argument(
        '--soft-prediction', dest='soft_prediction',
        action='store_true', 
        default=False,
        help='Use the soft sum Σ_c p̂_c * c label (default: False)'
    )
    # strict supports --strict / --no-strict
    parser.add_argument(
        '--strict', action=argparse.BooleanOptionalAction, 
        default=True,
        help='Assert output folder does not exist (default: True). Use --no-strict to allow overwrite.'
    )
    parser.add_argument(
        '--limit_treatment_ED', type=str, default="E_D_mean",
        help='How to handle ED score pairs. Options: "cap E_D sum to 5" or "E_D_mean"'
    )

    args = parser.parse_args()
    run_path = args.run_path
    soft_prediction = args.soft_prediction
    limit_treatment_ED = args.limit_treatment_ED
    strict = args.strict

    

    args = parser.parse_args()
    print("ARGS::")
    print(args)
    
    run_path = args.run_path
    soft_prediction = args.soft_prediction
    


    # Locate predictions
    preds = f"{run_path}/artifacts/predictions/valFinal_/*.npz"
    npz_files = glob.glob(preds)
    assert len(npz_files) == 1, f"Expected exactly one npz file, found {len(npz_files)}"
    npz_file = npz_files[0]


    # Create the output folder
    output_folder = f"{run_path}/artifacts/plots/00_summed_scores/valFinal/"
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
    fig, axis_tuple = plot_SHS_deltas(df_delta_SvH.dropna(), figsize=(5,5), name="ΔSvH H+F", regplot=True)


    fig.savefig(dst_score_progression)
    print(f" Saved figure to {dst_score_progression}")

    #plt.show()




    # Combined plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # Recreate plots directly into subplots
    _ = plot_SHS_sums(df_summed_SvH.dropna(),    ax_scatter=axs[0], name="SvH H+F", plot_histograms=False, regplot=True)
    _ = plot_SHS_sums(df_summed_H_F_JSN.dropna(), ax_scatter=axs[1], name="JSN H+F", plot_histograms=False, regplot=True)
    _ = plot_SHS_sums(df_summed_ERO_H_F.dropna(), ax_scatter=axs[2], name="ERO H+F", plot_histograms=False, regplot=True)

    fig.tight_layout()


    fig.savefig(dst_sums_plot)
    print(f" Saved figure to {dst_sums_plot}")


    #plt.show()
    print("DONE!!!")

if __name__ == "__main__":
    main()
