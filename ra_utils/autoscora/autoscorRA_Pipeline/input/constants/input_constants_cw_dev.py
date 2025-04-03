import os
import datetime
import pandas as pd
import numpy as np
# from copy import copy, deepcopy
import re

print("running input_constants.py at " + str(datetime.datetime.now()))

RUN_LOCALLY = True

if RUN_LOCALLY:
    #HOME_DIR_SEP = "/mnthome2" + os.sep
    HOME_DIR_SEP = "/home/cwatzenboeck/data/AutoPIX_cirdata/" + os.sep  
    TDEIMEL_HOME_AUTOSCORA_SEP = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/autoscoRA_TDEIMEL_HOME"  + os.sep 
    ANNOTAION_DIR_SEP = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/autoscoRA_Annotation" + os.sep 
    PROJECT_DIR_SEP = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora" + os.sep
    MATPLOTLIB_BACKEND = 'TkAgg'
else:
    HOME_DIR_SEP = "/home/cir/tdeimel" + os.sep
    PROJECT_DIR_SEP = "/project" + os.sep
    MATPLOTLIB_BACKEND = 'agg'

# img hands
IMAGE_DIR = "/home/cwatzenboeck/data/tmp/dev_autoscora/H"

# IMAGE_DIR = PROJECT_DIR_SEP + \
#              "autoscoRA_images/H_images_of_interest_2_renamed_mirrored_inverted_dicoms"
ARRAY_DIR = PROJECT_DIR_SEP + \
             "autoscoRA_images/H_images_of_interest_2_renamed_mirrored_inverted_arrays"
# PATCH_DIR = PROJECT_DIR_SEP + \
#              "autoscoRA_images/H_images_of_interest_2_renamed_mirrored_inverted_patches"
PATCH_DIR = "/home/cwatzenboeck/data/tmp/dev_autoscora/out_patches_H"

AUGM_PATCH_DIR = PROJECT_DIR_SEP + \
                 "autoscoRA_images/H_images_of_interest_2_renamed_mirrored_inverted_augmented_patches"
CV_PATCH_DIR = PROJECT_DIR_SEP + \
            "autoscoRA_images/H_CV_images_of_interest_2_renamed_mirrored_inverted_patches"
CV_AUGM_PATCH_DIR = PROJECT_DIR_SEP + \
                 "autoscoRA_images/H_CV_images_of_interest_2_renamed_mirrored_inverted_augmented_patches"
FAIL_DIR = PROJECT_DIR_SEP + "autoscoRA_images/failures"

# img feet
F_IMAGE_DIR = PROJECT_DIR_SEP + \
              "autoscoRA_images/F_images_of_interest_2_renamed_mirrored_inverted_dicoms"
F_ARRAY_DIR = PROJECT_DIR_SEP + \
              "autoscoRA_images/F_images_of_interest_2_renamed_mirrored_inverted_arrays"
F_PATCH_DIR = "/home/cwatzenboeck/data/tmp/dev_autoscora/out_patches_F"
# F_PATCH_DIR = PROJECT_DIR_SEP + \
#               "autoscoRA_images/F_images_of_interest_2_renamed_mirrored_inverted_patches"
F_AUGM_PATCH_DIR = PROJECT_DIR_SEP + \
                   "autoscoRA_images/F_images_of_interest_2_renamed_mirrored_inverted_augmented_patches"
F_CV_PATCH_DIR = PROJECT_DIR_SEP + \
              "autoscoRA_images/F_CV_images_of_interest_2_renamed_mirrored_inverted_patches"
F_CV_AUGM_PATCH_DIR = PROJECT_DIR_SEP + \
                   "autoscoRA_images/F_CV_images_of_interest_2_renamed_mirrored_inverted_augmented_patches"
F_FAIL_DIR = PROJECT_DIR_SEP + "autoscoRA_images/F_failures"

# joints and rois gt
JOINTS_PATH_GT_100 = ANNOTAION_DIR_SEP + "output/annotations/" \
                                "hands_combined/all_hand_joints_100_onlynecessaryjoints.csv"

ROIS_PATH_GT_100 = ANNOTAION_DIR_SEP + "output/annotations/" \
                              "hands_combined/all_hand_rois_100.csv"

F_JOINTS_PATH_GT_100 = ANNOTAION_DIR_SEP + "output/annotations/" \
                                      "feet/all_foot_joints_100.csv"

F_ROIS_PATH_GT_100 = ANNOTAION_DIR_SEP + "output/annotations/" \
                                    "feet/all_foot_rois_100.csv"

# for evaluating localization accuracy
JOINTS_PATH_VAL_30 = ANNOTAION_DIR_SEP + "output/annotations/hands_combined/" \
                                    "SCN_output/all_hand_joints_train70_pred30_onlynecessaryjoints_nodec.csv"

F_JOINTS_PATH_VAL_30 = ANNOTAION_DIR_SEP + "output/annotations/feet/" \
                                    "SCN_output/all_feet_joints_train70_pred30_nodec.csv"

# every predicted joint coord img except those in JOINTS_PATH_GT_100
# JOINTS_PATH_PRED_REST100 = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/landmark_data/output/H_coordinates__bb72a1aaf7f9476885d559d7baba9b8e_DB_renamed_transposed.csv"
JOINTS_PATH_PRED_REST100 = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/landmark_data/output/H_coordinates__bb72a1aaf7f9476885d559d7baba9b8e_DB_with_lm_names.csv"
# old: /home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/autoscoRA_Annotation/output/annotations/hands_combined/SCN_output/all_hand_joints_train100_predrest_onlynecessaryjoints_nodec.csv"
# JOINTS_PATH_PRED_REST100 = ANNOTAION_DIR_SEP + "output/annotations/hands_combined/" \
#                                           "SCN_output/all_hand_joints_train100_predrest_onlynecessaryjoints_nodec.csv"

F_JOINTS_PATH_PRED_REST100 = ANNOTAION_DIR_SEP + "output/annotations/feet/" \
                                          "SCN_output/all_feet_joints_train100_predrest_nodec.csv"

# joint localization cross validation
JOINTS_PATH_100_SPLIT0 = ANNOTAION_DIR_SEP + "output/annotations/crossval/" \
                                        "SCN_output/all_hand_joints_100img_split0_onlynecessaryjoints_nodec.csv"
JOINTS_PATH_100_SPLIT1 = ANNOTAION_DIR_SEP + "output/annotations/crossval/" \
                                        "SCN_output/all_hand_joints_100img_split1_onlynecessaryjoints_nodec.csv"
JOINTS_PATH_100_SPLIT2 = ANNOTAION_DIR_SEP + "output/annotations/crossval/" \
                                        "SCN_output/all_hand_joints_100img_split2_onlynecessaryjoints_nodec.csv"

F_JOINTS_PATH_100_SPLIT0 = ANNOTAION_DIR_SEP + "output/annotations/crossval/" \
                                          "SCN_output/all_feet_joints_100img_split0_nodec.csv"
F_JOINTS_PATH_100_SPLIT1 = ANNOTAION_DIR_SEP + "output/annotations/crossval/" \
                                          "SCN_output/all_feet_joints_100img_split1_nodec.csv"
F_JOINTS_PATH_100_SPLIT2 = ANNOTAION_DIR_SEP + "output/annotations/crossval/" \
                                          "SCN_output/all_feet_joints_100img_split2_nodec.csv"

# pixel spacing
PIXEL_SPACING = TDEIMEL_HOME_AUTOSCORA_SEP + "autoscoRA_Pipeline/output/loc_and_segm_output/" \
                               "localization_cv_splits/H_100img_mm_per_px.csv"

F_PIXEL_SPACING = TDEIMEL_HOME_AUTOSCORA_SEP + "autoscoRA_Pipeline/output/loc_and_segm_output/" \
                                 "localization_cv_splits/F_100img_mm_per_px.csv"

# parameters from patch_fit
JOINT_TO_ROIS_PARAMETERS_PATH = TDEIMEL_HOME_AUTOSCORA_SEP + \
                                "autoscoRA_Pipeline/output/loc_and_segm_output/patch_fit_out/" \
                                "chosen_patch_fit/patch_fit_hand_plotannotFalse_2022-06-29_10-24-22/parameters.json"
# OLD 2: "archive_chosen_patch_fit/patch_fit_hand_plotannotTrue_2022-02-07_13-43-41/parameters.json"
# OLD: "patch_fit_out/patch_fit_2021-04-05_13-35-53/parameters.json"
CV_SPLIT_DIR = TDEIMEL_HOME_AUTOSCORA_SEP  + \
               "autoscoRA_Pipeline/output/loc_and_segm_output/localization_cv_splits/" \
               "cv_splits_hand_2022-06-26_09-55-48_H_before_F"
CV_JOINT_TO_ROIS_PARAMETERS_PATH = TDEIMEL_HOME_AUTOSCORA_SEP + \
                                "autoscoRA_Pipeline/output/loc_and_segm_output/patch_fit_out/" \
                                "chosen_patch_fit/patch_fit_hand_plotannotFalse_2022-06-29_10-24-22/" \
                                "cv_parameters_by_split.json"

F_JOINT_TO_ROIS_PARAMETERS_PATH = TDEIMEL_HOME_AUTOSCORA_SEP + \
                                  "autoscoRA_Pipeline/output/loc_and_segm_output/patch_fit_out/" \
                                  "chosen_patch_fit/patch_fit_foot_plotannotFalse_2022-06-29_10-24-22/parameters.json"
# OLD: "patch_fit_out/patch_fit_foot_plotannotFalse_2021-11-02_11-47-26/parameters.json"
F_CV_SPLIT_DIR = TDEIMEL_HOME_AUTOSCORA_SEP + \
                 "autoscoRA_Pipeline/output/loc_and_segm_output/localization_cv_splits/" \
                 "cv_splits_foot_2022-06-26_09-55-48_H_before_F"
F_CV_JOINT_TO_ROIS_PARAMETERS_PATH = TDEIMEL_HOME_AUTOSCORA_SEP  + \
                                  "autoscoRA_Pipeline/output/loc_and_segm_output/patch_fit_out/" \
                                  "chosen_patch_fit/patch_fit_foot_plotannotFalse_2022-06-29_10-24-22/" \
                                  "cv_parameters_by_split.json"

# scores
# pat_df_manual
PREPROCESSING_OUTPUT_DIR = TDEIMEL_HOME_AUTOSCORA_SEP + "autoscoRA_Preprocessing/output"
PAT_DF_MANUAL_VERSION = \
    "data_split" + os.sep + \
    "pat_df_medstream_manual_H_dp_img_of_int_2_" \
    "summary_cols_segm_sets_RL_stratmean_split6543_chosen345_2019-05-03_21-12-37.csv"
PAT_DF_MANUAL_PATH = PREPROCESSING_OUTPUT_DIR + os.sep + PAT_DF_MANUAL_VERSION

F_PAT_DF_MANUAL_VERSION = \
    "pat_df_manual_FEET/data_split/" \
    "pat_df_medstream_manual_F_dp_img_of_int_2_" \
    "summary_cols_segm_sets_RL_stratmean_split6543_chosen345_2021-05-25_13-45-01_fixedComTBD.csv"
F_PAT_DF_MANUAL_PATH = PREPROCESSING_OUTPUT_DIR + os.sep + F_PAT_DF_MANUAL_VERSION

H_F_DOUBLE_SCORE_PATH = PREPROCESSING_OUTPUT_DIR + os.sep + \
                        "double_scoring/Re_Scoring_50_v2_FINAL_pseudonymized_colrenamed.csv"

# datasets
PAT_TRAIN_PATH = TDEIMEL_HOME_AUTOSCORA_SEP + "autoscoRA_Pipeline/evaluation/figures/table_one/train_patients.npy"
PAT_VAL_PATH = TDEIMEL_HOME_AUTOSCORA_SEP + "autoscoRA_Pipeline/evaluation/figures/table_one/val_patients.npy"
PAT_TEST_PATH = TDEIMEL_HOME_AUTOSCORA_SEP + "autoscoRA_Pipeline/evaluation/figures/table_one/test_patients.npy"

"""LEGENDE

PATCHES/ROIs
SWR = wrist
SCD# = #.Finger Carpometacarpalgelenk, distaler Anteil
SMP# = #.Finger metacarpophalangealgelenk, proximaler Anteil
SMD# = #.Finger metacarpophalangealgelenk, distaler Anteil
SPP# = #.Finger proximales Interphalangealgelenk, proximaler Anteil

SCORES


NETWORKS
PIP I-V erosion
PIP II-V JSN
MCP I-V erosion
MCP I-V JSN
CMC I erosion
CMC III-V JSN
wrist erosion
wrist JSN

"""

# end of script
