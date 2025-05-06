import numpy as np
import torch
import time
import os
import shutil
import sys
import argparse
import csv
from torch.utils.data import DataLoader

import ra_utils
import ra_utils.autoscora.autoscorRA_Pipeline.scoring.src

from ra_utils.autoscora.autoscorRA_Pipeline.scoring.src.io_scoring_method import io_scoring, mandatory_train_val_test_ids
from ra_utils.autoscora.autoscorRA_Pipeline.scoring.src.network import Custom_VGG
from ra_utils.autoscora.autoscorRA_Pipeline.scoring.src.load import get_indices4, load2, balance_train_set, balanced_mean, get_patient_nr, CustomDataset
from ra_utils.autoscora.autoscorRA_Pipeline.scoring.src.train import train, validate, test
from ra_utils.autoscora.autoscorRA_Pipeline.scoring.src.wrist import get_sum_score

import ra_utils
import ra_utils.utils.config_parser
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd


def paths_list_scores_list_from_score_types(chosen_score, chosen_score_type, extremity):
    path_list = []
    score_list = []
    for i in range(len(chosen_score)):
        path_list_tmp, score_list_tmp = io_scoring(chosen_score = chosen_score[i],
                                                    chosen_score_type = chosen_score_type,
                                                    extremity = extremity)
        path_list = path_list + path_list_tmp
        score_list = score_list + score_list_tmp

    if chosen_score == ["LunatE", "RadiusE", "ScaphE", "TrapE", "UlnaE"]:
        print("LunatE, RadiusE, ScaphE, TrapE, UlnaE:  Summing scores")
        path_list, score_list = get_sum_score(path_list, score_list)
    if chosen_score == ["Rad_Carp", "Sca_Cap", "Tra_Sca"]:
        path_list, score_list = get_sum_score(path_list, score_list)
    return path_list, score_list


from typing import Literal
def restructure_paths_and_scores(chosen_score: List[str], 
                                 chosen_score_type: str, 
                                 extremity: Literal["H", "F"], 
                                 sum_wrist= True,
                                 score_path_H = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/autoscoRA_TDEIMEL_HOME/autoscoRA_Preprocessing/output/data_split/pat_df_medstream_manual_H_dp_img_of_int_2_summary_cols_segm_sets_RL_stratmean_split6543_chosen345_2019-05-03_21-12-37.csv",
                                 score_path_F = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/autoscoRA_TDEIMEL_HOME/autoscoRA_Preprocessing/output/pat_df_manual_FEET/data_split/pat_df_medstream_manual_F_dp_img_of_int_2_summary_cols_segm_sets_RL_stratmean_split6543_chosen345_2021-05-25_13-45-01_fixedComTBD.csv"
                                 ):
    path_list = []
    score_list = []
    chosen_score_list = []
    for i in range(len(chosen_score)):
        path_list_tmp, score_list_tmp = io_scoring(chosen_score = chosen_score[i],
                                                    chosen_score_type = chosen_score_type,
                                                    extremity = extremity,
                                                    score_path_H = score_path_H, 
                                                    score_path_F = score_path_F)
        path_list = path_list + path_list_tmp
        score_list = score_list + score_list_tmp
        chosen_score_list = chosen_score_list + [chosen_score[i]] * len(path_list_tmp)

    if sum_wrist: 
        if chosen_score == ["LunatE", "RadiusE", "ScaphE", "TrapE", "UlnaE"]:
            print("LunatE, RadiusE, ScaphE, TrapE, UlnaE:  Summing scores")
            path_list, score_list = get_sum_score(path_list, score_list)
            chosen_score_list = ["+".join(["LunatE", "RadiusE", "ScaphE", "TrapE", "UlnaE"])] * len(path_list) # overwrite 
        if chosen_score == ["Rad_Carp", "Sca_Cap", "Tra_Sca"]:
            path_list, score_list = get_sum_score(path_list, score_list)
            chosen_score_list = ["+".join(["Rad_Carp", "Sca_Cap", "Tra_Sca"])] * len(path_list) # overwrite 
    


    data_dct = {"file_name": path_list, 
                "score": score_list, 
                "chosen_score": chosen_score_list, 
                "chosen_score_type": [chosen_score_type] * len(path_list),
                "extremity": [extremity] * len(path_list)}
    df = pd.DataFrame(data_dct)
    return df



def restructure_paths_and_scores_v2(
        chosen_scores: List[str], 
        score_path_H = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/autoscoRA_TDEIMEL_HOME/autoscoRA_Preprocessing/output/data_split/pat_df_medstream_manual_H_dp_img_of_int_2_summary_cols_segm_sets_RL_stratmean_split6543_chosen345_2019-05-03_21-12-37.csv",
        score_path_F = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/autoscoRA_TDEIMEL_HOME/autoscoRA_Preprocessing/output/pat_df_manual_FEET/data_split/pat_df_medstream_manual_F_dp_img_of_int_2_summary_cols_segm_sets_RL_stratmean_split6543_chosen345_2021-05-25_13-45-01_fixedComTBD.csv"
        ):
    
    import yaml
    from importlib import resources
    import pandas as pd
    import json
    with resources.files("ra_utils.resources.scores_metadata").joinpath("score_abbreviations_info_dct.yml").open("r") as f:
        score_abbreviations_info_dct = yaml.safe_load(f)

    with resources.files("ra_utils.resources.scores_metadata").joinpath("roi_scores_matching.csv").open("r") as f:
        df_score_roi_matching = pd.read_csv(f)
    score_2_roi_name_dct = df_score_roi_matching.set_index("score_name").transpose().to_dict()


    path_list = []
    score_list = []
    chosen_score_list = []
    extremities = []
    score_types = []
    roi_names = []
    for chosen_score in chosen_scores:
        extremity = score_abbreviations_info_dct[chosen_score]["extremity"]
        score_type = score_abbreviations_info_dct[chosen_score]["score_type"]


        path_list_tmp, score_list_tmp = io_scoring(chosen_score = chosen_score,
                                                    chosen_score_type = score_type,
                                                    extremity = extremity,
                                                    score_path_H = score_path_H, 
                                                    score_path_F = score_path_F)
        path_list = path_list + path_list_tmp
        score_list = score_list + score_list_tmp
        chosen_score_list = chosen_score_list + [chosen_score] * len(path_list_tmp)
        extremities = extremities + [extremity] * len(path_list_tmp)
        score_types = score_types + [score_type] * len(path_list_tmp)
        roi_names += [score_2_roi_name_dct[chosen_score]["ROI_name"]] * len(path_list_tmp)




    data_dct = {"file_name": path_list, 
                "score": score_list, 
                "chosen_score": chosen_score_list, 
                "chosen_score_type": score_types,
                "extremity": extremities,
                "roi_name": roi_names
                }
    df = pd.DataFrame(data_dct)
    return df




    # TODO: CW: Bin nicht ganz sicher ob summieren der scores für Handgelenke weiterhin so gemacht werden soll
    # if chosen_score == ["LunatE", "RadiusE", "ScaphE", "TrapE", "UlnaE"]:
    #     print("LunatE, RadiusE, ScaphE, TrapE, UlnaE:  Summing scores")
    #     path_list, score_list = get_sum_score(path_list, score_list)
    # if chosen_score == ["Rad_Carp", "Sca_Cap", "Tra_Sca"]:
    #     path_list, score_list = get_sum_score(path_list, score_list)
    #return path_list, score_list






