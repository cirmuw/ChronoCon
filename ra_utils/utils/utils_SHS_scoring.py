import numpy as np
import yaml
from importlib import resources


def get_classes(config):
    with resources.files("ra_utils.resources.scores_metadata").joinpath("score_abbreviations_info_dct_w_scores.yml").open("r") as f:
        score_abbreviations_info_dct = yaml.safe_load(f)
    chosen_score = config["data"]["scores"] 
    sum_wrist_points = config["data"]["sum_wrist_points"]
    if (chosen_score == ["LunatE", "RadiusE", "ScaphE", "TrapE", "UlnaE"]) and sum_wrist_points:
        print("HACK Assuming LunatE, RadiusE, ScaphE, TrapE, UlnaE -> 5*5+1 classes")
        return np.array([str(i) for i in range(26)])
    elif (chosen_score == ["Rad_Carp", "Sca_Cap", "Tra_Sca"]) and sum_wrist_points:
        print("HACK Assuming Rad_Carp, Sca_Cap, Tra_Sca -> 4*3+1 classes")
        return np.array([str(i) for i in range(13)])
    else:
        score_ranges = []
        first = True
        for score in chosen_score:
            if first:
                first = False
                score_ranges = score_abbreviations_info_dct[score]["scores"]
            else:
                assert score_ranges == score_abbreviations_info_dct[score][
                    "scores"], f"Score ranges do not match for {score} and {chosen_score[0]}"
        return score_ranges

