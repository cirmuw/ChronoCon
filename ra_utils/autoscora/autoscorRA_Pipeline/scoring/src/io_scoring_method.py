import re
import os
import pandas as pd
import numpy as np
import yaml


def io_scoring(chosen_score, chosen_score_type, extremity="H"):
    """
    :param chosen_score: e.g., chosen_score = "PIPIIIED"
    :param chosen_score_type: e.g., chosen_score_type = 'ERO'  # or detect automatically
    :param extremity: str, "H" = hand, "F" = foot
    :return: list of paths to patches, list of corresponding scores
    """

    # score_type --> score_name --> roi
    if extremity == "H":
        scor_roi_matching_dict = {'JSN': {'CMCIII': ['SCD3'],
                                          'CMCIV': ['SCD4'],
                                          'CMCV': ['SCD5'],
                                          'MCPIII': ['SMD3'],
                                          'MCPII': ['SMD2'],
                                          'MCPI': ['SMD1'],
                                          'MCPIV': ['SMD4'],
                                          'MCPV': ['SMD5'],
                                          'PIPIII': ['SPD3'],
                                          'PIPII': ['SPD2'],
                                          'PIPIV': ['SPD4'],
                                          'PIPV': ['SPD5'],
                                          'Rad_Carp': ['SWR'],
                                          'Sca_Cap': ['SWR'],
                                          'Tra_Sca': ['SWR']},
                                  'ERO': {'Base_MCIE': ['SCD1'],
                                          'IPIED': ['SPD1'],
                                          'IPIEP': ['SPP1'],
                                          'LunatE': ['SWR'],
                                          'MCPIIIED': ['SMD3'],
                                          'MCPIIIEP': ['SMP3'],
                                          'MCPIIED': ['SMD2'],
                                          'MCPIIEP': ['SMP2'],
                                          'MCPIED': ['SMD1'],
                                          'MCPIEP': ['SMP1'],
                                          'MCPIVED': ['SMD4'],
                                          'MCPIVEP': ['SMP4'],
                                          'MCPVED': ['SMD5'],
                                          'MCPVEP': ['SMP5'],
                                          'PIPIIIED': ['SPD3'],
                                          'PIPIIIEP': ['SPP3'],
                                          'PIPIIED': ['SPD2'],
                                          'PIPIIEP': ['SPP2'],
                                          'PIPIVED': ['SPD4'],
                                          'PIPIVEP': ['SPP4'],
                                          'PIPVED': ['SPD5'],
                                          'PIPVEP': ['SPP5'],
                                          'RadiusE': ['SWR'],
                                          'ScaphE': ['SWR'],
                                          'TrapE': ['SWR'],
                                          'UlnaE': ['SWR']}
                                  }
    elif extremity == "F":
        scor_roi_matching_dict = {'JSN': {'MTPI': ['STD1'],
                                          'MTPII': ['STD2'],
                                          'MTPIII': ['STD3'],
                                          'MTPIV': ['STD4'],
                                          'MTPV': ['STD5'],
                                          'IP': ['SID1']},
                                  'ERO': {'MTPIEP': ['STP1'],
                                          'MTPIED': ['STD1'],
                                          'MTPIIEP': ['STP2'],
                                          'MTPIIED': ['STD2'],
                                          'MTPIIIEP': ['STP3'],
                                          'MTPIIIED': ['STD3'],
                                          'MTPIVEP': ['STP4'],
                                          'MTPIVED': ['STD4'],
                                          'MTPVEP': ['STP5'],
                                          'MTPVED': ['STD5'],
                                          'IPEP': ['SIP1'],
                                          'IPED': ['SID1']}
                                  }
    else:
        raise ValueError("extremity must bei either 'H' or 'F'")

    # roi --> score_type --> score_name
    roi_scor_matching_dict = {roi: {'JSN': [], 'ERO': []}
                              for roi in list(set([roi_name for score_type, v0 in scor_roi_matching_dict.items()
                                                   for score_name, v1 in v0.items()
                                                   for roi_name in v1]))}

    for roi_i, roi_dict_i in roi_scor_matching_dict.items():
        for score_type in roi_dict_i.keys():
            roi_scor_matching_dict[roi_i][score_type] = \
                [joint for joint, joint_rois in scor_roi_matching_dict[score_type].items()
                 if roi_i in joint_rois]

    # read in scores dataframe
    # score joint names
    r_neutral_hand_joints = ["r_Base_MCIE",
                             "r_CMCIII",
                             "r_CMCIV",
                             "r_CMCV",
                             "r_IPIED",
                             "r_IPIEP",
                             "r_LunatE",
                             "r_MCPIII",
                             "r_MCPIIIED",
                             "r_MCPIIIEP",
                             "r_MCPII",
                             "r_MCPIIED",
                             "r_MCPIIEP",
                             "r_MCPI",
                             "r_MCPIED",
                             "r_MCPIEP",
                             "r_MCPIV",
                             "r_MCPIVED",
                             "r_MCPIVEP",
                             "r_MCPV",
                             "r_MCPVED",
                             "r_MCPVEP",
                             "r_PIPIII",
                             "r_PIPIIIED",
                             "r_PIPIIIEP",
                             "r_PIPII",
                             "r_PIPIIED",
                             "r_PIPIIEP",
                             "r_PIPIV",
                             "r_PIPIVED",
                             "r_PIPIVEP",
                             "r_PIPV",
                             "r_PIPVED",
                             "r_PIPVEP",
                             "r_Rad_Carp",
                             "r_RadiusE",
                             "r_Sca_Cap",
                             "r_ScaphE",
                             "r_Tra_Sca",
                             "r_TrapE",
                             "r_UlnaE"
                             ]
    r_neutral_hand_jsn = ["r_CMCIII",
                          "r_CMCIV",
                          "r_CMCV",
                          "r_MCPIII",
                          "r_MCPII",
                          "r_MCPI",
                          "r_MCPIV",
                          "r_MCPV",
                          "r_PIPIII",
                          "r_PIPII",
                          "r_PIPIV",
                          "r_PIPV",
                          "r_Rad_Carp",
                          "r_Sca_Cap",
                          "r_Tra_Sca",
                          ]
    r_neutral_hand_ero = ["r_Base_MCIE",
                          "r_IPIED",
                          "r_IPIEP",
                          "r_LunatE",
                          "r_MCPIIIED",
                          "r_MCPIIIEP",
                          "r_MCPIIED",
                          "r_MCPIIEP",
                          "r_MCPIED",
                          "r_MCPIEP",
                          "r_MCPIVED",
                          "r_MCPIVEP",
                          "r_MCPVED",
                          "r_MCPVEP",
                          "r_PIPIIIED",
                          "r_PIPIIIEP",
                          "r_PIPIIED",
                          "r_PIPIIEP",
                          "r_PIPIVED",
                          "r_PIPIVEP",
                          "r_PIPVED",
                          "r_PIPVEP",
                          "r_RadiusE",
                          "r_ScaphE",
                          "r_TrapE",
                          "r_UlnaE"
                          ]
    r_neutral_foot_joints = ["r_IP",
                             "r_IPED",
                             "r_IPEP",
                             "r_MTPIII",
                             "r_MTPIIIED",
                             "r_MTPIIIEP",
                             "r_MTPII",
                             "r_MTPIIED",
                             "r_MTPIIEP",
                             "r_MTPI",
                             "r_MTPIED",
                             "r_MTPIEP",
                             "r_MTPIV",
                             "r_MTPIVED",
                             "r_MTPIVEP",
                             "r_MTPV",
                             "r_MTPVED",
                             "r_MTPVEP",
                             ]
    r_neutral_foot_jsn = ["r_IP",
                          "r_MTPIII",
                          "r_MTPII",
                          "r_MTPI",
                          "r_MTPIV",
                          "r_MTPV",
                          ]
    r_neutral_foot_ero = ["r_IPED",
                          "r_IPEP",
                          "r_MTPIIIED",
                          "r_MTPIIIEP",
                          "r_MTPIIED",
                          "r_MTPIIEP",
                          "r_MTPIED",
                          "r_MTPIEP",
                          "r_MTPIVED",
                          "r_MTPIVEP",
                          "r_MTPVED",
                          "r_MTPVEP",
                          ]

    # remove 'r_'
    neutral_hand_joints = [re.sub("^r_", "", joint_name) for joint_name in r_neutral_hand_joints]
    neutral_hand_jsn = [re.sub("^r_", "", joint_name) for joint_name in r_neutral_hand_jsn]
    neutral_hand_ero = [re.sub("^r_", "", joint_name) for joint_name in r_neutral_hand_ero]
    neutral_foot_joints = [re.sub("^r_", "", joint_name) for joint_name in r_neutral_foot_joints]
    neutral_foot_jsn = [re.sub("^r_", "", joint_name) for joint_name in r_neutral_foot_jsn]
    neutral_foot_ero = [re.sub("^r_", "", joint_name) for joint_name in r_neutral_foot_ero]

    if extremity == "H":
        score_path = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/autoscoRA_TDEIMEL_HOME/autoscoRA_Preprocessing/output/data_split/pat_df_medstream_manual_H_dp_img_of_int_2_summary_cols_segm_sets_RL_stratmean_split6543_chosen345_2019-05-03_21-12-37.csv"
        # score_path = \
        #     '/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/output/data_split/' \
        #     'pat_df_medstream_manual_H_dp_img_of_int_2_' \
        #     'summary_cols_segm_sets_RL_stratmean_split6543_chosen345_2019-05-03_21-12-37.csv'
        # const.PAT_DF_MANUAL_PATH
        r_neutral_jsn = r_neutral_hand_jsn
        r_neutral_ero = r_neutral_hand_ero
    elif extremity == "F":
        score_path = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/autoscoRA_TDEIMEL_HOME/autoscoRA_Preprocessing/output/pat_df_manual_FEET/data_split/pat_df_medstream_manual_F_dp_img_of_int_2_summary_cols_segm_sets_RL_stratmean_split6543_chosen345_2021-05-25_13-45-01_fixedComTBD.csv"
        # score_path = \
        #     '/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/output/pat_df_manual_FEET/data_split/' \
        #     'pat_df_medstream_manual_F_dp_img_of_int_2_' \
        #     'summary_cols_segm_sets_RL_stratmean_split6543_chosen345_2021-05-25_13-45-01_fixedComTBD.csv'
        # const.F_PAT_DF_MANUAL_PATH
        r_neutral_jsn = r_neutral_foot_jsn
        r_neutral_ero = r_neutral_foot_ero
    else:
        raise ValueError("extremity must bei either 'H' or 'F'")

    # load df
    score_df = pd.read_csv(score_path)

    # transform to dict
    img_scoretype_score = {row['filename_manual']: {'JSN': {re.sub("^r_", "", JSN_joint): row[JSN_joint]
                                                            for JSN_joint in r_neutral_jsn},
                                                    'ERO': {re.sub("^r_", "", ero_joint): row[ero_joint]
                                                            for ero_joint in r_neutral_ero}
                                                    }
                           for index, row in score_df.iterrows()}

    # scoretype -> score -> img
    scoretype_score_img = {"JSN": {}, "ERO": {}}
    for score_type, score_roi in scor_roi_matching_dict.items():
        for score in score_roi.keys():
            scoretype_score_img[score_type][score] = {}.fromkeys(img_scoretype_score)
            for img, score_type_dict in img_scoretype_score.items():
                scoretype_score_img[score_type][score][img] = img_scoretype_score[img][score_type][score]

    # get rois for given score

    # example
    #chosen_score = "PIPIIIED"
    #chosen_score_type = 'ERO'  # or detect automatically

    patchpath_score_dict = {img + "_" + scor_roi_matching_dict[chosen_score_type][chosen_score][0] + ".npy": score
                            for img, score in scoretype_score_img[chosen_score_type][chosen_score].items()}
    path_list = []
    score_list = []
    for path, score in patchpath_score_dict.items():
        path_list = path_list + [path]
        score_list = score_list + [score]

    return path_list, score_list


def mandatory_train_val_test_ids(segm_to_train=True, double_to_test=True, missing_to_train=False, conflict='train',
                                 get_paths_from_const=False,
                                 input_constants_yml_path = "/home/cwatzenboeck/code/RA/ra_utils/ra_utils/autoscora/autoscorRA_Pipeline/input/constants/input_constants_cw.yml"):
    """
    :param segm_to_train:
    :param double_to_test:
    :param missing_to_train:
    :param conflict:
    :param get_paths_from_const:
    :return: a dict {"train": [...], "val": [...], "test": [...]} containing the patient ids that need to be
             in training (first), validation (second), and test (third) dataset, respectively.
    """
    if get_paths_from_const:
        """
        import input.constants.input_constants as const
        H_score_path = const.PAT_DF_MANUAL_PATH
        F_score_path = const.F_PAT_DF_MANUAL_PATH
        H_segm_path = const.JOINTS_PATH_GT_100
        F_segm_path = const.F_JOINTS_PATH_GT_100
        double_score_path = const.H_F_DOUBLE_SCORE_PATH
        """
        pass
    else:
        
        with open(input_constants_yml_path, "r") as f:
            const = yaml.safe_load(f)
        H_score_path = const["PAT_DF_MANUAL_PATH"]
        F_score_path = const["F_PAT_DF_MANUAL_PATH"]
        H_segm_path = const["JOINTS_PATH_GT_100"]
        F_segm_path = const["F_JOINTS_PATH_GT_100"]
        double_score_path = const["H_F_DOUBLE_SCORE_PATH"]

        """
        H_score_path = \
            '/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/output/data_split/' \
            'pat_df_medstream_manual_H_dp_img_of_int_2_' \
            'summary_cols_segm_sets_RL_stratmean_split6543_chosen345_2019-05-03_21-12-37.csv'
        F_score_path = \
            '/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/output/pat_df_manual_FEET/data_split/' \
            'pat_df_medstream_manual_F_dp_img_of_int_2_' \
            'summary_cols_segm_sets_RL_stratmean_split6543_chosen345_2021-05-25_13-45-01_fixedComTBD.csv'

        H_segm_path = "/home/cir/tdeimel/autoscoRA/autoscoRA_Annotation/output/annotations/" \
                      "hands_combined/all_hand_joints_100_onlynecessaryjoints.csv"

        F_segm_path = "/home/cir/tdeimel/autoscoRA/autoscoRA_Annotation/output/annotations/" \
                      "feet/all_foot_joints_100.csv"

        double_score_path = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/output/double_scoring/" \
                            "Re_Scoring_50_v2_FINAL_pseudonymized.csv"
        """

    H = pd.read_csv(H_score_path)
    F = pd.read_csv(F_score_path)
    H_segm = pd.read_csv(H_segm_path)
    F_segm = pd.read_csv(F_segm_path)
    D = pd.read_csv(double_score_path)

    mand_train = []
    mand_val = []
    mand_test = []

    if missing_to_train:
        # missing extremities = MUST BE IN train set
        H_miss = (H
                  .groupby('id_RO_nr')[['laterality_manual']]
                  .agg(lambda x: x.sum() in ['RL', 'LR'])
                  .query("laterality_manual == False")
                  .index
                  .tolist()
                  )
        # a = H[H.id_nr==203].sort_values(by="RO_datum")
        F_miss = (F
                  .groupby('id_RO_nr')[['laterality_manual', 'bodypart_manual']]
                  .agg(lambda x: (x['laterality_manual'].sum() in ['RL', 'LR']) and (x['bodypart_manual'].sum() in ['HF', 'FH']))
                  .query("laterality_manual == False")
                  .index
                  .tolist()
                  )
        FH_miss = list(set([f for f in F.id_RO_nr if f not in list(H.id_RO_nr)] + [h for h in H.id_RO_nr if h not in list(F.id_RO_nr)]))
        # F.query("id_nr == 689").filter(['id_RO_nr', 'laterality_manual', 'bodypart_manual'])
        # H.query("id_nr == 689").filter(['id_RO_nr', 'laterality_manual', 'bodypart_manual'])

        mand_train = mand_train + list(set([re.sub("_.*", "", i) for i in H_miss] +
                                           [re.sub("_.*", "", i) for i in F_miss] +
                                           [re.sub("_.*", "", i) for i in FH_miss]))
    if segm_to_train:
        # segm annotation pat = MUST BE IN train set
        mand_train = mand_train + list(set([re.sub("_.*", "", i) for i in H_segm.img] + [re.sub("_.*", "", i) for i in F_segm.img]))

    if double_to_test:
        # double scoring available = MUST BE IN final test set
        mand_test = mand_test + list(set([str(int(i)) for i in D.id_nr if not np.isnan(i)]))

    if conflict == 'train':
        # if ids in "MUST BE TRAIN" and "MUST BE VAL", put them in train
        mand_test_return = [i for i in mand_test if i not in mand_train]
        excluded_mand_test = [i for i in mand_test if i in mand_train]
        # [i in [re.sub("_.*", "", i) for i in (H_segm.filter(['img']))['img']] for i in excluded_mand_test]
        # [i in [re.sub("_.*", "", i) for i in (F_segm.filter(['img']))['img']] for i in excluded_mand_test]

        mand_train_return = list(set(mand_train + excluded_mand_test))
        # print("mand_train NOT in excluded_mand_test")
        # print([i for i in mand_train if i not in excluded_mand_test])
        # print(list(set(mand_train)) == mand_train_return)

    elif conflict == 'test':
        raise NotImplementedError()
    else:
        raise ValueError()

    mand_val_return = mand_val

    # von 100 segm img in F (bzw. bissl anderen id_RO_nr (aber gleichen id_nr) 100 in H) sind nur 80 unique id_nr
    return {"train": mand_train_return, "val": mand_val_return, "test": mand_test_return}

#

