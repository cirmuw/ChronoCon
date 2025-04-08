# import input.constants.input_constants as const
import re
import os
import pandas as pd


extremity = "H"  # "H", "F"

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
    score_path = \
        '/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/output/data_split/' \
        'pat_df_medstream_manual_H_dp_img_of_int_2_' \
        'summary_cols_segm_sets_RL_stratmean_split6543_chosen345_2019-05-03_21-12-37.csv'
    # const.PAT_DF_MANUAL_PATH
    r_neutral_jsn = r_neutral_hand_jsn
    r_neutral_ero = r_neutral_hand_ero
elif extremity == "F":
    score_path = \
        '/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/output/pat_df_manual_FEET/data_split/' \
        'pat_df_medstream_manual_F_dp_img_of_int_2_' \
        'summary_cols_segm_sets_RL_stratmean_split6543_chosen345_2021-05-25_13-45-01_fixedComTBD.csv'
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
"""
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
"""

#
