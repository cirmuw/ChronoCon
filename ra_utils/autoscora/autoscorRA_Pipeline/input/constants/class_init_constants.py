import os
import datetime
from collections import OrderedDict
import pandas as pd
import numpy as np
# from copy import copy, deepcopy
import re


print("running class_init_constants at " + str(datetime.datetime.now()))


###################
# Notes and TODOs #
###################

# maybe more efficient to have variables such as all_coords_odict as class variables - rather than instance vars
# # see, e.g., https://stackoverflow.com/questions/68645/are-static-class-variables-possible-in-python/36216964#36216964
# # and then take those class variables as the default value when creating instance
# # the "instance copy" probably isn't a real copy anyway but just points towards the object's memory location

#######################
# roi and point names #
#######################

ONLY_USE_ONE_ROI_FOR_SCORING = True

# points + rois
COORD_NAMES = [
    # "MCB5-R",
    # "MCB5-U", "TRI-U",
    # "RAD-UP", "RAD-UD", "RAD-RD",
    # "TRA-R", "MCB1-R", "MCB1-U", "MCB2-R", "MCB2-U",
    "CMC1-D", "MCB1-P",
    "MCP1-P", "MCP1-D", "IPJ1-P", "IPJ1-D", "END1",  # 25
    "CMC2-D", "MCB2-P",
    "MCP2-P", "MCP2-D", "PIP2-P", "PIP2-D", "DIP2-P",  # 25
    "CMC3-D", "MCB3-P",
    "MCP3-P", "MCP3-D", "PIP3-P", "PIP3-D", "DIP3-P",  # 25
    "CMC4-D", "MCB4-P",
    "MCP4-P", "MCP4-D", "PIP4-P", "PIP4-D", "DIP4-P",  # 25
    "CMC5-D", "MCB5-P",
    "MCP5-P", "MCP5-D", "PIP5-P", "PIP5-D", "DIP5-P",  # 25
    # "MPR1", "MPU1", "MDR1", "MDU1",
    # "MPR2", "MPU2", "MDR2", "MDU2",
    # "MPR3", "MPU3", "MDR3", "MDU3",
    # "MPR4", "MPU4", "MDR4", "MDU4",
    # "MPR5", "MPU5", "MDR5", "MDU5",
    # "PPR1", "PPU1", "PDR1", "PDU1",
    # "PPR2", "PPU2", "PDR2", "PDU2",
    # "PPR3", "PPU3", "PDR3", "PDU3",
    # "PPR4", "PPU4", "PDR4", "PDU4",
    # "PPR5", "PPU5", "PDR5", "PDU5",
    # patches: e.g., proximal square patch MCP 4 = SMP4
    # "SMP4-DR", "SMP4-PR", "SMP4-PU", "SMP4-DU",
    # "SMD4-DR", "SMD4-PR", "SMD4-PU", "SMD4-DU",
    # "SPP4-DR", "SPP4-PR", "SPP4-PU", "SPP4-DU",
    # "SPD4-DR", "SPD4-PR", "SPD4-PU", "SPD4-DU"
]

# roi corner coords
ROI_COORD_NAMES_ODICT = OrderedDict([("SMP2", ["SMP2-DR", "SMP2-PR", "SMP2-PU", "SMP2-DU"]),
                                     ("SMD2", ["SMD2-DR", "SMD2-PR", "SMD2-PU", "SMD2-DU"]),
                                     ("SPP2", ["SPP2-DR", "SPP2-PR", "SPP2-PU", "SPP2-DU"]),
                                     ("SPD2", ["SPD2-DR", "SPD2-PR", "SPD2-PU", "SPD2-DU"]),
                                     ("SMP3", ["SMP3-DR", "SMP3-PR", "SMP3-PU", "SMP3-DU"]),
                                     ("SMD3", ["SMD3-DR", "SMD3-PR", "SMD3-PU", "SMD3-DU"]),
                                     ("SPP3", ["SPP3-DR", "SPP3-PR", "SPP3-PU", "SPP3-DU"]),
                                     ("SPD3", ["SPD3-DR", "SPD3-PR", "SPD3-PU", "SPD3-DU"]),
                                     ("SMP4", ["SMP4-DR", "SMP4-PR", "SMP4-PU", "SMP4-DU"]),
                                     ("SMD4", ["SMD4-DR", "SMD4-PR", "SMD4-PU", "SMD4-DU"]),
                                     ("SPP4", ["SPP4-DR", "SPP4-PR", "SPP4-PU", "SPP4-DU"]),
                                     ("SPD4", ["SPD4-DR", "SPD4-PR", "SPD4-PU", "SPD4-DU"]),
                                     ("SMP5", ["SMP5-DR", "SMP5-PR", "SMP5-PU", "SMP5-DU"]),
                                     ("SMD5", ["SMD5-DR", "SMD5-PR", "SMD5-PU", "SMD5-DU"]),
                                     ("SPP5", ["SPP5-DR", "SPP5-PR", "SPP5-PU", "SPP5-DU"]),
                                     ("SPD5", ["SPD5-DR", "SPD5-PR", "SPD5-PU", "SPD5-DU"])
                                     ])

# only points
POINT_COORD_NAMES = [name for name in COORD_NAMES
                     if name not in [values for top_keys, sublist in ROI_COORD_NAMES_ODICT.items()
                                     for values in sublist]
                     ]

# roi names and points used in their calculation from centers
ROI_POINT_COORD_NAMES_ODICT = OrderedDict([
    ("SMP2", ["MCP1-P", "MCP1-D",
              "MCB2-P", "MCP2-P", "MCP2-D", "PIP2-P",
              "MCP3-P", "MCP3-D",
              ]),
    ("SMD2", ["MCP1-P", "MCP1-D",
              "MCB2-P", "MCP2-P", "MCP2-D", "PIP2-P",
              "MCP3-P", "MCP3-D",
              ]),
    ("SPP2", ["IPJ1-P", "IPJ1-D",
              "MCP2-D", "PIP2-P", "PIP2-D", "DIP2-P",
              "PIP3-P", "PIP3-D",
              ]),
    ("SPD2", ["IPJ1-P", "IPJ1-D",
              "MCP2-D", "PIP2-P", "PIP2-D", "DIP2-P",
              "PIP3-P", "PIP3-D",
              ]),
    ("SMP3", ["MCP2-P", "MCP2-D",
              "MCB3-P", "MCP3-P", "MCP3-D", "PIP3-P",
              "MCP4-P", "MCP4-D",
              ]),
    ("SMD3", ["MCP2-P", "MCP2-D",
              "MCB3-P", "MCP3-P", "MCP3-D", "PIP3-P",
              "MCP4-P", "MCP4-D",
              ]),
    ("SPP3", ["PIP2-P", "PIP2-D",
              "MCP3-D", "PIP3-P", "PIP3-D", "DIP3-P",
              "PIP4-P", "PIP4-D",
              ]),
    ("SPD3", ["PIP2-P", "PIP2-D",
              "MCP3-D", "PIP3-P", "PIP3-D", "DIP3-P",
              "PIP4-P", "PIP4-D",
              ]),
    ("SMP4", ["MCP3-P", "MCP3-D",
              "MCB4-P", "MCP4-P", "MCP4-D", "PIP4-P",
              "MCP5-P", "MCP5-D",
              ]),
    ("SMD4", ["MCP3-P", "MCP3-D",
              "MCB4-P", "MCP4-P", "MCP4-D", "PIP4-P",
              "MCP5-P", "MCP5-D",
              ]),
    ("SPP4", ["PIP3-P", "PIP3-D",
              "MCP4-D", "PIP4-P", "PIP4-D", "DIP4-P",
              "PIP5-P", "PIP5-D",
              ]),
    ("SPD4", ["PIP3-P", "PIP3-D",
              "MCP4-D", "PIP4-P", "PIP4-D", "DIP4-P",
              "PIP5-P", "PIP5-D",
              ]),
    ("SMP5", ["MCP4-P", "MCP4-D",
              "MCB5-P", "MCP5-P", "MCP5-D", "PIP5-P",
              "MCP5-P", "MCP5-D",
              ]),
    ("SMD5", ["MCP4-P", "MCP4-D",
              "MCB5-P", "MCP5-P", "MCP5-D", "PIP5-P",
              "MCP5-P", "MCP5-D",
              ]),
    ("SPP5", ["PIP4-P", "PIP4-D",
              "MCP5-D", "PIP5-P", "PIP5-D", "DIP5-P",
              "PIP5-P", "PIP5-D",
              ]),
    ("SPD5", ["PIP4-P", "PIP4-D",
              "MCP5-D", "PIP5-P", "PIP5-D", "DIP5-P",
              "PIP5-P", "PIP5-D",
              ]),
])

# roi names and corresponding joint centers
ROI_TO_BONE_CENTER_POINT_ODICT = OrderedDict([('SMP2', 'MCP2-P'),
                                              ('SMD2', 'MCP2-D'),
                                              ('SPP2', 'PIP2-P'),
                                              ('SPD2', 'PIP2-D'),
                                              ('SMP3', 'MCP3-P'),
                                              ('SMD3', 'MCP3-D'),
                                              ('SPP3', 'PIP3-P'),
                                              ('SPD3', 'PIP3-D'),
                                              ('SMP4', 'MCP4-P'),
                                              ('SMD4', 'MCP4-D'),
                                              ('SPP4', 'PIP4-P'),
                                              ('SPD4', 'PIP4-D'),
                                              ('SMP5', 'MCP5-P'),
                                              ('SMD5', 'MCP5-D'),
                                              ('SPP5', 'PIP5-P'),
                                              ('SPD5', 'PIP5-D'),
                                              ])

# filter out all ROIs that do not occur in ROI_NAMES
# (only use if you want to train network faster (i.e., avoid, at each generator batch run,
# that I always extract and augment also those rois/joints
# that are not of interest in a given training process))
if ONLY_USE_ONE_ROI_FOR_SCORING:
    from input.constants.score_train_constants import ROI
    from input.constants.score_train_constants import SCORE_TYPE
    ROI_NAMES = np.array([ROI]).flatten().tolist()
    SCORE_TYPES = np.array([SCORE_TYPE]).flatten().tolist()
else:
    ROI_NAMES = ["SMP2", "SMD2", "SPP2", "SPD2",
                 "SMP3", "SMD3", "SPP3", "SPD3",
                 "SMP4", "SMD4", "SPP4", "SPD4",
                 "SMP5", "SMD5", "SPP5", "SPD5"]
    SCORE_TYPES = ['JSN', 'ero']

ROI_COORD_NAMES_ODICT = OrderedDict([(key, value) for key, value in ROI_COORD_NAMES_ODICT.items()
                                     if key in ROI_NAMES])
ROI_POINT_COORD_NAMES_ODICT = OrderedDict([(key, value) for key, value in ROI_POINT_COORD_NAMES_ODICT.items()
                                           if key in ROI_NAMES])
ROI_TO_BONE_CENTER_POINT_ODICT = OrderedDict([(key, value) for key, value in ROI_TO_BONE_CENTER_POINT_ODICT.items()
                                              if key in ROI_NAMES])

###############
# score names #
###############

"""
neutral_joints = ["r_Base_MCIE",
                  "r_CMCIII",
                  "r_CMCIV",
                  "r_CMCV",
                  "r_IPIED",
                  "r_IPIEP",
                  "r_IP",
                  "r_IPED",
                  "r_IPEP",
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

neutral_JSN = ["r_CMCIII",
               "r_CMCIV",
               "r_CMCV",
               "r_IP",
               "r_MCPIII",
               "r_MCPII",
               "r_MCPI",
               "r_MCPIV",
               "r_MCPV",
               "r_MTPIII",
               "r_MTPII",
               "r_MTPI",
               "r_MTPIV",
               "r_MTPV",
               "r_PIPIII",
               "r_PIPII",
               "r_PIPIV",
               "r_PIPV",
               "r_Rad_Carp",
               "r_Sca_Cap",
               "r_Tra_Sca",
               ]

neutral_ero = ["r_Base_MCIE",
               "r_IPIED",
               "r_IPIEP",
               "r_IPED",
               "r_IPEP",
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

"""

# score joint names
R_NEUTRAL_HAND_JOINTS = ["r_Base_MCIE",
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
R_NEUTRAL_HAND_JSN = ["r_CMCIII",
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
R_NEUTRAL_HAND_ERO = ["r_Base_MCIE",
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
R_NEUTRAL_FOOT_JOINTS = ["r_IP",
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
R_NEUTRAL_FOOT_JSN = ["r_IP",
                      "r_MTPIII",
                      "r_MTPII",
                      "r_MTPI",
                      "r_MTPIV",
                      "r_MTPV",
                      ]
R_NEUTRAL_FOOT_ERO = ["r_IPED",
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
NEUTRAL_HAND_JOINTS = [re.sub("^r_", "", joint_name) for joint_name in R_NEUTRAL_HAND_JOINTS]
NEUTRAL_HAND_JSN = [re.sub("^r_", "", joint_name) for joint_name in R_NEUTRAL_HAND_JSN]
NEUTRAL_HAND_ERO = [re.sub("^r_", "", joint_name) for joint_name in R_NEUTRAL_HAND_ERO]
NEUTRAL_FOOT_JINTS = [re.sub("^r_", "", joint_name) for joint_name in R_NEUTRAL_FOOT_JOINTS]
NEUTRAL_FOOT_JSN = [re.sub("^r_", "", joint_name) for joint_name in R_NEUTRAL_FOOT_JSN]
NEUTRAL_FOOT_ERO = [re.sub("^r_", "", joint_name) for joint_name in R_NEUTRAL_FOOT_ERO]


#############################################
# relate rois to score joint names and v.v. #
#############################################

# score_type --> score_name --> roi
SCOR_ROI_MATCHING_ODICT = OrderedDict([('JSN', OrderedDict([('CMCIII', []),
                                                            ('CMCIV', []),
                                                            ('CMCV', []),
                                                            ('MCPIII', ['SMP3', 'SMD3']),
                                                            ('MCPII', ['SMP2', 'SMD2']),
                                                            ('MCPI', []),
                                                            ('MCPIV', ['SMP4', 'SMD4']),
                                                            ('MCPV', ['SMP5', 'SMD5']),
                                                            ('PIPIII', ['SPP3', 'SPD3']),
                                                            ('PIPII', ['SPP2', 'SPD2']),
                                                            ('PIPIV', ['SPP4', 'SPD4']),
                                                            ('PIPV', ['SPP5', 'SPD5']),
                                                            ('Rad_Carp', []),
                                                            ('Sca_Cap', []),
                                                            ('Tra_Sca', [])])),
                                       ('ero', OrderedDict([('Base_MCIE', []),
                                                            ('IPIED', []),
                                                            ('IPIEP', []),
                                                            ('LunatE', []),
                                                            ('MCPIIIED', ['SMD3']),
                                                            ('MCPIIIEP', ['SMP3']),
                                                            ('MCPIIED', ['SMD2']),
                                                            ('MCPIIEP', ['SMP2']),
                                                            ('MCPIED', []),
                                                            ('MCPIEP', []),
                                                            ('MCPIVED', ['SMD4']),
                                                            ('MCPIVEP', ['SMP4']),
                                                            ('MCPVED', ['SMD5']),
                                                            ('MCPVEP', ['SMP5']),
                                                            ('PIPIIIED', ['SPD3']),
                                                            ('PIPIIIEP', ['SPP3']),
                                                            ('PIPIIED', ['SPD2']),
                                                            ('PIPIIEP', ['SPP2']),
                                                            ('PIPIVED', ['SPD4']),
                                                            ('PIPIVEP', ['SPP4']),
                                                            ('PIPVED', ['SPD5']),
                                                            ('PIPVEP', ['SPP5']),
                                                            ('RadiusE', []),
                                                            ('ScaphE', []),
                                                            ('TrapE', []),
                                                            ('UlnaE', [])
                                                            ]))])

# filter out all ROIs that do not occur in ROI_NAMES
SCOR_ROI_MATCHING_ODICT = OrderedDict([(key,
                                        OrderedDict([(subkey,
                                                      [roi for roi in subvalue
                                                       if (roi in ROI_NAMES and key in SCORE_TYPES)]
                                                      )
                                                     for subkey, subvalue in value.items()])
                                        )
                                       for key, value in SCOR_ROI_MATCHING_ODICT.items()])

# roi --> score_type --> score_name
ROI_SCOR_MATCHING_ODICT = OrderedDict([(roi, OrderedDict([('JSN', []), ('ero', [])])) for roi in
                                       list(ROI_COORD_NAMES_ODICT.keys())])

for roi_i, roi_odict_i in ROI_SCOR_MATCHING_ODICT.items():
    for score_type in roi_odict_i.keys():
        ROI_SCOR_MATCHING_ODICT[roi_i][score_type] = \
            [joint for joint, joint_rois in SCOR_ROI_MATCHING_ODICT[score_type].items()
             if roi_i in joint_rois]

################
# source paths #
################

RUN_LOCALLY = True

if RUN_LOCALLY:
    HOME_DIR_SEP = "/mnthome2" + os.sep
    PROJECT_DIR_SEP = "/project" + os.sep
    MATPLOTLIB_BACKEND = 'TkAgg'
else:
    HOME_DIR_SEP = "/home/cir/tdeimel" + os.sep
    PROJECT_DIR_SEP = "/project" + os.sep
    MATPLOTLIB_BACKEND = 'agg'

# images
EXTRACTED_ROIS_DIR = PROJECT_DIR_SEP + "autoscora/autoscoRA_rois/allsize05_trainvaltest_noOPnoComm_hdf5"
SCORING_IMAGES_DIR = PROJECT_DIR_SEP + \
                     "autoscora/autoscoRA_images/H_images_of_interest_2_renamed_mirrored_inverted_dicoms"
SEGM_IMAGES_DIR = PROJECT_DIR_SEP + "autoscora/autoscoRA_images/segmentation_dicoms/segm_RL_train_dicoms"

# output dirs
SCORING_OUTPUT_DIR = HOME_DIR_SEP + "autoscoRA/autoscoRA_Pipeline/output/score_and_evaluate_output"
SEGMENTATION_OUTPUT_DIR = HOME_DIR_SEP + "autoscoRA/autoscoRA_Pipeline/output/loc_and_segm_output"

# pat_df_manual
PREPROCESSING_OUTPUT_DIR = HOME_DIR_SEP + "autoscoRA/autoscoRA_Preprocessing/output"
PAT_DF_MANUAL_VERSION = \
    "data_split" + os.sep + \
    "pat_df_medstream_manual_H_dp_img_of_int_2_" \
    "summary_cols_segm_sets_RL_stratmean_split6543_chosen345_2019-05-03_21-12-37.csv"
PAT_DF_MANUAL_PATH = PREPROCESSING_OUTPUT_DIR + os.sep + PAT_DF_MANUAL_VERSION

# gt_points
COORDS_PATH = "input/gt_points/saved_points/applic_points_for_scoring/" \
              "35PointsNoRois_72trainapplytoall_CMCMCMCPPIPEND.csv"
# candidates_points/35Points4Rois_72_CMCMCMCPPIPEND_MCP4PIP4.csv
# applic_points_for_scoring/35PointsNoRois_72train_CMCMCMCPPIPEND.csv

# pred_points
PREDICTED_COORDS_PATH = "input/pred_points" + os.sep + "pred_" + os.path.basename(COORDS_PATH)

# pred_scores
USE_DEFAULT_PREDICTED_SCORES_ODICT = True
GET_SCORES_FROM_NETWORK = False
SCORING_NETWORK_PATH = SCORING_OUTPUT_DIR + os.sep + \
                       "2019-11-04_17-43-36_100epochs_PIP4_patches_JSN_PIPIV_SPP4_dense_net/weights/" \
                       "2019-11-04_17-43-36_weights_dense_net-92-0.29.hdf5"
JSN_PREDICTED_SCORES_PATH = \
    SCORING_OUTPUT_DIR + os.sep + \
    "2020-04-02_10-29-42_dropout_dense_patches_JSN_MCPII_MCPIII_MCPIV_MCPV_SMD2_SMD3_SMD4_SMD5_vgg16_manual/" \
    "epoch_predictions/2020-04-02_10-29-42_predictions_vgg16_manual_epoch_29.csv"

    # PIPII
    # "2020-01-31_16-33-21_dropout_dense_patches_JSN_PIPII_SPP2_vgg16_manual/" \
    # "epoch_predictions/2020-01-31_16-33-21_predictions_vgg16_manual_epoch_12.csv"

    # PIPIV
    # "2020-01-31_16-17-15_dropout_dense_patches_JSN_PIPIV_SPP4_vgg16_manual/" \
    # "epoch_predictions/2020-01-31_16-17-15_predictions_vgg16_manual_epoch_8.csv"

    # PIPV
    # "2020-01-31_06-34-21_dropout_dense_patches_JSN_PIPV_SPP5_vgg16_manual/" \
    # "epoch_predictions/2020-01-31_06-34-21_predictions_vgg16_manual_epoch_7.csv"

    # PIPIII
    # "2020-01-31_00-57-13_dropout_dense_patches_JSN_PIPIII_SPP3_vgg16_manual/" \
    # "epoch_predictions/2020-01-31_00-57-13_predictions_vgg16_manual_epoch_42.csv"

    # MCPV
    # "2020-01-30_06-29-35_dropout_dense_patches_JSN_MCPV_SMP5_vgg16_manual/" \
    # "epoch_predictions/2020-01-30_06-29-35_predictions_vgg16_manual_epoch_22.csv"

    # MCPIV
    # "2020-01-29_16-52-49_dropout_dense_patches_JSN_MCPIV_SMP4_vgg16_manual/" \
    # "epoch_predictions/2020-01-29_16-52-49_predictions_vgg16_manual_epoch_6.csv"

    # MCPIII
    # "2020-01-30_06-44-53_dropout_dense_patches_JSN_MCPIII_SMP3_vgg16_manual/" \
    # "epoch_predictions/2020-01-30_06-44-53_predictions_vgg16_manual_epoch_64.csv"

    # MCPII
    # "2020-01-30_16-19-25_dropout_dense_patches_JSN_MCPII_SMP2_vgg16_manual/" \
    # "epoch_predictions/2020-01-30_16-19-25_predictions_vgg16_manual_epoch_28.csv"
"""
    SCORING_OUTPUT_DIR + os.sep + \
                            "2019-12-29_17-54-05_no_decrease_lr_ero_patches_JSN_PIPIV_SPP4_vgg16_manual/" \
                            "new_predictions/predictions_2019-12-29_17-54-05_weights_vgg16_manual-29-0.79.csv"
                            # "2019-11-18_23-27-35_no_decrease_lr_ero_patches_JSN_PIPIV_SPP4_dense_net/" \
                            # "test_predictions/predictions_2019-11-18_23-27-35_weights_dense_net-113-0.29.csv"
                            # "2019-11-13_18-33-56_250epochs_PIP4_patches_JSN_PIPIV_SPP4_dense_net/" \
                            # "predictions/2019-11-13_18-33-56_predictions_dense_net_epoch_81.csv"
"""
ERO_PREDICTED_SCORES_PATH = SCORING_OUTPUT_DIR + os.sep + \
    "2020-04-27_14-25-25_dropout_dense_patches_ero_MCPIIEP_MCPIIIEP_MCPIVEP_MCPVEP_SMP2_SMP3_SMP4_SMP5_vgg16_manual/" \
    "epoch_predictions/2020-04-27_14-25-25_predictions_vgg16_manual_epoch_5.csv"  # None

"""
SCORING_OUTPUT_DIR + os.sep + \
                            "2020-01-01_21-32-36_no_decrease_lr_ero_patches_ero_MCPIVEP_SMP4_vgg16_manual/" \
                            "predictions/2020-01-01_21-32-36_predictions_vgg16_manual_epoch_31.csv"
"""
"""SCORING_OUTPUT_DIR + os.sep + \
                            "2019-11-13_18-33-56_250epochs_PIP4_patches_JSN_PIPIV_SPP4_dense_net/" \
                            "predictions/2019-11-13_18-33-56_predictions_dense_net_epoch_81.csv"
"""
#####################################
# pat_df_manual including medstream #
#####################################

# read in medstream
# pat_df_medstream = pd.read_excel(data_folder + os.sep + "AutoScoreRA_Data_Pseudonymized_v1_17102018.xlsx")

# read in pat_df_manual
# root = tk.Tk()
# root.withdraw()
# pat_df_manual_path = filedialog.askopenfilename()
PAT_DF_MANUAL = pd.read_csv(PAT_DF_MANUAL_PATH)

# create backup copy of pat_df_manual
PAT_DF_MANUAL_ORIGINAL = PAT_DF_MANUAL.copy()
# pat_df_manual = pat_df_manual_original.copy()


###################
# data split sets #
###################

# img file names categorized by segm set
SEGM_SET_KEYS = list(set(PAT_DF_MANUAL["segm_set"]))
SEGM_SET_DICT = {segm_key: [filepath for i, filepath in enumerate(PAT_DF_MANUAL["filename_manual"])
                            if PAT_DF_MANUAL["segm_set"].iloc[i] == segm_key]
                 for segm_key in SEGM_SET_KEYS}
# or: assign segm train, val, test manually
READ_IN_SEGM_TRAIN_VAL_TEST_FROM_TEXT_FILES = False
if READ_IN_SEGM_TRAIN_VAL_TEST_FROM_TEXT_FILES:
    SEGM_SET_DICT = dict.fromkeys(SEGM_SET_KEYS)
    with open('input/manual_data_split/segm_train.txt', 'r') as f:
        SEGM_SET_DICT['train'] = [i for i in f.read().splitlines() if len(i.strip()) > 0]
    with open('input/manual_data_split/segm_val.txt', 'r') as f:
        SEGM_SET_DICT['val'] = [i for i in f.read().splitlines() if len(i.strip()) > 0]
    if os.path.exists('input/manual_data_split/score_test.txt'):
        with open('input/manual_data_split/segm_test.txt', 'r') as f:
            SEGM_SET_DICT['test'] = [i for i in f.read().splitlines() if len(i.strip()) > 0]
    else:
        SEGM_SET_DICT['test'] = []
    if os.path.exists('input/manual_data_split/not_segm.txt'):
        with open('input/manual_data_split/not_segm.txt', 'r') as f:
            SEGM_SET_DICT['not_segm'] = [i for i in f.read().splitlines() if len(i.strip()) > 0]
    else:
        SEGM_SET_DICT['not_segm'] = []

# img file names categorized by set
SCOR_SET_KEYS = list(set(PAT_DF_MANUAL["set"]))
SCOR_SET_DICT = {scor_key: [filepath for i, filepath in enumerate(PAT_DF_MANUAL["filename_manual"])
                            if PAT_DF_MANUAL["set"].iloc[i] == scor_key]
                 for scor_key in SCOR_SET_KEYS}

# for initial experiments: use only segm test + val (as score val) and segm train (as score train)
USE_SEGM_TEST_V_SEGM_VALANDTRAIN_AS_SCOR_SET = False
if USE_SEGM_TEST_V_SEGM_VALANDTRAIN_AS_SCOR_SET:
    SCOR_SET_DICT = dict.fromkeys(SCOR_SET_KEYS)
    SCOR_SET_DICT['train'] = SEGM_SET_DICT['train']
    SCOR_SET_DICT['val'] = SEGM_SET_DICT['val'] + SEGM_SET_DICT['test']  # ONLY USE THIS AS VAL UNTIL YOU'VE USED IT FOR FINAL SEGM EVALUATION
    SCOR_SET_DICT['test'] = SEGM_SET_DICT['not_segm']  # DO NOT TOUCH THIS --> CONTAINS FINAL VAL AND TEST SETS!

# for initial experiments: use only segm test (as score val) and segm val + segm train (as score train)
USE_SEGM_TEST_V_SEGM_VALANDTRAIN_AS_SCOR_SET = False
if USE_SEGM_TEST_V_SEGM_VALANDTRAIN_AS_SCOR_SET:
    SCOR_SET_DICT = dict.fromkeys(SCOR_SET_KEYS)
    SCOR_SET_DICT['train'] = SEGM_SET_DICT['train'] + SEGM_SET_DICT['val']
    SCOR_SET_DICT['val'] = SEGM_SET_DICT['test']  # ONLY USE THIS AS VAL UNTIL YOU'VE USED IT FOR FINAL SEGM EVALUATION
    SCOR_SET_DICT['test'] = SEGM_SET_DICT['not_segm']  # DO NOT TOUCH THIS --> CONTAINS FINAL VAL AND TEST SETS!

# use segm val (as score val) vs. segm train (as score train) - but not just the chosen imgs (i.e., not just 1 per pat)
USE_SEGM_VAL_V_SEGM_TRAIN_AS_SCOR_SET = True
if USE_SEGM_VAL_V_SEGM_TRAIN_AS_SCOR_SET:
    SCOR_SET_DICT = dict.fromkeys(SCOR_SET_KEYS)
    SCOR_SET_DICT['train'] = SEGM_SET_DICT['train']
    SCOR_SET_DICT['val'] = SEGM_SET_DICT['val']  # ONLY USE THIS AS VAL UNTIL YOU'VE USED IT FOR FINAL SEGM EVALUATION
    SCOR_SET_DICT['test'] = SEGM_SET_DICT['test']  # DO NOT TOUCH SEGM_TEST SET YET!
    SCOR_SET_DICT['not_segm'] = SEGM_SET_DICT['not_segm']

# use segm val (as score val) vs. segm train (as score train), use segm_test + score_val as score test set
USE_SEGM_VAL_V_SEGM_TRAIN_AND_SEGMTEST_SCOREVAL_AS_SCOR_SET = False  # this was used for EULAR2020
if USE_SEGM_VAL_V_SEGM_TRAIN_AND_SEGMTEST_SCOREVAL_AS_SCOR_SET:
    SCOR_SET_DICT_TMP = dict.fromkeys(SCOR_SET_KEYS)
    SCOR_SET_DICT_TMP['train'] = SEGM_SET_DICT['train']
    SCOR_SET_DICT_TMP['val'] = SEGM_SET_DICT['val']  # ONLY USE THIS AS VAL UNTIL YOU'VE USED IT FOR FINAL SEGM EVALUATION
    SCOR_SET_DICT_TMP['test'] = SEGM_SET_DICT['test'] + SCOR_SET_DICT['val']  # DO NOT TOUCH SEGM_TEST SET YET!
    SCOR_SET_DICT_TMP['not_segm'] = SCOR_SET_DICT['test']
    SCOR_SET_DICT = SCOR_SET_DICT_TMP

# use segm val (as score val) vs. segm train (as score train), use segm_test + score_val as score test set
USE_SEGM_VAL_V_SEGM_TRAIN_AND_SEGMTEST_SCOREVAL_AS_SCOR_SET = False
if USE_SEGM_VAL_V_SEGM_TRAIN_AND_SEGMTEST_SCOREVAL_AS_SCOR_SET:
    SCOR_SET_DICT_TMP = dict.fromkeys(SCOR_SET_KEYS)
    SCOR_SET_DICT_TMP['train'] = SEGM_SET_DICT['train'] + SEGM_SET_DICT['val']
    SCOR_SET_DICT_TMP['val'] = SEGM_SET_DICT['test'] + SCOR_SET_DICT['val']  # ONLY USE THIS AS VAL UNTIL YOU'VE USED IT FOR FINAL SEGM EVALUATION
    SCOR_SET_DICT_TMP['test'] = SCOR_SET_DICT['test']  # DO NOT TOUCH SEGM_TEST SET YET!
    # SCOR_SET_DICT_TMP['not_segm'] = SCOR_SET_DICT['test']
    SCOR_SET_DICT = SCOR_SET_DICT_TMP

# use all as train for patch extraction and saving
USE_TRAIN_VAL_AS_SCOR_SET_FOR_ALL_IMAGES = False
if USE_TRAIN_VAL_AS_SCOR_SET_FOR_ALL_IMAGES:
    SCOR_SET_DICT_TMP = dict.fromkeys(SCOR_SET_KEYS)
    SCOR_SET_DICT_TMP['train'] = SCOR_SET_DICT['train'] + SCOR_SET_DICT['val']
    SCOR_SET_DICT_TMP['val'] = SCOR_SET_DICT['test']
    SCOR_SET_DICT_TMP['test'] = []
    SCOR_SET_DICT_TMP['not_segm'] = []
    SCOR_SET_DICT = SCOR_SET_DICT_TMP


# store train, val lists
# w = ("/Users/Tommi/Desktop/test_list_1")
# wr = open(w, 'w')
# for i in SCOR_SET_DICT['train'] + SCOR_SET_DICT['val']:
#     wr.write(i + '\n')

# or: assign score train, val, test manually
READ_IN_SCOR_TRAIN_VAL_TEST_FROM_TEXT_FILES = False
if READ_IN_SCOR_TRAIN_VAL_TEST_FROM_TEXT_FILES:
    SCOR_SET_DICT = dict.fromkeys(SCOR_SET_KEYS)
    with open('input/manual_data_split/score_train.txt', 'r') as f:
        SCOR_SET_DICT['train'] = [i for i in f.read().splitlines() if len(i.strip()) > 0]
    with open('input/manual_data_split/score_val.txt', 'r') as f:
        SCOR_SET_DICT['val'] = [i for i in f.read().splitlines() if len(i.strip()) > 0]
    if os.path.exists('input/manual_data_split/score_test.txt'):
        with open('input/manual_data_split/score_test.txt', 'r') as f:
            SCOR_SET_DICT['test'] = [i for i in f.read().splitlines() if len(i.strip()) > 0]
    else:
        SCOR_SET_DICT['test'] = []


####################
# gt + pred coords #
####################

delim_gt = ","
delim_pred = ";"  # ","
# read in ground truth coords
# coords_path = segmentation_output + os.sep + coords_version
COORDS_IMG_NAMES = np.genfromtxt(COORDS_PATH, delimiter=delim_gt, usecols=range(0, 1), dtype=np.str).tolist()
COORDS_IMG_NAMES = [name.strip('\"') for name in COORDS_IMG_NAMES]
COORDS_ARRAY = np.genfromtxt(COORDS_PATH, delimiter=delim_gt,
                             usecols=range(1, len(COORD_NAMES * 2) + 1), dtype=np.int32).tolist()
COORDS_ODICT_ARRAY = OrderedDict(zip(COORDS_IMG_NAMES, COORDS_ARRAY))
COORDS_ODICT_ODICT = OrderedDict.fromkeys(list(COORDS_IMG_NAMES))
for img_id in COORDS_IMG_NAMES:  # assign to each joint name a numpy array containing x & y-coord
    x_idx = np.arange(0, len(COORDS_ODICT_ARRAY[img_id]), 2).tolist()
    y_idx = np.arange(1, len(COORDS_ODICT_ARRAY[img_id]), 2).tolist()
    x_coords = [COORDS_ODICT_ARRAY[img_id][i] for i in x_idx]
    y_coords = [COORDS_ODICT_ARRAY[img_id][i] for i in y_idx]
    COORDS_ODICT_ODICT[img_id] = OrderedDict.fromkeys(COORD_NAMES)
    for j in range(0, len(COORD_NAMES)):
        COORDS_ODICT_ODICT[img_id][COORD_NAMES[j]] = np.array([x_coords[j], y_coords[j]])

# read in predicte d coords
# predicted_coords_path = segmentation_output + os.sep + predicted_coords_version
PREDICTED_COORDS_IMG_NAMES = np.genfromtxt(PREDICTED_COORDS_PATH, delimiter=delim_pred,
                                           usecols=range(0, 1), dtype=np.str).tolist()
PREDICTED_COORDS_IMG_NAMES = [name.strip('\"') for name in PREDICTED_COORDS_IMG_NAMES]
PREDICTED_COORDS_ARRAY = np.genfromtxt(PREDICTED_COORDS_PATH, delimiter=delim_pred,
                                       usecols=range(1, len(COORD_NAMES * 2) + 1), dtype=np.int32).tolist()
PREDICTED_COORDS_ODICT_ARRAY = OrderedDict(zip(PREDICTED_COORDS_IMG_NAMES, PREDICTED_COORDS_ARRAY))
PREDICTED_COORDS_ODICT_ODICT = OrderedDict.fromkeys(list(PREDICTED_COORDS_IMG_NAMES))
for img_id in PREDICTED_COORDS_IMG_NAMES:  # assign to each joint name a numpy array containing x & y-coord
    x_idx = np.arange(0, len(PREDICTED_COORDS_ODICT_ARRAY[img_id]), 2).tolist()
    y_idx = np.arange(1, len(PREDICTED_COORDS_ODICT_ARRAY[img_id]), 2).tolist()
    x_predicted_coords = [PREDICTED_COORDS_ODICT_ARRAY[img_id][i] for i in x_idx]
    y_predicted_coords = [PREDICTED_COORDS_ODICT_ARRAY[img_id][i] for i in y_idx]
    PREDICTED_COORDS_ODICT_ODICT[img_id] = OrderedDict.fromkeys(COORD_NAMES)
    for j in range(0, len(COORD_NAMES)):
        PREDICTED_COORDS_ODICT_ODICT[img_id][COORD_NAMES[j]] = np.array([x_predicted_coords[j], y_predicted_coords[j]])


####################
# gt + pred scores #
####################
# read in ground truth scores
SCORES_ODICT = OrderedDict([(row['filename_manual'],
                             {'ero': OrderedDict([(re.sub("^r_", "", ero_joint), row[ero_joint])
                                                  for ero_joint in R_NEUTRAL_HAND_ERO]),
                              'JSN': OrderedDict([(re.sub("^r_", "", JSN_joint), row[JSN_joint])
                                                  for JSN_joint in R_NEUTRAL_HAND_JSN])})
                            for index, row in PAT_DF_MANUAL.iterrows()])

# read in predicted scores
PRED_FLAG = (SCORING_NETWORK_PATH is None and
             (JSN_PREDICTED_SCORES_PATH is None and
              ERO_PREDICTED_SCORES_PATH is None)) or USE_DEFAULT_PREDICTED_SCORES_ODICT
if PRED_FLAG:
    print("CAVE: cic.PREDICTED_SCORES_DF is same as ground truth for now (since no score predictor yet)")
    PREDICTED_SCORES_DF = PAT_DF_MANUAL.copy()
    PREDICTED_SCORES_ODICT = OrderedDict([(row['filename_manual'],
                                           {'ero': OrderedDict([(re.sub("^r_", "", ero_joint), row[ero_joint])
                                                                for ero_joint in R_NEUTRAL_HAND_ERO]),
                                            'JSN': OrderedDict([(re.sub("^r_", "", JSN_joint), row[JSN_joint])
                                                                for JSN_joint in R_NEUTRAL_HAND_JSN])})
                                          for index, row in PREDICTED_SCORES_DF.iterrows()])

elif SCORING_NETWORK_PATH is not None and GET_SCORES_FROM_NETWORK:
    print("Getting cic.PREDICTED_SCORES_ODICT from cic.SCORING_NETWORK_PATH")

else:
    print("Getting cic.PREDICTED_SCORES_ODICT from cic.PREDICTED_SCORES_PATH")
    if JSN_PREDICTED_SCORES_PATH is not None and 'JSN' in SCORE_TYPES:
        JSN_PREDICTED_SCORES_DF = pd.read_csv(JSN_PREDICTED_SCORES_PATH, header=None)
        if len(JSN_PREDICTED_SCORES_DF.columns) > 2:
            JSN_PREDICTED_SCORES_DF.columns = ['filename_manual', 'roi', 'score']
            drop_dupl_ids = JSN_PREDICTED_SCORES_DF[JSN_PREDICTED_SCORES_DF.loc[:, ['filename_manual', 'roi']].duplicated()]
            drop_dupl_ids = drop_dupl_ids.filename_manual.unique().tolist()
            JSN_PREDICTED_SCORES_DF_FILTERED = JSN_PREDICTED_SCORES_DF[~JSN_PREDICTED_SCORES_DF.filename_manual.isin(drop_dupl_ids)]
            JSN_PREDICTED_SCORES_DF_FILTERED.set_index('filename_manual', inplace=True)
            JSN_PREDICTED_SCORES_ODICT = \
                OrderedDict([(filename, OrderedDict([(roi_i,
                                                      JSN_PREDICTED_SCORES_DF_FILTERED.loc[filename, 'score'][JSN_PREDICTED_SCORES_DF_FILTERED.loc[filename, 'roi'].eq(ROI_SCOR_MATCHING_ODICT[roi_i]['JSN'][0])][0])
                                                     for roi_i in ROI_NAMES]))
                             for filename in JSN_PREDICTED_SCORES_DF_FILTERED.index])
        else:
            JSN_PREDICTED_SCORES_DF.columns = ['filename_manual', ROI]
            JSN_PREDICTED_SCORES_DF.set_index('filename_manual', inplace=True)
            JSN_PREDICTED_SCORES_ODICT = OrderedDict([(filename, OrderedDict([(roi,
                                                                               JSN_PREDICTED_SCORES_DF.loc[
                                                                                   filename, roi])
                                                                              for roi in ROI_NAMES]))
                                                      for filename in JSN_PREDICTED_SCORES_DF.index])
        """
        JSN_PRED_JOINT, JSN_PRED_ROI = \
            re.split('_', re.sub('_JSN_', '', re.search("_JSN_[A-Z]+_[A-Z, 1-5]+_",
                                                        JSN_PREDICTED_SCORES_PATH)[0][0:-1]))
        """
    if ERO_PREDICTED_SCORES_PATH is not None and 'ero' in SCORE_TYPES:
        ERO_PREDICTED_SCORES_DF = pd.read_csv(ERO_PREDICTED_SCORES_PATH, header=None)
        if len(ERO_PREDICTED_SCORES_DF.columns) > 2:
            ERO_PREDICTED_SCORES_DF.columns = ['filename_manual', 'roi', 'score']
            drop_dupl_ids = ERO_PREDICTED_SCORES_DF[ERO_PREDICTED_SCORES_DF.loc[:, ['filename_manual', 'roi']].duplicated()]
            drop_dupl_ids = drop_dupl_ids.filename_manual.unique().tolist()
            ERO_PREDICTED_SCORES_DF_FILTERED = ERO_PREDICTED_SCORES_DF[
                ~ERO_PREDICTED_SCORES_DF.filename_manual.isin(drop_dupl_ids)]
            ERO_PREDICTED_SCORES_DF_FILTERED.set_index('filename_manual', inplace=True)
            ERO_PREDICTED_SCORES_ODICT = \
                OrderedDict([(filename, OrderedDict([(roi_i,
                                                      ERO_PREDICTED_SCORES_DF_FILTERED.loc[filename, 'score'][
                                                          ERO_PREDICTED_SCORES_DF_FILTERED.loc[filename, 'roi'].eq(
                                                              ROI_SCOR_MATCHING_ODICT[roi_i]['ero'][0])][0])
                                                     for roi_i in ROI_NAMES]))
                             for filename in ERO_PREDICTED_SCORES_DF_FILTERED.index])
        else:
            ERO_PREDICTED_SCORES_DF.columns = ['filename_manual', ROI]
            ERO_PREDICTED_SCORES_DF.set_index('filename_manual', inplace=True)
            ERO_PREDICTED_SCORES_ODICT = OrderedDict([(filename, OrderedDict([(roi,
                                                                               ERO_PREDICTED_SCORES_DF.loc[
                                                                                   filename, roi])
                                                                              for roi in ROI_NAMES]))
                                                      for filename in ERO_PREDICTED_SCORES_DF.index])

        """
        ERO_PRED_JOINT, ERO_PRED_ROI = \
            re.split('_', re.sub('_ero_', '', re.search("_ero_[A-Z]+_[A-Z, 1-5]+_",
                                                        ERO_PREDICTED_SCORES_PATH)[0][0:-1]))
        """

    PREDICTED_SCORES_ODICT = \
        OrderedDict([(row['filename_manual'],
                      {'ero': OrderedDict([(re.sub("^r_", "", ero_joint),
                                            OrderedDict([(roi,
                                                          ERO_PREDICTED_SCORES_ODICT[
                                                              row['filename_manual']][roi])
                                                         if ('ERO_PREDICTED_SCORES_ODICT' in globals()
                                                             and roi is not None
                                                             and (row['filename_manual']
                                                             in list(ERO_PREDICTED_SCORES_ODICT.keys())))
                                                         else (roi, None)
                                                         for roi in SCOR_ROI_MATCHING_ODICT['ero'][re.sub("^r_", "",
                                                                                                          ero_joint)]]))
                                           if len(SCOR_ROI_MATCHING_ODICT['ero'][re.sub("^r_", "", ero_joint)]) > 0
                                           else (re.sub("^r_", "", ero_joint), None)
                                           for ero_joint in R_NEUTRAL_HAND_ERO]),
                       'JSN': OrderedDict([(re.sub("^r_", "", JSN_joint),
                                            OrderedDict([(roi,
                                                          JSN_PREDICTED_SCORES_ODICT[
                                                              row['filename_manual']][roi])
                                                         if ('JSN_PREDICTED_SCORES_ODICT' in globals()
                                                             and roi is not None
                                                             and (row['filename_manual']
                                                             in list(JSN_PREDICTED_SCORES_ODICT.keys())))
                                                         else (roi, None)
                                                         for roi in SCOR_ROI_MATCHING_ODICT['JSN'][re.sub("^r_", "",
                                                                                                          JSN_joint)]]))
                                           if len(SCOR_ROI_MATCHING_ODICT['JSN'][re.sub("^r_", "", JSN_joint)]) > 0
                                           else (re.sub("^r_", "", JSN_joint), None)
                                           for JSN_joint in R_NEUTRAL_HAND_JSN])})
                     for index, row in PAT_DF_MANUAL.iterrows()])

    if 'ERO_PREDICTED_SCORES_ODICT' in globals():
        PREDICTED_SCORES_ODICT = OrderedDict([(key, value) for key, value in PREDICTED_SCORES_ODICT.items()
                                              if key in ERO_PREDICTED_SCORES_ODICT.keys()])
    if 'JSN_PREDICTED_SCORES_ODICT' in globals():
        PREDICTED_SCORES_ODICT = OrderedDict([(key, value) for key, value in PREDICTED_SCORES_ODICT.items()
                                              if key in JSN_PREDICTED_SCORES_ODICT.keys()])

#
