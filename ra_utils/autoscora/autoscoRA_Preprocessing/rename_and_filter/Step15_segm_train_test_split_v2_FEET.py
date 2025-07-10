# import SimpleITK as sitk
# import pydicom
import numpy as np
import os
# from os.path import dirname, abspath, exists, splitext  # , basename
# from os.path import join as join_path
# import itertools
# import sys
import pandas as pd
import dfply
from tableone import TableOne
# import re
import random
# import ntpath
import datetime
# from collections import Counter
# from distutils.dir_util import copy_tree
# import shutil
# from itertools import compress
# import tkinter as tk
# from tkinter import filedialog
from copy import deepcopy  # , copy
# from time import sleep
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


print(datetime.datetime.now())

#############
# load data #
#############
dicom_server = "/project/autoscora/autoscoRA_images/F_images_of_interest_2_renamed_mirrored_inverted_dicoms"
output_folder = "/mnthome2/autoscoRA/autoscoRA_Preprocessing/output/pat_df_manual_FEET"
# output_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/output/pat_df_manual_FEET"
pat_df_manual_version = "data_split/pat_df_medstream_manual_F_dp_img_of_int_2_summary_cols_pat_sets_stratmean_split699_2021-05-25_13-14-02.csv"
short_medstream_version = "data_split/pat_df_medstream_F_dp_img_of_int_2_summary_cols_pat_sets_stratmean_split699_2021-05-25_13-12-48.csv"
data_folder = "/mnthome2/autoscoRA/autoscoRA_Preprocessing/data"
# data_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/data"

split_seed = 6543  # 987  # 39840943  # 823/90  # 699/90
mean_strat = True
if mean_strat:
    strat_seed = 'mean'
    chosen_seed = 345  # 986534456  # 75645  # 69  # 20
else:
    strat_seed = 80  # 80  # 20
    chosen_seed = 'asstrat'
percentiles = [20, 40, 60, 80]  # [25, 50, 75]
# plot
cumulative_flag = True
histtype = 'step'
nbin = 100
normed_flag = True

# read in medstream
pat_df_medstream = pd.read_excel(data_folder + os.sep + "AutoScoreRA_Data_Pseudonymized_v1_17102018.xlsx")

# joints
nonremovable_joints = ['r_Base_MCILE',
                       'r_CMCIIIL',
                       'r_CMCIVL',
                       'r_CMCVL',
                       'r_IPILED',
                       'r_IPILEP',
                       'r_LunatLE',
                       'r_MCPIIIL',
                       'r_MCPIIILED',
                       'r_MCPIIILEP',
                       'r_MCPIIL',
                       'r_MCPIILED',
                       'r_MCPIILEP',
                       'r_MCPIL',
                       'r_MCPILED',
                       'r_MCPILEP',
                       'r_MCPIVL',
                       'r_MCPIVLED',
                       'r_MCPIVLEP',
                       'r_MCPVL',
                       'r_MCPVLED',
                       'r_MCPVLEP',
                       'r_PIPIIIL',
                       'r_PIPIIILED',
                       'r_PIPIIILEP',
                       'r_PIPIIL',
                       'r_PIPIILED',
                       'r_PIPIILEP',
                       'r_PIPIVL',
                       'r_PIPIVLED',
                       'r_PIPIVLEP',
                       'r_PIPVL',
                       'r_PIPVLED',
                       'r_PIPVLEP',
                       'r_Rad_CarpL',
                       'r_RadiusLE',
                       'r_Sca_CapL',
                       'r_ScaphLE',
                       'r_Tra_ScaL',
                       'r_TrapLE',
                       'r_UlnaLE',
                       'r_Base_MCIRE',
                       'r_CMCIIIR',
                       'r_CMCIVR',
                       'r_CMCVR',
                       'r_IPIRED',
                       'r_IPIREP',
                       'r_LunatRE',
                       'r_MCPIIIR',
                       'r_MCPIIIRED',
                       'r_MCPIIIREP',
                       'r_MCPIIR',
                       'r_MCPIIRED',
                       'r_MCPIIREP',
                       'r_MCPIR',
                       'r_MCPIRED',
                       'r_MCPIREP',
                       'r_MCPIVR',
                       'r_MCPIVRED',
                       'r_MCPIVREP',
                       'r_MCPVR',
                       'r_MCPVRED',
                       'r_MCPVREP',
                       'r_PIPIIIR',
                       'r_PIPIIIRED',
                       'r_PIPIIIREP',
                       'r_PIPIIR',
                       'r_PIPIIRED',
                       'r_PIPIIREP',
                       'r_PIPIVR',
                       'r_PIPIVRED',
                       'r_PIPIVREP',
                       'r_PIPVR',
                       'r_PIPVRED',
                       'r_PIPVREP',
                       'r_Rad_CarpR',
                       'r_RadiusRE',
                       'r_Sca_CapR',
                       'r_ScaphRE',
                       'r_Tra_ScaR',
                       'r_TrapRE',
                       'r_UlnaRE',
                       'r_IPL',
                       'r_IPLED',
                       'r_IPLEP',
                       'r_MTPIIIL',
                       'r_MTPIIILED',
                       'r_MTPIIILEP',
                       'r_MTPIIL',
                       'r_MTPIILED',
                       'r_MTPIILEP',
                       'r_MTPIL',
                       'r_MTPILED',
                       'r_MTPILEP',
                       'r_MTPIVL',
                       'r_MTPIVLED',
                       'r_MTPIVLEP',
                       'r_MTPVL',
                       'r_MTPVLED',
                       'r_MTPVLEP',
                       'r_IPR',
                       'r_IPRED',
                       'r_IPREP',
                       'r_MTPIIIR',
                       'r_MTPIIIRED',
                       'r_MTPIIIREP',
                       'r_MTPIIR',
                       'r_MTPIIRED',
                       'r_MTPIIREP',
                       'r_MTPIR',
                       'r_MTPIRED',
                       'r_MTPIREP',
                       'r_MTPIVR',
                       'r_MTPIVRED',
                       'r_MTPIVREP',
                       'r_MTPVR',
                       'r_MTPVRED',
                       'r_MTPVREP'
                       ]

JSN = ['r_CMCIIIL',
       'r_CMCIVL',
       'r_CMCVL',
       'r_MCPIIIL',
       'r_MCPIIL',
       'r_MCPIL',
       'r_MCPIVL',
       'r_MCPVL',
       'r_PIPIIIL',
       'r_PIPIIL',
       'r_PIPIVL',
       'r_PIPVL',
       'r_Rad_CarpL',
       'r_Sca_CapL',
       'r_Tra_ScaL',
       'r_CMCIIIR',
       'r_CMCIVR',
       'r_CMCVR',
       'r_MCPIIIR',
       'r_MCPIIR',
       'r_MCPIR',
       'r_MCPIVR',
       'r_MCPVR',
       'r_PIPIIIR',
       'r_PIPIIR',
       'r_PIPIVR',
       'r_PIPVR',
       'r_Rad_CarpR',
       'r_Sca_CapR',
       'r_Tra_ScaR',
       'r_IPL',
       'r_MTPIIIL',
       'r_MTPIIL',
       'r_MTPIL',
       'r_MTPIVL',
       'r_MTPVL',
       'r_IPR',
       'r_MTPIIIR',
       'r_MTPIIR',
       'r_MTPIR',
       'r_MTPIVR',
       'r_MTPVR'
       ]

ero = ['r_Base_MCILE',
       'r_IPILED',
       'r_IPILEP',
       'r_LunatLE',
       'r_MCPIIILED',
       'r_MCPIIILEP',
       'r_MCPIILED',
       'r_MCPIILEP',
       'r_MCPILED',
       'r_MCPILEP',
       'r_MCPIVLED',
       'r_MCPIVLEP',
       'r_MCPVLED',
       'r_MCPVLEP',
       'r_PIPIIILED',
       'r_PIPIIILEP',
       'r_PIPIILED',
       'r_PIPIILEP',
       'r_PIPIVLED',
       'r_PIPIVLEP',
       'r_PIPVLED',
       'r_PIPVLEP',
       'r_RadiusLE',
       'r_ScaphLE',
       'r_TrapLE',
       'r_UlnaLE',
       'r_Base_MCIRE',
       'r_IPIRED',
       'r_IPIREP',
       'r_LunatRE',
       'r_MCPIIIRED',
       'r_MCPIIIREP',
       'r_MCPIIRED',
       'r_MCPIIREP',
       'r_MCPIRED',
       'r_MCPIREP',
       'r_MCPIVRED',
       'r_MCPIVREP',
       'r_MCPVRED',
       'r_MCPVREP',
       'r_PIPIIIRED',
       'r_PIPIIIREP',
       'r_PIPIIRED',
       'r_PIPIIREP',
       'r_PIPIVRED',
       'r_PIPIVREP',
       'r_PIPVRED',
       'r_PIPVREP',
       'r_RadiusRE',
       'r_ScaphRE',
       'r_TrapRE',
       'r_UlnaRE',
       'r_IPLED',
       'r_IPLEP',
       'r_MTPIIILED',
       'r_MTPIIILEP',
       'r_MTPIILED',
       'r_MTPIILEP',
       'r_MTPILED',
       'r_MTPILEP',
       'r_MTPIVLED',
       'r_MTPIVLEP',
       'r_MTPVLED',
       'r_MTPVLEP',
       'r_IPRED',
       'r_IPREP',
       'r_MTPIIIRED',
       'r_MTPIIIREP',
       'r_MTPIIRED',
       'r_MTPIIREP',
       'r_MTPIRED',
       'r_MTPIREP',
       'r_MTPIVRED',
       'r_MTPIVREP',
       'r_MTPVRED',
       'r_MTPVREP'
       ]

removable_joints = ["r_erosion",
                    "r_erosion_foot",
                    "r_erosion_hand",
                    "r_jsn",
                    "r_jsn_foot",
                    "r_jsn_hand",
                    "r_svdhst",
                    "r_svdhst1",
                    "r_jsn1",
                    "r_jsn_foot1",
                    "r_jsn_hand1",
                    "r_erosion1",
                    "r_erosion_foot1",
                    "r_erosion_hand1",
                    ]

left_hand_joints = ["r_Base_MCILE",
                    "r_CMCIIIL",
                    "r_CMCIVL",
                    "r_CMCVL",
                    "r_IPILED",
                    "r_IPILEP",
                    "r_LunatLE",
                    "r_MCPIIIL",
                    "r_MCPIIILED",
                    "r_MCPIIILEP",
                    "r_MCPIIL",
                    "r_MCPIILED",
                    "r_MCPIILEP",
                    "r_MCPIL",
                    "r_MCPILED",
                    "r_MCPILEP",
                    "r_MCPIVL",
                    "r_MCPIVLED",
                    "r_MCPIVLEP",
                    "r_MCPVL",
                    "r_MCPVLED",
                    "r_MCPVLEP",
                    "r_PIPIIIL",
                    "r_PIPIIILED",
                    "r_PIPIIILEP",
                    "r_PIPIIL",
                    "r_PIPIILED",
                    "r_PIPIILEP",
                    "r_PIPIVL",
                    "r_PIPIVLED",
                    "r_PIPIVLEP",
                    "r_PIPVL",
                    "r_PIPVLED",
                    "r_PIPVLEP",
                    "r_Rad_CarpL",
                    "r_RadiusLE",
                    "r_Sca_CapL",
                    "r_ScaphLE",
                    "r_Tra_ScaL",
                    "r_TrapLE",
                    "r_UlnaLE"
                    ]

right_hand_joints = ["r_Base_MCIRE",
                     "r_CMCIIIR",
                     "r_CMCIVR",
                     "r_CMCVR",
                     "r_IPIRED",
                     "r_IPIREP",
                     "r_LunatRE",
                     "r_MCPIIIR",
                     "r_MCPIIIRED",
                     "r_MCPIIIREP",
                     "r_MCPIIR",
                     "r_MCPIIRED",
                     "r_MCPIIREP",
                     "r_MCPIR",
                     "r_MCPIRED",
                     "r_MCPIREP",
                     "r_MCPIVR",
                     "r_MCPIVRED",
                     "r_MCPIVREP",
                     "r_MCPVR",
                     "r_MCPVRED",
                     "r_MCPVREP",
                     "r_PIPIIIR",
                     "r_PIPIIIRED",
                     "r_PIPIIIREP",
                     "r_PIPIIR",
                     "r_PIPIIRED",
                     "r_PIPIIREP",
                     "r_PIPIVR",
                     "r_PIPIVRED",
                     "r_PIPIVREP",
                     "r_PIPVR",
                     "r_PIPVRED",
                     "r_PIPVREP",
                     "r_Rad_CarpR",
                     "r_RadiusRE",
                     "r_Sca_CapR",
                     "r_ScaphRE",
                     "r_Tra_ScaR",
                     "r_TrapRE",
                     "r_UlnaRE"
                     ]

hand_joints = left_hand_joints + right_hand_joints

hand_JSN = [i for i in hand_joints if i in JSN]

hand_ero = [i for i in hand_joints if i in ero]

left_foot_joints = ["r_IPL",
                    "r_IPLED",
                    "r_IPLEP",
                    "r_MTPIIIL",
                    "r_MTPIIILED",
                    "r_MTPIIILEP",
                    "r_MTPIIL",
                    "r_MTPIILED",
                    "r_MTPIILEP",
                    "r_MTPIL",
                    "r_MTPILED",
                    "r_MTPILEP",
                    "r_MTPIVL",
                    "r_MTPIVLED",
                    "r_MTPIVLEP",
                    "r_MTPVL",
                    "r_MTPVLED",
                    "r_MTPVLEP"
                    ]

right_foot_joints = ["r_IPR",
                     "r_IPRED",
                     "r_IPREP",
                     "r_MTPIIIR",
                     "r_MTPIIIRED",
                     "r_MTPIIIREP",
                     "r_MTPIIR",
                     "r_MTPIIRED",
                     "r_MTPIIREP",
                     "r_MTPIR",
                     "r_MTPIRED",
                     "r_MTPIREP",
                     "r_MTPIVR",
                     "r_MTPIVRED",
                     "r_MTPIVREP",
                     "r_MTPVR",
                     "r_MTPVRED",
                     "r_MTPVREP"
                     ]

foot_joints = left_foot_joints + right_foot_joints

foot_JSN = [i for i in foot_joints if i in JSN]

foot_ero = [i for i in foot_joints if i in ero]

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

neutral_hand_joints = ["r_Base_MCIE",
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

neutral_foot_joints = ["r_IP",
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

neutral_hand_JSN = ["r_CMCIII",
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

neutral_hand_ero = ["r_Base_MCIE",
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

neutral_foot_JSN = ["r_IP",
                    "r_MTPIII",
                    "r_MTPII",
                    "r_MTPI",
                    "r_MTPIV",
                    "r_MTPV",
                    ]

neutral_foot_ero = ["r_IPED",
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

# read in pat_df_medstream with only visits of interest
pat_df_medstream_short_path = output_folder + os.sep + short_medstream_version
pat_df_medstream_short = pd.read_csv(pat_df_medstream_short_path)

# read in pat_df_manual
pat_df_manual_path = output_folder + os.sep + pat_df_manual_version
# root = tk.Tk()
# root.withdraw()
# pat_df_manual_path = filedialog.askopenfilename()
pat_df_manual = pd.read_csv(pat_df_manual_path)

# create backup copy of pat_df_manual
pat_df_manual_original = pat_df_manual.copy()
# pat_df_manual = pat_df_manual_original.copy()

# manual columns
manual_columns_selected = ['filename_manual',
                           'bodypart_manual', 'laterality_manual', 'view_position_manual',
                           'photometric_interpretation_new', 'inverted_manual',
                           'rotated_manual', 'black_manual', 'operated_manual',
                           'inappropriate_manual', 'img_of_interest_FEET',
                           'comment_status', 'comment',
                           'filename_new_dupl']

########################################################################
# From training set, randomly choose train, val, test for segmentation #
########################################################################

short_medstream = deepcopy(pat_df_medstream_short)
short_medstream_R_L_manual = deepcopy(pat_df_manual)

train_ids = list(set([x for i, x in enumerate(short_medstream_R_L_manual['id_nr'])
                      if short_medstream_R_L_manual['set'].iloc[i] == 'train']))

short_medstream_train_idx = list([i for i, x in enumerate(short_medstream['id_nr'])
                                  if short_medstream['set'].iloc[i] == 'train'])

short_medstream = short_medstream.iloc[short_medstream_train_idx, :]

if mean_strat:
    # stratification according to median of all visits
    strat_cols_mean = ['age', 'disease_duration', 'r_total_hand_JSN', 'r_total_foot_JSN',
                       'r_total_hand_ero', 'r_total_foot_ero', 'r_total_hand', 'r_total_foot',
                       'r_total_ero', 'r_total_JSN', 'r_total_score']
    strat_cols_majority = ['sex', 'rf_pos', 'ccp_pos']
    id_cols = ['id_nr']
    summary_operations_dict = dict.fromkeys(strat_cols_mean + strat_cols_majority)
    summary_operations_dict = {i: 'mean' for i in summary_operations_dict.keys()}
    short_medstream_pat = ((short_medstream >>
                            dfply.mutate(sex=1.0 * (dfply.X.sex == 'M')) >>
                            # dfply.group_by(dfply.X.id_nr) >>
                            dfply.select(id_cols + strat_cols_mean + strat_cols_majority)
                            ).groupby(['id_nr']).agg(summary_operations_dict)
                           )
    short_medstream_pat.loc[:, strat_cols_majority] = round(short_medstream_pat.loc[:, strat_cols_majority])
    sex_ambig = any([((0 < i < 1) and pd.notnull(i)) for i in list(short_medstream_pat.sex)])
    rf_pos_ambig = any([((0 < i < 1) and pd.notnull(i)) for i in list(short_medstream_pat.rf_pos)])
    ccp_pos_ambig = any([((0 < i < 1) and pd.notnull(i)) for i in list(short_medstream_pat.ccp_pos)])
    print('sex_ambig, rf_pos_ambig, ccp_pos_ambig:', sex_ambig, rf_pos_ambig, ccp_pos_ambig)

    strat_dict = short_medstream_pat.to_dict(orient='index')

else:
    # stratification according to a random visit
    strat_dict = dict.fromkeys(train_ids)
    np.random.seed(strat_seed)
    seeds = np.random.randint(low=0, high=10000, size=len(strat_dict.keys()))
    seeds_dict = dict(zip(strat_dict.keys(), seeds))
    for keys in strat_dict.keys():
        pat_visits = [list(short_medstream_R_L_manual.index)[i]
                      for i, x in enumerate(short_medstream_R_L_manual["id_nr"])
                      if x == keys]
        random.seed(seeds_dict[keys])
        random_visit = random.choices(pat_visits, k=1)
        strat_dict[keys] = short_medstream_R_L_manual.loc[random_visit, :].to_dict()
        strat_dict[keys] = {key: sub_value
                            for key, value in strat_dict[keys].items()
                            for sub_key, sub_value in value.items()}
    short_medstream_pat = pd.DataFrame.from_dict(strat_dict, orient='index')
    short_medstream_pat.index.name = 'id_nr'

weighted_strat = True

if weighted_strat:
    strat_total_scores = list(short_medstream['r_total_score'])
else:
    strat_total_scores = [i['r_total_score'] for keys, i in strat_dict.items()]

strat_total_score_cuts = list(np.percentile(a=strat_total_scores, q=percentiles)) + [max(strat_total_scores)]
strat_total_score_cuts_names = ['r_total_score_quant_' + str(i)
                                for i in list(range(1, len(strat_total_score_cuts) + 1))]
strat_id_dict = {'r_total_score': {}
                 , 'rf_pos': {}
                 , 'sex': {}
                 }
strat_id_cuts = {'r_total_score': {'cuts': strat_total_score_cuts,
                                   'names': strat_total_score_cuts_names}
                 , 'rf_pos': {'cuts': [0, 1], 'names': ['rf_0', 'rf_1']}
                 , 'sex': {'cuts': [0, 1], 'names': ['f', 'm']}
                 }

for var, var_categs in strat_id_cuts.items():
    strat_id_dict[var] = dict.fromkeys(var_categs['names'])
    strat_id_dict[var] = {keys: [] for keys in strat_id_dict[var].keys()}
    assigned_pats = []
    for cut_i in range(0, len(var_categs['cuts'])):
        for pat_i, dict_i in strat_dict.items():
            if (pat_i not in assigned_pats) and (dict_i[var] <= var_categs['cuts'][cut_i]):
                strat_id_dict[var][var_categs['names'][cut_i]] = strat_id_dict[var][var_categs['names'][cut_i]] + \
                                                                 [pat_i]
                assigned_pats = assigned_pats + [pat_i]


###########################
# train, val, test split  #
###########################

# random split into train, val, test
test_prop = 0.25  # 100/775 patients
val_prop = 0.25  # 100/775 patients
# train_prop = remaining  # 575/775

stratify = True
stratify_by = 'r_total_score'

if stratify:
    pat_train = []
    pat_val = []
    pat_test = []
    for keys, values in strat_id_dict[stratify_by].items():
        pat_list = values
        random.seed(split_seed)  # 99
        random.shuffle(pat_list)
        split_test = int(len(pat_list) - test_prop*len(pat_list))
        split_val = int(len(pat_list) - test_prop*len(pat_list) - val_prop*len(pat_list))

        pat_train = pat_train + pat_list[:split_val]
        pat_val = pat_val + pat_list[split_val:split_test]
        pat_test = pat_test + pat_list[split_test:]
else:
    # create list of unique patients
    pat_list = list(set(short_medstream["id_nr"]))
    pat_list.sort()  # make sure that the filenames have a fixed order before shuffling
    # split it randomly
    random.seed(split_seed)  # 197 # 99
    random.shuffle(pat_list)

    split_test = int(len(pat_list) - int(np.ceil(test_prop*len(pat_list))))
    split_val = int(len(pat_list) - int(np.ceil(test_prop*len(pat_list))) - int(np.ceil(val_prop*len(pat_list))))

    pat_train = pat_list[:split_val]
    pat_val = pat_list[split_val:split_test]
    pat_test = pat_list[split_test:]


###############################################
# comparison of split sets on a patient level #
###############################################
# prep medstream_short
# add a column "segm_set" to the data with entries 'train', 'val', 'test'
short_medstream["segm_set"] = ['train' if x in pat_train
                               else 'val' if x in pat_val
                               else 'test' if x in pat_test
                               else 'NaN'
                               for i, x in enumerate(short_medstream["id_nr"])
                               ]
print("any NaNs in segm_set: " + str(any(pd.isnull(short_medstream['segm_set'])) or any(short_medstream['segm_set'] == 'NaN')))

train_idx = [list(short_medstream.index)[i] for i, x in enumerate(short_medstream["segm_set"])
             if x == 'train']
val_idx = [list(short_medstream.index)[i] for i, x in enumerate(short_medstream["segm_set"])
           if x == 'val']
test_idx = [list(short_medstream.index)[i] for i, x in enumerate(short_medstream["segm_set"])
            if x == 'test']

set_dict = {'train': {'pat': pat_train, 'idx': train_idx},
            'val': {'pat': pat_val, 'idx': val_idx},
            'test': {'pat': pat_test, 'idx': test_idx}
            }

# prep
summary_cols_mean = ['age', 'disease_duration', 'r_total_hand_JSN', 'r_total_foot_JSN',
                     'r_total_hand_ero', 'r_total_foot_ero', 'r_total_hand', 'r_total_foot',
                     'r_total_ero', 'r_total_JSN', 'r_total_score']
summary_cols_freq = ['sex', 'rf_pos', 'ccp_pos']
for key, value in set_dict.items():
    cols_mean_df = short_medstream.loc[set_dict[key]['idx'], summary_cols_mean]
    cols_mean_means = cols_mean_df.mean(axis=0, skipna=True)
    for col_mean in summary_cols_mean:
        set_dict[key][col_mean] = cols_mean_means[col_mean]
    for col_freq in summary_cols_freq:
        cols_freq_df = short_medstream.loc[set_dict[key]['idx'], col_freq]
        set_dict[key][col_freq] = cols_freq_df.value_counts(dropna=False)/sum(pd.notnull(cols_freq_df))

short_medstream_pat['id_nr'] = short_medstream_pat.index
short_medstream_pat['segm_set'] = ['train' if x in pat_train
                                   else 'val' if x in pat_val
                                   else 'test' if x in pat_test
                                   else 'not_segm'
                                   for i, x in enumerate(short_medstream_pat["id_nr"])
                                   ]
# # table 1 (use 'tableone' python library)
pat_table_one = TableOne(data=short_medstream_pat, columns=summary_cols_mean + summary_cols_freq,
                         categorical=summary_cols_freq, groupby="segm_set", nonnormal=summary_cols_mean,
                         label_suffix=True, pval=True, sort=False)

overall_pat_table_one = TableOne(data=short_medstream_pat, columns=summary_cols_mean + summary_cols_freq,
                                 categorical=summary_cols_freq, groupby=None, nonnormal=summary_cols_mean,
                                 label_suffix=True, sort=False)


#############################################
# comparison of split sets on a image level #
#############################################

# NOTE: in a way, I should assign a weight (1 or 2) to each visit depending on how many feet (R, L or R + L)
# I have for this visit. But this will probably not affect the results very much...

# add a column "segm_set" to the data with entries 'train', 'val', 'test'
short_medstream_R_L_manual["segm_set"] = ['train' if x in pat_train
                                          else 'val' if x in pat_val
                                          else 'test' if x in pat_test
                                          else 'not_segm'
                                          for i, x in enumerate(short_medstream_R_L_manual["id_nr"])
                                          ]
print("any NaNs in segm_set: " + str(any(pd.isnull(short_medstream['segm_set']))
                                     or any(short_medstream['segm_set'] == 'NaN')))

train_idx = [list(short_medstream_R_L_manual.index)[i] for i, x in enumerate(short_medstream_R_L_manual["segm_set"])
             if x == 'train']
val_idx = [list(short_medstream_R_L_manual.index)[i] for i, x in enumerate(short_medstream_R_L_manual["segm_set"])
           if x == 'val']
test_idx = [list(short_medstream_R_L_manual.index)[i] for i, x in enumerate(short_medstream_R_L_manual["segm_set"])
            if x == 'test']

set_dict = {'train': {'pat': pat_train, 'idx': train_idx},
            'val': {'pat': pat_val, 'idx': val_idx},
            'test': {'pat': pat_test, 'idx': test_idx}
            }

# compare epidem info on image level [CAVE: THIS IS FOR ONE HAND!]
summary_cols_mean = ['age', 'disease_duration', 'r_total_one_foot_ero', 'r_total_one_foot_JSN', 'r_total_one_foot']
summary_cols_freq = ['sex', 'rf_pos', 'ccp_pos']
for key, value in set_dict.items():
    cols_mean_df = short_medstream_R_L_manual.loc[set_dict[key]['idx'], summary_cols_mean]
    cols_mean_means = cols_mean_df.mean(axis=0, skipna=True)
    for col_mean in summary_cols_mean:
        set_dict[key][col_mean] = cols_mean_means[col_mean]
    for col_freq in summary_cols_freq:
        cols_freq_df = short_medstream_R_L_manual.loc[set_dict[key]['idx'], col_freq]
        set_dict[key][col_freq] = cols_freq_df.value_counts(dropna=False)/sum(pd.notnull(cols_freq_df))

# # table 1 (use 'tableone' python library)
image_table_one = TableOne(data=short_medstream_R_L_manual, columns=summary_cols_mean + summary_cols_freq,
                           categorical=summary_cols_freq, groupby="segm_set", nonnormal=summary_cols_mean,
                           label_suffix=True, pval=True, sort=False)

overall_image_table_one = TableOne(data=short_medstream_R_L_manual, columns=summary_cols_mean + summary_cols_freq,
                                   categorical=summary_cols_freq, groupby=None, nonnormal=summary_cols_mean,
                                   label_suffix=True, sort=False)

fig0 = plt.figure()
fig0.suptitle('All Segmentation Images')
total_ax = fig0.add_subplot(131)
total_ax.set_title('r_total_one_foot')
total_ax.hist(short_medstream_R_L_manual.loc[short_medstream_R_L_manual.segm_set == 'train', 'r_total_one_foot'],
              bins=nbin, label='train',
              density=normed_flag, cumulative=cumulative_flag, histtype=histtype)
total_ax.hist(short_medstream_R_L_manual.loc[short_medstream_R_L_manual.segm_set == 'val', 'r_total_one_foot'],
              bins=nbin, alpha=0.7, label='val',
              density=normed_flag, cumulative=cumulative_flag, histtype=histtype)
total_ax.hist(short_medstream_R_L_manual.loc[short_medstream_R_L_manual.segm_set == 'test', 'r_total_one_foot'],
              bins=nbin, alpha=0.5, label='test',
              density=normed_flag, cumulative=cumulative_flag, histtype=histtype)
# total_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=3)

jsn_ax = fig0.add_subplot(132)
jsn_ax.set_title('r_total_one_foot_JSN')
jsn_ax.hist(short_medstream_R_L_manual.loc[short_medstream_R_L_manual.segm_set == 'train', 'r_total_one_foot_JSN'],
            bins=nbin, label='train',
            density=True, cumulative=cumulative_flag, histtype=histtype)
jsn_ax.hist(short_medstream_R_L_manual.loc[short_medstream_R_L_manual.segm_set == 'val', 'r_total_one_foot_JSN'],
            bins=nbin, alpha=0.7, label='val',
            density=normed_flag, cumulative=cumulative_flag, histtype=histtype)
jsn_ax.hist(short_medstream_R_L_manual.loc[short_medstream_R_L_manual.segm_set == 'test', 'r_total_one_foot_JSN'],
            bins=nbin, alpha=0.5, label='test',
            density=normed_flag, cumulative=cumulative_flag, histtype=histtype)
jsn_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=3)

ero_ax = fig0.add_subplot(133)
ero_ax.set_title('r_total_one_foot_ero')
ero_ax.hist(short_medstream_R_L_manual.loc[short_medstream_R_L_manual.segm_set == 'train', 'r_total_one_foot_ero'],
            bins=nbin, label='train',
            density=normed_flag, cumulative=cumulative_flag, histtype=histtype)
ero_ax.hist(short_medstream_R_L_manual.loc[short_medstream_R_L_manual.segm_set == 'val', 'r_total_one_foot_ero'],
            bins=nbin, alpha=0.7, label='val',
            density=normed_flag, cumulative=cumulative_flag, histtype=histtype)
ero_ax.hist(short_medstream_R_L_manual.loc[short_medstream_R_L_manual.segm_set == 'test', 'r_total_one_foot_ero'],
            bins=nbin, alpha=0.5, label='test',
            density=normed_flag, cumulative=cumulative_flag, histtype=histtype)
# ero_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=3)


########################################################
# For each segmentation pat, randomly choose one image #
########################################################
segm_ids = list(set([x for i, x in enumerate(short_medstream_R_L_manual['id_nr'])
                     if short_medstream_R_L_manual['segm_set'].iloc[i] != 'not_segm']))

if mean_strat:
    np.random.seed(chosen_seed)
    seeds = np.random.randint(low=0, high=10000, size=len(segm_ids))
    seeds_dict = dict(zip(segm_ids, seeds))
    chosen_segm = []
    for segm_id in segm_ids:
        segm_id_img_ids = [x for i, x in enumerate(short_medstream_R_L_manual['img_id'])
                           if short_medstream_R_L_manual['id_nr'].iloc[i] == segm_id
                           and short_medstream_R_L_manual['laterality_manual'].iloc[i] == 'R']
        if len(segm_id_img_ids) > 0:
            random.seed(seeds_dict[segm_id])
            chosen_segm = chosen_segm + random.choices(segm_id_img_ids, k=1)
        else:
            print('no right image for ' + str(segm_id))

    for segm_id in segm_ids:
        segm_id_img_ids = [x for i, x in enumerate(short_medstream_R_L_manual['img_id'])
                           if short_medstream_R_L_manual['id_nr'].iloc[i] == segm_id
                           and short_medstream_R_L_manual['laterality_manual'].iloc[i] == 'L']
        if len(segm_id_img_ids) > 0:
            random.seed(seeds_dict[segm_id])
            chosen_segm = chosen_segm + random.choices(segm_id_img_ids, k=1)
        else:
            print('no left image for ' + str(segm_id))
else:
    chosen_segm = list(short_medstream_pat['img_id'])

# add a column "segm_chosen" to the data with entries 'train', 'val', 'test'
short_medstream_R_L_manual["segm_chosen"] = ['yes' if x in chosen_segm
                                             else 'no' if (x not in chosen_segm
                                                           and short_medstream_R_L_manual["id_nr"].iloc[i]
                                                           in segm_ids)
                                             else 'not_segm'
                                             for i, x in enumerate(short_medstream_R_L_manual["img_id"])
                                             ]
print("any NaNs in segm_chosen: " + str(any(pd.isnull(short_medstream_R_L_manual['segm_chosen']))))

# compare chosen sets
chosen_df = deepcopy(short_medstream_R_L_manual.loc[short_medstream_R_L_manual['segm_chosen'] == 'yes', :])

# # table 1 (use 'tableone' python library)
chosen_image_table_one = TableOne(data=chosen_df, columns=summary_cols_mean + summary_cols_freq,
                                  categorical=summary_cols_freq, groupby='segm_set', nonnormal=summary_cols_mean,
                                  label_suffix=True, pval=True, sort=False)

chosen_overall_image_table_one = TableOne(data=chosen_df,
                                          columns=summary_cols_mean + summary_cols_freq,
                                          categorical=summary_cols_freq, groupby=None, nonnormal=summary_cols_mean,
                                          label_suffix=True, sort=False)

fig = plt.figure()
fig.suptitle('Chosen Segmentation Images')
total_ax = fig.add_subplot(131)
total_ax.set_title('r_total_one_foot')
total_ax.hist(chosen_df.loc[chosen_df.segm_set == 'train', 'r_total_one_foot'],
              bins=nbin, label='train',
              density=normed_flag, cumulative=cumulative_flag, histtype=histtype)
total_ax.hist(chosen_df.loc[chosen_df.segm_set == 'val', 'r_total_one_foot'],
              bins=nbin, alpha=0.7, label='val',
              density=normed_flag, cumulative=cumulative_flag, histtype=histtype)
total_ax.hist(chosen_df.loc[chosen_df.segm_set == 'test', 'r_total_one_foot'],
              bins=nbin, alpha=0.5, label='test',
              density=normed_flag, cumulative=cumulative_flag, histtype=histtype)
# total_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=3)

jsn_ax = fig.add_subplot(132)
jsn_ax.set_title('r_total_one_foot_JSN')
jsn_ax.hist(chosen_df.loc[chosen_df.segm_set == 'train', 'r_total_one_foot_JSN'],
            bins=nbin, label='train',
            density=True, cumulative=cumulative_flag, histtype=histtype)
jsn_ax.hist(chosen_df.loc[chosen_df.segm_set == 'val', 'r_total_one_foot_JSN'],
            bins=nbin, alpha=0.7, label='val',
            density=normed_flag, cumulative=cumulative_flag, histtype=histtype)
jsn_ax.hist(chosen_df.loc[chosen_df.segm_set == 'test', 'r_total_one_foot_JSN'],
            bins=nbin, alpha=0.5, label='test',
            density=normed_flag, cumulative=cumulative_flag, histtype=histtype)
jsn_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=3)

ero_ax = fig.add_subplot(133)
ero_ax.set_title('r_total_one_foot_ero')
ero_ax.hist(chosen_df.loc[chosen_df.segm_set == 'train', 'r_total_one_foot_ero'],
            bins=nbin, label='train',
            density=normed_flag, cumulative=cumulative_flag, histtype=histtype)
ero_ax.hist(chosen_df.loc[chosen_df.segm_set == 'val', 'r_total_one_foot_ero'],
            bins=nbin, alpha=0.7, label='val',
            density=normed_flag, cumulative=cumulative_flag, histtype=histtype)
ero_ax.hist(chosen_df.loc[chosen_df.segm_set == 'test', 'r_total_one_foot_ero'],
            bins=nbin, alpha=0.5, label='test',
            density=normed_flag, cumulative=cumulative_flag, histtype=histtype)
# ero_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=3)

plt.show()


##############################################################################################
# save short_medstream, short_medstream_R_L_manual, and table one on patient and image level #
##############################################################################################

# save table 1 on visit and pat level, and overall table 1
image_table_one.to_html(output_folder + os.sep + 'data_split' + os.sep +
                        "segmentation_image_table_one_strat" + str(strat_seed) +
                        "_split" + str(split_seed) + ".html")
overall_image_table_one.to_html(output_folder + os.sep + 'data_split' + os.sep +
                                "segmentation_overall_image_table_one_strat" + str(strat_seed) +
                                "_split" + str(split_seed) + ".html")
pat_table_one.to_html(output_folder + os.sep + 'data_split' + os.sep +
                      "segmentation_pat_table_one_strat" + str(strat_seed) +
                      "_split" + str(split_seed) + ".html")
overall_pat_table_one.to_html(output_folder + os.sep + 'data_split' + os.sep +
                              "segmentation_overall_pat_table_one_strat" + str(strat_seed) +
                              "_split" + str(split_seed) + ".html")

# save chosen table 1
chosen_image_table_one.to_html(output_folder + os.sep + 'data_split' + os.sep +
                               "segmentation_chosen_image_table_one_strat" + str(strat_seed) +
                               "_split" + str(split_seed) + "_chosen" + str(chosen_seed) + ".html")
chosen_overall_image_table_one.to_html(output_folder + os.sep + 'data_split' + os.sep +
                                       "segmentation_chosen_overall_image_table_one_strat" + str(strat_seed) +
                                       "_split" + str(split_seed) + "_chosen" + str(chosen_seed) + ".html")

# save short_medstream with new 'segm_set'
current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
short_medstream_filename = output_folder + os.sep + 'data_split' + os.sep + \
                           "pat_df_medstream_F_dp_img_of_int_2_summary_cols_pat_segm_sets_strat" + \
                           str(strat_seed) + "_split" + str(split_seed) + "_" + \
                           current_datetime + ".csv"
short_medstream.to_csv(short_medstream_filename)

# save df with new 'segm_set'
current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
short_medstream_R_L_manual_filename = output_folder + os.sep + 'data_split' + os.sep + \
                                      "pat_df_medstream_manual_F_dp_img_of_int_2_summary_cols_segm_sets_RL_strat" + \
                                      str(strat_seed) + "_split" + str(split_seed) + "_chosen" \
                                      + str(chosen_seed) + "_" + current_datetime + ".csv"
short_medstream_R_L_manual.to_csv(short_medstream_R_L_manual_filename)
