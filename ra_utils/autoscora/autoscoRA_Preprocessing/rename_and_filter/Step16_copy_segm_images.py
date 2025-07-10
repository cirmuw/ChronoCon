# import SimpleITK as sitk
# import pydicom
# import numpy as np
import os
# from os.path import dirname, abspath, exists, splitext  # , basename
# from os.path import join as join_path
# import itertools
# import sys
import pandas as pd
# from tableone import TableOne
# import re
# import random
# import ntpath
import datetime
# from collections import Counter
# from distutils.dir_util import copy_tree
import shutil
# from itertools import compress
# import tkinter as tk
# from tkinter import filedialog
# from copy import copy, deepcopy
# from time import sleep
# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt


print(datetime.datetime.now())

#############
# load data #
#############

dicom_server = "/project/autoscora/autoscoRA_images/H_images_of_interest_2_renamed_mirrored_inverted_dicoms"
new_server = "/project/autoscora/autoscoRA_images/segmentation_dicoms"
output_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/output"
# output_folder = "/mnthome2/autoscoRA/autoscoRA_Preprocessing/output"
pat_df_manual_version = \
    "data_split" + os.sep + \
    "pat_df_medstream_manual_H_dp_img_of_int_2_" \
    "summary_cols_segm_sets_RL_stratmean_split6543_chosen345_2019-05-03_21-12-37.csv"
# data_folder = "/mnthome2/autoscoRA/autoscoRA_Preprocessing/data"
# data_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/data"
random_one_of_BOTH_R_AND_L = True

# sample data
# output_folder = "/mnthome2/autoscoRA/autoscoRA_Preprocessing/output/test"
# output_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/output/test"
# dicom_server = "/mnthome2/autoscoRA/autoscoRA_Preprocessing/sample_xrays/sorted_by_categ_dicoms"
# dicom_server = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/sample_xrays/sorted_by_categ_dicoms"
# old_images_server = "/mnthome/autoscoRA/autoscoRA_Preprocessing/sample_xrays/changed_metadata_dicoms_copy"
# old_images_server = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/sample_xrays/changed_metadata_dicoms_copy"
# data_folder = "/mnthome2/autoscoRA/autoscoRA_Preprocessing/data"
# data_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/data"

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
                           'inappropriate_manual', 'img_of_interest',
                           'comment_status', 'comment',
                           'filename_new_dupl']


##################################################
# Copy Chosen Segmentation Images to New Folders #
##################################################

# create folders
if random_one_of_BOTH_R_AND_L:
    segm_train_dir = new_server + os.sep + 'segm_RL_train_dicoms'
    segm_val_dir = new_server + os.sep + 'segm_RL_val_dicoms'
    segm_test_dir = new_server + os.sep + 'segm_RL_test_dicoms'
else:
    segm_train_dir = new_server + os.sep + 'segm_train_dicoms'
    segm_val_dir = new_server + os.sep + 'segm_val_dicoms'
    segm_test_dir = new_server + os.sep + 'segm_test_dicoms'

for dir_name in [new_server, segm_train_dir, segm_val_dir, segm_test_dir]:
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    else:
        print("dir already exists: " + dir_name)


# select images to copy
train_images = [x + '.dcm' for i, x in enumerate(pat_df_manual['filename_manual'])
                if pat_df_manual['segm_set'].iloc[i] == 'train'
                and pat_df_manual['segm_chosen'].iloc[i] == 'yes']

val_images = [x + '.dcm' for i, x in enumerate(pat_df_manual['filename_manual'])
              if pat_df_manual['segm_set'].iloc[i] == 'val'
              and pat_df_manual['segm_chosen'].iloc[i] == 'yes']

test_images = [x + '.dcm' for i, x in enumerate(pat_df_manual['filename_manual'])
               if pat_df_manual['segm_set'].iloc[i] == 'test'
               and pat_df_manual['segm_chosen'].iloc[i] == 'yes']

# copy selected images to target directory

for chosen_img in train_images:
    old_path = dicom_server + os.sep + chosen_img
    if os.path.exists(old_path):
        new_path = segm_train_dir + os.sep + chosen_img
        if not os.path.exists(new_path):
            shutil.copy(old_path, new_path)
    else:
        print("not a file: " + chosen_img)

for chosen_img in val_images:
    old_path = dicom_server + os.sep + chosen_img
    if os.path.exists(old_path):
        new_path = segm_val_dir + os.sep + chosen_img
        if not os.path.exists(new_path):
            shutil.copy(old_path, new_path)
    else:
        print("not a file: " + chosen_img)

for chosen_img in test_images:
    old_path = dicom_server + os.sep + chosen_img
    if os.path.exists(old_path):
        new_path = segm_test_dir + os.sep + chosen_img
        if not os.path.exists(new_path):
            shutil.copy(old_path, new_path)
    else:
        print("not a file: " + chosen_img)
