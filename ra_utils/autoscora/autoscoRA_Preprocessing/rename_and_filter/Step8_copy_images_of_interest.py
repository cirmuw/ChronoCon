import SimpleITK as sitk
# import pydicom
import numpy as np
import os
# from os.path import dirname, abspath, exists, splitext  # , basename
# from os.path import join as join_path
# import itertools
import sys
import pandas as pd
import re
# import ntpath
import datetime
# from collections import Counter
# from distutils.dir_util import copy_tree
import shutil
# from itertools import compress
# import tkinter as tk
# from tkinter import filedialog
import pydicom
from copy import copy, deepcopy
# from time import sleep
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


print(datetime.datetime.now())

#############
# load data #
#############

dicom_server = "/project/autoscora/autoscoRA_images/pixel_action_finished_dicoms"
old_images_server = "/project/autoscora/autoscoRA_images/sorted_by_categ_dicoms"
output_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/output"
# output_folder = "/mnthome2/autoscoRA/autoscoRA_Preprocessing/output"
pat_df_manual_version = "pat_df_manual_2019-04-13_19-28-59.csv"

# sample data
# output_folder = "/mnthome2/autoscoRA/autoscoRA_Preprocessing/output/test"
# output_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/output/test"
# dicom_server = "/mnthome2/autoscoRA/autoscoRA_Preprocessing/sample_xrays/sorted_by_categ_dicoms"
# dicom_server = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/sample_xrays/sorted_by_categ_dicoms"
# old_images_server = "/mnthome/autoscoRA/autoscoRA_Preprocessing/sample_xrays/changed_metadata_dicoms_copy"
# old_images_server = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/sample_xrays/changed_metadata_dicoms_copy"
# data_folder = "/mnthome2/autoscoRA/autoscoRA_Preprocessing/data"
# data_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/data"

# read in medstream
# pat_df_medstream = pd.read_excel(data_folder + os.sep + "AutoScoreRA_Data_Pseudonymized_v1_17102018.xlsx")

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
manual_columns_wo_comment = ['bodypart_manual', 'laterality_manual', 'view_position_manual',
                             'inverted_manual', 'rotated_manual', 'black_manual', 'operated_manual',
                             'inappropriate_manual', 'category_manual', 'filename_manual',
                             'splitpoint_manual']


# ------ copy pixel action and unchanged images of interest to new dir ------ #
# fix errors from previous scripts
anonym_dir = '/project/autoscora/autoscoRA_images/pixel_action_finished_dicoms/anonym_prob_all_fixed_dicoms'
rot_dir = '/project/autoscora/autoscoRA_images/pixel_action_finished_dicoms/rotated_fixed_dicoms'
anonym_rot_idx = [i for i, x in enumerate(pat_df_manual['pixel_action_filepath']) if x in [anonym_dir, rot_dir]]
pat_df_manual.loc[anonym_rot_idx, 'pixel_action_filepath'] = \
    pat_df_manual.loc[anonym_rot_idx, 'pixel_action_filepath'] + os.sep + \
    pat_df_manual.loc[anonym_rot_idx, 'filename_new_dupl']

pat_df_manual['pixel_action_filepath'] = pat_df_manual['pixel_action_filepath'] + ".dcm"

# indices
images_of_interest_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
                          if pat_df_manual.loc[i, 'bodypart_manual'] == "H"
                          and pat_df_manual.loc[i, 'laterality_manual'] in ["R", "L"]
                          and pat_df_manual.loc[i, 'view_position_manual'] == "dp"
                          and pat_df_manual.loc[i, 'black_manual'] == "BOk"
                          and pat_df_manual.loc[i, 'inappropriate_manual'] == "app"
                          and not bool(re.search("split_done", pat_df_manual.loc[i, 'comment']))
                          ]
# images
images_of_interest_images = list(pat_df_manual.loc[images_of_interest_idx, 'pixel_action_filepath'])

# dict
images_of_interest_dict = dict(zip(images_of_interest_idx, images_of_interest_images))

# directory
images_of_interest_dir = re.sub(os.path.basename(dicom_server), "", dicom_server) + \
                         "H_images_of_interest_1_dicoms"

if not os.path.isdir(images_of_interest_dir):
    os.mkdir(images_of_interest_dir)
else:
    print("dir already exists: " + images_of_interest_dir)

# add image of interest column to pat_df_manual
pat_df_manual["img_of_interest"] = 0
pat_df_manual.loc[images_of_interest_idx, "img_of_interest"] = 1

# copy images of interest to new directory
for idx, img in images_of_interest_dict.items():
    if os.path.exists(img):
        new_path = images_of_interest_dir + os.sep + os.path.basename(img)
        shutil.copy(img, new_path)
    else:
        print("not a file: " + img)

# ------ save the new pat_df ------ #
# current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# new_pat_df_filename = output_folder + os.sep + os.path.splitext(pat_df_manual_version)[0] + \
#                       "_Step8_" + current_datetime + ".csv"
# pat_df_manual.to_csv(new_pat_df_filename)
