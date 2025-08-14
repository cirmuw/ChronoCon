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

old_images_server = "/project/autoscora/autoscoRA_images/sorted_by_categ_dicoms"
dicom_server = "/project/autoscora/autoscoRA_images/pixel_action_finished_dicoms_FEET"
rot_dir = dicom_server + os.sep + 'rotated_fixed_excluding_rot_split_dicoms'
old_rot_dir = dicom_server + os.sep + 'rotated_fixed_dicoms'
split_dir = dicom_server + os.sep + 'splitpoint_incl_rotsplit_fixed_dicoms'
four_dir = dicom_server + os.sep + 'four_limbs_fixed_dicoms'

# output_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/output"
# output_folder = "/mnthome2/autoscoRA/autoscoRA_Preprocessing/output"
output_folder = "/Volumes/Research/Diplomarbeiten/Diplomand_innen_Thomas_Deimel/Kanta_Chrysa/spreadsheets"
pat_df_manual_version = "pat_df_split/pat_df_manual_2021-05-06_16-15-24.csv"

# read in pat_df_manual
pat_df_manual_path = output_folder + os.sep + pat_df_manual_version
pat_df_manual = pd.read_csv(pat_df_manual_path)
# create backup copy of pat_df_manual
pat_df_manual_original = pat_df_manual.copy()
# select feet
pat_df_manual_feet = pat_df_manual[pat_df_manual["bodypart_manual"] == "F"].copy().reset_index()
pat_df_manual = pat_df_manual_feet

# manual columns
manual_columns_wo_comment = ['bodypart_manual', 'laterality_manual', 'view_position_manual',
                             'inverted_manual', 'rotated_manual', 'black_manual', 'operated_manual',
                             'inappropriate_manual', 'category_manual', 'filename_manual',
                             'splitpoint_manual']

# ------ copy pixel action and unchanged images of interest to new dir ------ #

# assign correct image location
rot_nosplit_filenames = [os.path.splitext(i)[0] for i in os.listdir(rot_dir) if not i.startswith(".")]
for i in [i for i, x in enumerate(pat_df_manual['filename_new_dupl']) if x in rot_nosplit_filenames]:
    pat_df_manual.loc[i, 'pixel_action_filepath'] = \
        rot_dir + os.sep + pat_df_manual.loc[i, 'filename_new_dupl'] + ".dcm"
# check: pat_df_manual[[i in rot_nosplit_filenames for i in pat_df_manual['filename_new_dupl']]].loc[:, 'pixel_action_filepath']

# for split, four_limbs, and rotated+split images, pixel_action_filepath is already correct

# choose correct images
images_of_interest_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
                          if pat_df_manual.loc[i, 'bodypart_manual'] == "F"
                          and pat_df_manual.loc[i, 'laterality_manual'] in ["R", "L"]
                          and pat_df_manual.loc[i, 'view_position_manual'] == "dp"
                          and pat_df_manual.loc[i, 'black_manual'] == "BOk"
                          and pat_df_manual.loc[i, 'inappropriate_manual'] == "app"
                          and not bool(re.search("split_done", pat_df_manual.loc[i, 'comment']))
                          ]

# add image of interest column to pat_df_manual
pat_df_manual["img_of_interest_FEET"] = 0
pat_df_manual.loc[images_of_interest_idx, "img_of_interest_FEET"] = 1

# images
images_of_interest_images = list(pat_df_manual.loc[images_of_interest_idx, 'pixel_action_filepath'])

# dict
images_of_interest_dict = dict(zip(images_of_interest_idx, images_of_interest_images))

# directory
images_of_interest_dir = re.sub(os.path.basename(dicom_server), "", dicom_server) + \
                         "F_images_of_interest_1_dicoms"

if not os.path.isdir(images_of_interest_dir):
    os.mkdir(images_of_interest_dir)
else:
    print("dir already exists: " + images_of_interest_dir)

# copy images of interest to new directory
for idx, img in images_of_interest_dict.items():
    if os.path.exists(img):
        print(os.path.basename(img))
        new_path = images_of_interest_dir + os.sep + os.path.basename(img)
        shutil.copy(img, new_path)
    else:
        print("not a file: " + img)

# for copying in bash instead
# with open(output_folder + os.sep + 'pat_df_img_of_interest_FEET/imofint_paths.txt', mode='wt', encoding='utf-8') as myfile:
#     myfile.write('\n'.join(images_of_interest_images))
# cd /Volumes/Research/Diplomarbeiten/Diplomand_innen_Thomas_Deimel/Kanta_Chrysa/spreadsheets/pat_df_img_of_interest_FEET
# cat imofint_paths.txt | head -200 | xargs -J % cp % /project/autoscora/autoscoRA_images/F_images_of_interest_1_dicoms


# ------ save the new pat_df ------ #
# current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# new_pat_df_filename = output_folder + os.sep + os.path.splitext(pat_df_manual_version)[0] + \
#                       "_Step8_" + current_datetime + ".csv"
# pat_df_manual.to_csv(new_pat_df_filename)
