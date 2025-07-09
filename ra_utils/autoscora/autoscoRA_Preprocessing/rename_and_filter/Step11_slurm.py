import SimpleITK as sitk
import pydicom
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
from copy import copy, deepcopy
# from time import sleep
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


print(datetime.datetime.now())

#############
# load data #
#############

dicom_server = "/project/autoscora/autoscoRA_images/H_images_of_interest_2_slurm_dicoms"
old_server = "/project/autoscora/autoscoRA_images/H_images_of_interest_1_dicoms"
output_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/output"
# output_folder = "/mnthome2/autoscoRA/autoscoRA_Preprocessing/output"
pat_df_manual_version = "pat_df_manual_2019-04-14_20-46-23_img_of_interest2_2019-04-15_12-53-41.csv"

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

images_of_interest_1_files = os.listdir(old_server)

###############
# copy images #
###############

# find all images with img_of_interest == 2
img_of_interest_2_idx = [i for i, x in enumerate(pat_df_manual["img_of_interest"])
                         if x == 2]

# check that they're all in img_of_interst_1 dir
all(i + ".dcm" in images_of_interest_1_files for i in
    pat_df_manual.loc[img_of_interest_2_idx, "filename_new_dupl"])

# create dir
img_of_interest_2_dir = dicom_server
if not os.path.isdir(img_of_interest_2_dir):
    os.mkdir(img_of_interest_2_dir)
else:
    print("dir already exists: " + img_of_interest_2_dir)

# copy images to new folder
old_paths = old_server + os.sep + pat_df_manual.loc[img_of_interest_2_idx, "filename_new_dupl"] + ".dcm"
for file_of_interest in old_paths:
    if os.path.exists(file_of_interest):
        new_path = img_of_interest_2_dir + os.sep + os.path.basename(file_of_interest)
        if not os.path.exists(new_path):
            shutil.copy(file_of_interest, new_path)
            print("copied")
    else:
        print("not a file: " + file_of_interest)
