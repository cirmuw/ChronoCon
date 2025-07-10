# import SimpleITK as sitk
# import pydicom
# import numpy as np
import os
# from os.path import dirname, abspath, exists, splitext  # , basename
# from os.path import join as join_path
# import itertools
# import sys
import pandas as pd
import re
# import ntpath
import datetime
# from collections import Counter
# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
# from distutils.dir_util import copy_tree
import shutil
# from itertools import compress
# import tkinter as tk
# from tkinter import filedialog

print(datetime.datetime.now())

#############
# load data #
#############

# hands
dicom_server = "/project/autoscora/autoscoRA_images/sorted_by_categ_dicoms"
# old_images_server = "/project/autoscora/autoscoRA_images/changed_metadata_dicoms_copy"
# output_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/output"
output_folder = "/mnthome2/autoscoRA/autoscoRA_Preprocessing/output"
pat_df_manual_version = "pat_df_manual_2019-04-05_18-07-06.csv"  # hands

# feet (Chrysa/Rheum Server)
dicom_server = "/Volumes/Research/Diplomarbeiten/Diplomand_innen_Thomas_Deimel/Kanta_Chrysa/images/F_Dicoms"
output_folder = "/Volumes/Research/Diplomarbeiten/Diplomand_innen_Thomas_Deimel/Kanta_Chrysa/spreadsheets"
pat_df_manual_version = "pat_df_manual_2020-11-06_17-43-03.csv"  # feet

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


# ------ after manual correction, look for remaining NaNs ------ #

# find NaN_manual images (filter out inapp)
NaN_idx = [i for i, x in enumerate(pat_df_manual['category_manual'])
           if bool(re.search("NaN", x)) and pat_df_manual.loc[i, 'inappropriate_manual'] == 'app']

# find __ images (filter out inapp)
uu_idx = [i for i, x in enumerate(pat_df_manual['category_manual'])
          if bool(re.search("__", x)) and pat_df_manual.loc[i, 'inappropriate_manual'] == 'app']

# combine them
NaN_uu_idx = sorted(NaN_idx + uu_idx)

# create target directory
NaN_uu_dir = re.sub(os.path.split(dicom_server)[1] + "$",
                    "notes_issues_dicoms/NaN_uu_dicoms",
                    dicom_server)
if not os.path.isdir(NaN_uu_dir):
    os.mkdir(NaN_uu_dir)
else:
    print("dir already exists: " + NaN_uu_dir)

# filepaths of images
list_of_NaN_uu_paths = pat_df_manual.loc[NaN_uu_idx, 'sub_sub_dir'] + os.sep +\
                       pat_df_manual.loc[NaN_uu_idx, 'filename_new_dupl'] + '.dcm'
if not (dicom_server.startswith("/home/cir/tdeimel/") or dicom_server.startswith("/project")):
    list_of_candidates_paths = [re.sub("^/home/cir/tdeimel", "/mnthome2", x) for x in list_of_NaN_uu_paths]

# copy images to target directory
for NaN_uu_filename in list_of_NaN_uu_paths:
    if os.path.exists(NaN_uu_filename):
        new_path = NaN_uu_dir + os.sep + os.path.basename(NaN_uu_filename)
        shutil.copy(NaN_uu_filename, new_path)
    else:
        print("not a file: " + NaN_uu_filename)

# set splitpoint to TBD (so Fiji finds them)
pat_df_manual.loc[NaN_uu_idx, "splitpoint_manual"] = "TBD"

# save pat_df_manual
current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
new_pat_df_filename = output_folder + os.sep + os.path.splitext(pat_df_manual_version)[0] + \
                      "_Step6b_" + current_datetime + ".csv"
pat_df_manual.to_csv(new_pat_df_filename)
