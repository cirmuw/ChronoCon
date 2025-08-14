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

dicom_server = "/project/autoscora/autoscoRA_images/sorted_by_categ_dicoms"
# old_images_server = "/project/autoscora/autoscoRA_images/changed_metadata_dicoms_copy"
# output_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/output"
output_folder = "/mnthome2/autoscoRA/autoscoRA_Preprocessing/output"
pat_df_manual_version = "pat_df_manual_2019-04-08_10-12-47.csv"
one_note_issues_version = "one_note_issues_v2.csv"

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

# read in one_note_issues csv
one_note_issues_path = output_folder + os.sep + one_note_issues_version

one_note_issues_df = pd.read_csv(one_note_issues_path)

one_note_issues_dict_raw = one_note_issues_df.to_dict('list')  # turn to dict
one_note_issues_dict_nonan = {key: [element for element in value if not pd.isnull(element)]
                              for key, value in one_note_issues_dict_raw.items()}  # delete nans
required_action = {key: value[0] for key, value in one_note_issues_dict_nonan.items()}
one_note_issues_dict = {key: [str.strip(i) for i in value[1:]]
                        for key, value in one_note_issues_dict_nonan.items()}  # remove white spaces at start/end

# ------ anonym_prob ------ #
# anonym prob from issues.csv
anonym_prob_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
                   if x in one_note_issues_dict['anonym_prob']]
missing_anonym_prob = [x for x in one_note_issues_dict['anonym_prob']
                       if x not in list(pat_df_manual['filename_new_dupl'])]
print("missing_anonym_prob: " + str(len(missing_anonym_prob)))

# find all (incl. previously commented) images with commment anonym prob
anonym_prob_comment_idx = [i for i, x in enumerate(pat_df_manual['comment'])
                           if bool(re.search("anonym_prob", x))]

# indices
anonym_prob_all_idx = sorted(anonym_prob_idx + anonym_prob_comment_idx)

# add comment where not already done
old_anonym_prob_all_comment = pat_df_manual.loc[anonym_prob_all_idx, "comment"]
new_anonym_prob_all_comment = [x if "anonym_prob" in x
                               else "anonym_prob" if x in ["none", "TBD"] else x + "__anonym_prob"
                               for x in old_anonym_prob_all_comment]
pat_df_manual.loc[anonym_prob_all_idx, "comment"] = new_anonym_prob_all_comment
pat_df_manual.loc[anonym_prob_all_idx, "comment_status"] = "ComYes"

# images
anonym_prob_all_images = list(pat_df_manual.loc[anonym_prob_all_idx, 'sub_sub_dir'] +
                              os.sep +
                              pat_df_manual.loc[anonym_prob_all_idx, 'filename_new_dupl'] +
                              '.dcm')

# directory
anonym_prob_all_dir = re.sub(os.path.split(dicom_server)[1] + "$",
                             "pixel_action_dicoms/anonym_prob_all_dicoms",
                             dicom_server)


# ------ put images in a folder to go through again------ #

go_through_again_idx = {anonym_prob_all_dir: {"idx": anonym_prob_all_idx, "img": anonym_prob_all_images}
                        }

# set >= 1 TBD for those I have to go through again
# for directory, files in go_through_again_idx.items():
#     pat_df_manual.loc[files["idx"], "splitpoint_manual"] = "TBD"

# create pixel action dir
pixel_action_dir = re.sub(os.path.split(dicom_server)[1] + "$",
                          "pixel_action_dicoms",
                          dicom_server)
if not os.path.isdir(pixel_action_dir):
    os.mkdir(pixel_action_dir)
else:
    print("dir already exists: " + pixel_action_dir)

# create target directories
for target_dir in go_through_again_idx.keys():
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    else:
        print("dir already exists: " + target_dir)

# copy images to target directories
for directory, files in go_through_again_idx.items():
    for img in files["img"]:
        if os.path.exists(img):
            new_path = directory + os.sep + os.path.basename(str(img))
            shutil.copy(str(img), new_path)
        else:
            print("not a file: " + str(img))


# ------ check > 45° rotated images ------ #
# angle (degrees)
rotation_E = 90
rotation_SE = 90
rotation_S = 180
rotation_SW = -90
rotation_W = -90

# indices
Rot_E_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
             if bool(re.search("Rot_E", pat_df_manual.loc[i, 'comment']))]
Rot_SE_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
              if bool(re.search("Rot_SE", pat_df_manual.loc[i, 'comment']))]
Rot_S_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
             if bool(re.search("Rot_S", pat_df_manual.loc[i, 'comment']))
             and not bool(re.search("Rot_SW", pat_df_manual.loc[i, 'comment']))
             and not bool(re.search("Rot_SE", pat_df_manual.loc[i, 'comment']))]
Rot_SW_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
              if bool(re.search("Rot_SW", pat_df_manual.loc[i, 'comment']))]
Rot_W_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
             if bool(re.search("Rot_W", pat_df_manual.loc[i, 'comment']))]

# put angles, indices in a dict
Rot_idx_dict = {"Rot_E": {"idx": Rot_E_idx, "img": [], "angle": rotation_E},
                "Rot_SE": {"idx": Rot_SE_idx, "img": [], "angle": rotation_SE},
                "Rot_S": {"idx": Rot_S_idx, "img": [], "angle": rotation_S},
                "Rot_SW": {"idx": Rot_SW_idx, "img": [], "angle": rotation_SW},
                "Rot_W": {"idx": Rot_W_idx, "img": [], "angle": rotation_W}}
# add image paths to dict
# hands
""" 
for Rot_idx_keys, Rot_idx_values in Rot_idx_dict.items():
    Rot_idx_values["img"] = list(pat_df_manual.loc[Rot_idx_values["idx"], 'sub_sub_dir'] +
                                 os.sep +
                                 pat_df_manual.loc[Rot_idx_values["idx"], 'filename_new_dupl'] +
                                 '.dcm')
                                 
# check directory
rotated_check_dir = re.sub(os.path.split(dicom_server)[1] + "$",
                           "pixel_action_dicoms/rotated_check_dir",
                           dicom_server)
if not os.path.isdir(rotated_check_dir):
    os.mkdir(rotated_check_dir)
else:
    print("dir already exists: " + rotated_check_dir)

# put images into check dirs
for directory, files in Rot_idx_dict.items():
    if not os.path.isdir(rotated_check_dir + os.sep + directory):
        os.mkdir(rotated_check_dir + os.sep + directory)
    else:
        print("dir already exists: " + rotated_check_dir + os.sep + directory)
    for img in files["img"]:
        print(img)
        if os.path.exists(img):
            new_path = rotated_check_dir + os.sep + directory + os.sep + os.path.basename(str(img))
            shutil.copy(str(img), new_path)
        else:
            print("not a file: " + str(img))

"""

# feet

# check directory
rotated_check_dir = re.sub(os.path.split(dicom_server)[1] + "$",
                           "pixel_action_dicoms/rotated_check_dir",
                           dicom_server)
if not os.path.isdir(rotated_check_dir):
    os.mkdir(rotated_check_dir)
else:
    print("dir already exists: " + rotated_check_dir)

for path_addon in ["F_B_dicoms/F_B_dp_dicoms", "F_L_dicoms/F_L_dp_dicoms",
                   "F_NaN_dicoms/F_NaN_dp_dicoms", "F_NaN_dicoms/F_NaN_NaN_dicoms"]:

    for Rot_idx_keys, Rot_idx_values in Rot_idx_dict.items():
        Rot_idx_values["img"] = list([dicom_server + os.sep + path_addon +
                                      os.sep +
                                      os.path.basename(i) + '.dcm'
                                      for i in pat_df_manual.loc[Rot_idx_values["idx"], 'filename_new_dupl']])

    # put images into check dirs
    for directory, files in Rot_idx_dict.items():
        if not os.path.isdir(rotated_check_dir + os.sep + directory):
            os.mkdir(rotated_check_dir + os.sep + directory)
        else:
            print("dir already exists: " + rotated_check_dir + os.sep + directory)
        for img in files["img"]:
            print(img)
            if os.path.exists(img):
                new_path = rotated_check_dir + os.sep + directory + os.sep + os.path.basename(str(img))
                shutil.copy(str(img), new_path)
            else:
                print("not a file: " + str(img))

# set splitpoint = 'TBD'
for directory, files in Rot_idx_dict.items():
    for idx in files["idx"]:
        pat_df_manual.loc[idx, "splitpoint_manual"] = "TBD"

current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
new_pat_df_filename = output_folder + os.sep + "pat_df_manual_2019-04-08_10-12-47" + \
                      "_Step7a_" + current_datetime + ".csv"
pat_df_manual.to_csv(new_pat_df_filename)


# ------ save pat_df to csv ------ #
current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
new_pat_df_filename = output_folder + os.sep + os.path.splitext(pat_df_manual_version)[0] + \
                      "_Step7a_" + current_datetime + ".csv"
pat_df_manual.to_csv(new_pat_df_filename)
