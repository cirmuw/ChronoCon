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
# from distutils.dir_util import copy_tree
import shutil
# from itertools import compress
# import tkinter as tk
# from tkinter import filedialog
# import pydicom
# from copy import copy, deepcopy
# from time import sleep
# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt


print(datetime.datetime.now())

#############
# load data #
#############

dicom_server = "/project/autoscora/autoscoRA_images/H_images_of_interest_1_dicoms"
old_images_server = "/project/autoscora/autoscoRA_images/sorted_by_categ_dicoms"
# output_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/output"
output_folder = "/mnthome2/autoscoRA/autoscoRA_Preprocessing/output"
pat_df_manual_version = "pat_df_manual_2019-04-13_19-28-59_Step8_2019-04-13_21-11-23.csv"

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


################################################################
#              find missing images and candidates              #
################################################################

pat_df_manual_2 = pat_df_manual.copy()
pat_df_manual_2['body_laterality_view_manual'] = pat_df_manual_2['bodypart_manual'].astype(str) + '_' \
                                                 + pat_df_manual_2['laterality_manual'] + '_' \
                                                 + pat_df_manual_2['view_position_manual']

pat_df_manual_2['id_date'] = pat_df_manual_2['pat_id'].astype(str) + '_' + pat_df_manual_2['study_date'].astype(str)
unique_id_dates = pat_df_manual_2.id_date.unique()

candidates_df = pd.DataFrame(columns=['filename_new_dupl', 'sub_sub_dir',
                                      'pixel_action_filepath'
                                      ])
duplicates_df = pd.DataFrame(columns=['filename_new_dupl', 'sub_sub_dir',
                                      'pixel_action_filepath',
                                      'filename_manual'])
for id_date_i in unique_id_dates:
    pat_df_manual_i = pat_df_manual_2[['id_date',
                                       'body_laterality_view_manual',
                                       'filename_new_dupl',
                                       'sub_sub_dir',
                                       'pixel_action_filepath',
                                       'filename_manual'
                                       ]].loc[pat_df_manual_2['id_date'] == id_date_i]
    nr_HdpL = sum(pat_df_manual_i['body_laterality_view_manual'] == 'H_L_dp')
    nr_HdpR = sum(pat_df_manual_i['body_laterality_view_manual'] == 'H_R_dp')
    if nr_HdpL == 0 or nr_HdpR == 0:
        pat_df_manual_i_sub = pat_df_manual_i[['filename_new_dupl', 'sub_sub_dir',
                                               'pixel_action_filepath'
                                               ]].loc[pat_df_manual_i['body_laterality_view_manual'] ==
                                                      'TBD_TBD_TBD', :]
        candidates_df = candidates_df.append(pat_df_manual_i_sub, ignore_index=True)
    if nr_HdpL > 1:
        pat_df_manual_i_dupl_L = pat_df_manual_i[['filename_new_dupl', 'sub_sub_dir',
                                                  'pixel_action_filepath',
                                                  'filename_manual'
                                                  ]].loc[pat_df_manual_i['body_laterality_view_manual'] ==
                                                         'H_L_dp', :]
        duplicates_df = duplicates_df.append(pat_df_manual_i_dupl_L)
    if nr_HdpR > 1:
        pat_df_manual_i_dupl_R = pat_df_manual_i[['filename_new_dupl', 'sub_sub_dir',
                                                  'pixel_action_filepath',
                                                  'filename_manual'
                                                  ]].loc[pat_df_manual_i['body_laterality_view_manual'] ==
                                                         'H_R_dp', :]
        duplicates_df = duplicates_df.append(pat_df_manual_i_dupl_R)


####################################################################
#              copy candidate dcm files into a folder              #
####################################################################
# NOTE: they can be manually assessed from that new folder and
# still remain in old folder as well because Fiji only looks at TBD -
# which those in the old folder will no longer be if I check them as missing candidates
# --> no unnecessary double check will occur due to copying

list_of_candidates_paths = candidates_df['pixel_action_filepath']
if not (dicom_server.startswith("/home/cir/tdeimel/") or dicom_server.startswith("/project")):
    list_of_candidates_paths = [re.sub("^/home/cir/tdeimel", "/mnthome2", x) for x in list_of_candidates_paths]

# create target directory
candidate_dir = re.sub(os.path.split(dicom_server)[1] + "$", "H_dp_missing_candidate_1_dicoms", dicom_server)
if not os.path.isdir(candidate_dir):
    os.mkdir(candidate_dir)
else:
    print("dir already exists: " + candidate_dir)

# copy candidate images to target directory
for candidate_filename in list_of_candidates_paths:
    if os.path.exists(candidate_filename):
        new_path = candidate_dir + os.sep + os.path.basename(candidate_filename)
        shutil.copy(candidate_filename, new_path)
    else:
        print("not a file: " + candidate_filename)

print("end of script")
