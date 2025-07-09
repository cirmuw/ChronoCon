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

dicom_server = "/project/autoscora/autoscoRA_images/F_images_of_interest_1_dicoms"
old_images_server = "/project/autoscora/autoscoRA_images/sorted_by_categ_dicoms"
# output_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/output"
output_folder = "/Volumes/Research/Diplomarbeiten/Diplomand_innen_Thomas_Deimel/Kanta_Chrysa/spreadsheets"
pat_df_manual_version = "pat_df_candidates/pat_df_manual_2021-05-17_21-15-27.csv"

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

# img of interest dir
images_of_interest_dir = re.sub(os.path.basename(dicom_server), "", dicom_server) + \
                         "F_images_of_interest_1_dicoms"

##########################################################
# put true candidates into F_images_of_interest_1_dicoms #
##########################################################

true_candidates_idx = [i for i, x in enumerate(pat_df_manual["comment"])
                       if "true_candidate" in x and pat_df_manual.iloc[i, :]["bodypart_manual"] == "F"
                       and pat_df_manual.iloc[i, :]["view_position_manual"] == "dp"]

true_candidate_paths = pat_df_manual.loc[true_candidates_idx, "pixel_action_filepath"]

# set img_of_interest_FEET = 1
pat_df_manual.loc[true_candidates_idx, "img_of_interest_FEET"] = 1

# copy candidate images to target directory
for true_cand in true_candidate_paths:
    if os.path.exists(true_cand):
        new_path = images_of_interest_dir + os.sep + os.path.basename(true_cand)
        if not os.path.exists(new_path):
            shutil.copy(true_cand, new_path)
            print("copied")
    else:
        print("not a file: " + true_cand)

##############################################
#            find duplicate images           #
##############################################

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
    nr_FdpL = sum(pat_df_manual_i['body_laterality_view_manual'] == 'F_L_dp')
    nr_FdpR = sum(pat_df_manual_i['body_laterality_view_manual'] == 'F_R_dp')
    if nr_FdpL == 0 or nr_FdpR == 0:
        pat_df_manual_i_sub = pat_df_manual_i[['filename_new_dupl', 'sub_sub_dir',
                                               'pixel_action_filepath'
                                               ]].loc[pat_df_manual_i['body_laterality_view_manual'] ==
                                                      'TBD_TBD_TBD', :]
        candidates_df = candidates_df.append(pat_df_manual_i_sub, ignore_index=True)
    if nr_FdpL > 1:
        pat_df_manual_i_dupl_L = pat_df_manual_i[['filename_new_dupl', 'sub_sub_dir',
                                                  'pixel_action_filepath',
                                                  'filename_manual'
                                                  ]].loc[pat_df_manual_i['body_laterality_view_manual'] ==
                                                         'F_L_dp', :]
        duplicates_df = duplicates_df.append(pat_df_manual_i_dupl_L)
    if nr_FdpR > 1:
        pat_df_manual_i_dupl_R = pat_df_manual_i[['filename_new_dupl', 'sub_sub_dir',
                                                  'pixel_action_filepath',
                                                  'filename_manual'
                                                  ]].loc[pat_df_manual_i['body_laterality_view_manual'] ==
                                                         'F_R_dp', :]
        duplicates_df = duplicates_df.append(pat_df_manual_i_dupl_R)

# exclude inapp and co and set 'TBD'
excluded_inapp = []
duplicates_df_app = pd.DataFrame(columns=['filename_new_dupl', 'sub_sub_dir',
                                          'pixel_action_filepath',
                                          'filename_manual'])
for i in duplicates_df.iterrows():
    pass
    if pat_df_manual.loc[i[0], "inappropriate_manual"] != "app":
        excluded_inapp = excluded_inapp + [i[0]]
    else:
        if pat_df_manual.loc[i[0], "img_of_interest_FEET"] != 1:
            excluded_inapp = excluded_inapp + [i[0]]
        else:
            duplicates_df_app = duplicates_df_app.append(duplicates_df.loc[i[0], :])
            print(pat_df_manual.loc[i[0], "img_of_interest_FEET"])

####################################################################
#              copy duplicate dcm files into a folder              #
####################################################################
# NOTE: they can be manually assessed from that new folder and
# still remain in old folder as well because Fiji only looks at TBD -
# which those in the old folder will no longer be if I check them as missing candidates
# --> no unnecessary double check

# set TBD
for i in duplicates_df_app.iterrows():
    pat_df_manual.loc[i[0], "splitpoint_manual"] = 'TBD'

# save the new pat_df
# current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# new_pat_df_filename = output_folder + os.sep + os.path.splitext(pat_df_manual_version)[0] + \
#                       "_Step10_" + current_datetime + ".csv"
# pat_df_manual.to_csv(new_pat_df_filename)


list_of_duplicates_paths = duplicates_df_app['pixel_action_filepath']
if not (dicom_server.startswith("/home/cir/tdeimel/") or dicom_server.startswith("/project")):
    list_of_duplicates_paths = [re.sub("^/home/cir/tdeimel", "/mnthome2", x) for x in list_of_duplicates_paths]

# create target directory
duplicate_dir = re.sub(os.path.split(dicom_server)[1] + "$", "F_dp_duplicate_1_dicoms", dicom_server)
if not os.path.isdir(duplicate_dir):
    os.mkdir(duplicate_dir)
else:
    print("dir already exists: " + duplicate_dir)

# copy duplicate images to target directory
for duplicate_filename in list_of_duplicates_paths:
    if os.path.exists(duplicate_filename):
        new_path = duplicate_dir + os.sep + os.path.basename(duplicate_filename)
        if not os.path.exists(new_path):
            shutil.copy(duplicate_filename, new_path)
    else:
        print("not a file: " + duplicate_filename)

# for copying in bash instead
with open(output_folder + os.sep + 'pat_df_duplicates/duplicate_paths.txt', mode='wt', encoding='utf-8') as myfile:
    myfile.write('\n'.join(list(list_of_duplicates_paths)))
# cd /Volumes/Research/Diplomarbeiten/Diplomand_innen_Thomas_Deimel/Kanta_Chrysa/spreadsheets/pat_df_candidates
# cat duplicate_paths.txt | head -200 | xargs -J % cp % '/project/autoscora/autoscoRA_images/F_dp_duplicate_1_dicoms'

duplicates_names = duplicates_df_app['filename_new_dupl'].to_list()
sorting_nr = [int(re.sub("_.*", "", s)) for s in duplicates_names]
duplicates_names_sorted = [x for _, x in sorted(zip(sorting_nr, duplicates_names))]
with open(output_folder + os.sep + 'pat_df_duplicates/duplicates_names.txt', mode='wt', encoding='utf-8') as myfile:
    myfile.write("',\n'".join(["'"] + duplicates_names_sorted + ["'"]))

print("end of script")
