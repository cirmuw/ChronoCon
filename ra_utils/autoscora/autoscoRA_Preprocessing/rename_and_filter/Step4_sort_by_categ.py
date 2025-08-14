# import SimpleITK as sitk
# import pydicom
# import numpy as np
# import shutil
import os
# from os.path import dirname, abspath, exists, splitext  # , basename
# from os.path import join as join_path
import itertools
# import sys
import pandas as pd
import re
# import ntpath
import datetime
# from collections import Counter
# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
from distutils.dir_util import copy_tree
import shutil

print(datetime.datetime.now())

# load data
dicom_server = "/project/autoscora/autoscoRA_images/sorted_by_categ_dicoms"
old_images_server = "/project/autoscora/autoscoRA_images/changed_metadata_dicoms_copy"
output_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/output"

# sample data
# output_folder = "/mnthome/autoscoRA/autoscoRA_Preprocessing/output/test"
# output_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/output/test"
# dicom_server = "/mnthome/autoscoRA/autoscoRA_Preprocessing/sample_xrays/sorted_by_categ_dicoms"
# dicom_server = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/sample_xrays/sorted_by_categ_dicoms"
# old_images_server = "/mnthome/autoscoRA/autoscoRA_Preprocessing/sample_xrays/changed_metadata_dicoms_copy"
# old_images_server = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/sample_xrays/changed_metadata_dicoms_copy"
# data_folder = "/mnthome/autoscoRA/autoscoRA_Preprocessing/data"
# data_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/data"

# create necessary dirs, copy dicoms if this has not been done manually
if not os.path.exists(dicom_server):
    os.makedirs(dicom_server)

if not os.path.exists(old_images_server):
    os.makedirs(old_images_server)
    print("START: copying changed_metadata_dicoms dir into" + old_images_server + "...")
    to_copy = re.sub("_copy$", "", old_images_server)
    copy_tree(to_copy, old_images_server)
    print("DONE: copying changed_metadata_dicoms dir into" + old_images_server + "...")
else:
    if len([i for i in os.listdir(old_images_server) if not i.startswith('.')]) == 0:
        print("START: copying changed_metadata_dicoms dir into" + old_images_server + "...")
        to_copy = re.sub("_copy$", "", old_images_server)
        copy_tree(to_copy, old_images_server)
        print("DONE: copying changed_metadata_dicoms dir into" + old_images_server + "...")
    else:
        print("GOOD: " + old_images_server + " is not empty.")

# read metadata dataframe
print("START: reading metadata dataframe...")
pat_df = pd.read_pickle(output_folder + os.sep + "pat_df_filenamenew.pkl")
# pat_df = pd.read_csv(output_folder + os.sep + "pat_df_filenamenew.csv")
print("DONE: reading metadata dataframe")

# find unique categories
unique_categs = pat_df.category_new.unique()

# create dir structure
print("START: creating new directories within" + dicom_server + "...")
# # main and sub categories
H_dicoms = dicom_server + os.sep + 'H_dicoms'
H_R_dicoms = H_dicoms + os.sep + 'H_R_dicoms'
H_L_dicoms = H_dicoms + os.sep + 'H_L_dicoms'
H_B_dicoms = H_dicoms + os.sep + 'H_B_dicoms'
H_NaN_dicoms = H_dicoms + os.sep + 'H_NaN_dicoms'

F_dicoms = dicom_server + os.sep + 'F_dicoms'
F_R_dicoms = F_dicoms + os.sep + 'F_R_dicoms'
F_L_dicoms = F_dicoms + os.sep + 'F_L_dicoms'
F_B_dicoms = F_dicoms + os.sep + 'F_B_dicoms'
F_NaN_dicoms = F_dicoms + os.sep + 'F_NaN_dicoms'

O_dicoms = dicom_server + os.sep + 'O_dicoms'
NaN_dicoms = dicom_server + os.sep + 'NaN_dicoms'

dir_names = [H_dicoms, H_R_dicoms, H_L_dicoms, H_B_dicoms, H_NaN_dicoms,
             F_dicoms, F_R_dicoms, F_L_dicoms, F_B_dicoms, F_NaN_dicoms,
             O_dicoms, NaN_dicoms]

for i in range(0, len(dir_names)):
    if not os.path.exists(dir_names[i]):
        os.makedirs(dir_names[i])

# # sub sub categories
main_categs = [(os.path.split(dir_name)[1][0:-7] + '_') for dir_name in dir_names
               if dir_name not in [H_dicoms, F_dicoms]]

sub_sub_dir_names = []
for i in range(0, len(main_categs)):
    subcategs_i_bool = [bool(re.search("^" + main_categs[i] + ".*$", subcateg))
                        for subcateg in unique_categs]
    subcategs_i_M = list(unique_categs[subcategs_i_bool])
    subcategs_i = list(set([re.sub("_(MOne|MTwo|NaN)$", "", subcateg_M) for subcateg_M in subcategs_i_M]))
    if main_categs[i] in ['O_', 'NaN_']:
        dirname_j = eval(main_categs[i] + 'dicoms')
        sub_sub_dir_names = sub_sub_dir_names + [dirname_j]
    else:
        for j in range(0, len(subcategs_i)):
            dirname_j = eval(main_categs[i] + 'dicoms') + os.sep + subcategs_i[j] + '_dicoms'
            sub_sub_dir_names = sub_sub_dir_names + [dirname_j]
            if not os.path.exists(dirname_j):
                os.makedirs(dirname_j)
print("DONE: creating new directories within" + dicom_server)

# put subsub, sub, main dirs in pat_df
print("START: putting names of respective new directory in pat_df...")
pat_df["sub_sub_dir"] = ['NaN'] * pat_df.shape[0]
pat_df["sub_dir"] = ['NaN'] * pat_df.shape[0]
pat_df["main_dir"] = ['NaN'] * pat_df.shape[0]
for i in range(0, pat_df.shape[0]):
    if bool(re.search("^O_.*$", pat_df["category_new"][i])):
        pat_df["sub_sub_dir"][i] = O_dicoms
        pat_df["sub_dir"][i] = O_dicoms
        pat_df["main_dir"][i] = O_dicoms
    elif bool(re.search("^NaN_.*$", pat_df["category_new"][i])):
        pat_df["sub_sub_dir"][i] = NaN_dicoms
        pat_df["sub_dir"][i] = NaN_dicoms
        pat_df["main_dir"][i] = NaN_dicoms
    else:
        matching_subsubdirs = [j for j in sub_sub_dir_names
                               if (re.sub("_(MOne|MTwo|NaN)$", "", pat_df["category_new"][i])
                                   == re.sub("_dicoms$", "", os.path.split(j)[1])
                                   and re.sub("_dicoms$", "", os.path.split(j)[1]) not in ['O', 'NaN'])
                               ]
        if len(matching_subsubdirs) > 1:
            raise("current category_new (" + pat_df["category_new"][i] +
                  ") matches more than one entry in sub_sub_dir_names.")
        pat_df["sub_sub_dir"][i] = matching_subsubdirs[0]
        pat_df["sub_dir"][i] = os.path.split(pat_df["sub_sub_dir"][i])[0]
        pat_df["main_dir"][i] = os.path.split(pat_df["sub_dir"][i])[0]
print("DONE: putting names of respective new directory in pat_df")

# check that no NaNs in pat_df[sub_sub_dir]
NaNs_in_subsubdir = len([i for i in pat_df.sub_sub_dir if i == "NaN"]) == 0
if NaNs_in_subsubdir:
    print("GOOD: no NaNs in pat_df[sub_sub_dir]")
else:
    print("BAD: NaNs in pat_df[sub_sub_dir]")

# save changed pat_df as pat_df_sortedbycateg
print("START: saving df with new filenames...")
pat_df.to_pickle(output_folder + os.sep + "pat_df_sortedbycateg.pkl")
pat_df.to_csv(output_folder + os.sep + "pat_df_sortedbycateg.csv")
print("DONE: saving df with new filenames")

# move files from changed_metadata_dicoms_copy to the new dirs in sorted_by_categ_dicoms
print("START: moving files to new directories...")
for i in range(0, pat_df.shape[0]):
    old_path_name = os.path.join(old_images_server, pat_df.filename_new_dupl[i] + '.dcm')
    new_path_name = os.path.join(pat_df.sub_sub_dir[i], pat_df.filename_new_dupl[i] + '.dcm')
    os.rename(old_path_name, new_path_name)
print("DONE: moving files to new directories")

# check that no images left in changed_metadata_dicoms_copy and if so, delete the empty dir
old_img_list = [i for i in os.listdir(old_images_server) if not i.startswith('.')]
old_img_list_hidden = [i for i in os.listdir(old_images_server)]
if len(old_img_list_hidden) == 0:
    os.rmdir(old_images_server)
    print("GOOD: no images left in " + old_images_server + ". Removed the empty dir.")
elif (len(old_img_list_hidden) > 0) and (len(old_img_list) == 0):
    print("OUTPUT: hidden files left in " + old_images_server + ":")
    print(old_img_list_hidden)
    shutil.rmtree(old_images_server, ignore_errors=True)
    print("GOOD: removed hidden files and the dir " + old_images_server)
if len(old_img_list) > 0:
    print("BAD: some non-hidden images left in " + old_images_server + ". Did not remove the non-empty dir.")
    print("OUTPUT: (hidden and non-hidden) images left:")
    print(old_img_list_hidden)

# print number of images in the new dir structure
sorted_img_list = list(itertools.chain.from_iterable([files for root, dirs, files in os.walk(dicom_server)]))
sorted_img_list = [f for f in sorted_img_list if not f.startswith('.')]
original_old_img_list = [f for f in os.listdir(re.sub("_copy$", "", old_images_server)) if not f.startswith('.')]
sorted_not_in_original = [x for x in sorted_img_list if x not in original_old_img_list]
original_not_in_sorted = [x for x in original_old_img_list if x not in sorted_img_list]
nr_of_files = len(sorted_img_list)
nr_of_original_files = len(original_old_img_list)
print("OUTPUT: Nr of files in the new directory structure: " + str(nr_of_files))
print("OUTPUT: Nr of files in the original directory: " + str(nr_of_original_files))

print(datetime.datetime.now())
