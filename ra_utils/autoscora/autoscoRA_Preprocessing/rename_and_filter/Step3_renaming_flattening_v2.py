# import SimpleITK as sitk
# import pydicom
# import numpy as np
# import shutil
import os
# from os.path import dirname, abspath, exists, splitext  # , basename
from os.path import join as join_path
# import itertools
# import sys
import pandas as pd
# import re
# import ntpath
import datetime
# from collections import Counter
import pydicom.valuerep  # don't import DS directly as can be changed by config
# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt


##############################################################
# Concatenate the new (corrected) categories                 #
# into new file name (= one col of metadata_df_corrected)    #
# and put that filename into the tag study_time              #
# then put all NaN images into a folders acc to their NaNs   #
# and according to their categories (feet, R/L, etc.)        #
# WHY IS IMAGE INVERTED IN SITK? WAS JUST 2 SPECIFIC IMGs    #
# --> how can I check for this and save them correctly?      #
##############################################################

print(datetime.datetime.now())

# server data
dicom_server = "/project/autoscora/autoscoRA_images/autoscoRA_original_images"
output_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/output"
# data_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/data"

# sample data
# output_folder = "/mnthome/autoscoRA/autoscoRA_Preprocessing/output/test"
# output_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/output/test"
# dicom_server = "/mnthome/autoscoRA/autoscoRA_Preprocessing/sample_xrays"
# dicom_server = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/sample_xrays"
# data_folder = "/mnthome/autoscoRA/autoscoRA_Preprocessing/data"
# data_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/data"

# read metadata dataframe
print("START: reading metadata dataframe...")
pat_df = pd.read_pickle(output_folder + os.sep + "pat_df_filenamenew.pkl")
# df = pd.read_csv(output_folder + os.sep + "pat_df_filenamenew.csv")
print("DONE: reading metadata dataframe")

# loop through the filenames and rename
print("START: renaming dicom files...")
for i in range(0, pat_df.shape[0]):
    oldpath = pat_df['filepath_old'][i]  # .replace('mnthome2', 'mnthome')
    oldpathfile = pat_df['filepathname_old'][i]  # .replace('mnthome2', 'mnthome')
    newfile = pat_df['filename_new_dupl'][i] + '.dcm'
    newpathfile = oldpath + os.sep + newfile
    os.rename(oldpathfile, newpathfile)
print("DONE: renaming dicom files")

# flattening file structure
print("START: flattening file structure...")
for i in range(0, pat_df.shape[0]):
    oldpath = pat_df['filepath_old'][i]  # .replace('mnthome2', 'mnthome')
    newfile = pat_df['filename_new_dupl'][i] + '.dcm'
    newpathfile = oldpath + os.sep + newfile
    newpath = dicom_server
    newpathnewpathfile = dicom_server + os.sep + newfile
    os.rename(newpathfile, newpathnewpathfile)
print("DONE: flattening file structure")

# remove resulting empty folders (if they indeed are empty)
print("START: removing empty folders...")

# # remove hidden files before running os.walk with topdown=False
hidden_files = [os.path.join(root, x) for root, dirs, files in os.walk(dicom_server) for x in files
                if x.startswith('.')]
for i in hidden_files:
    if os.path.exists(i):
        os.remove(i)
        print("OUTPUT: removed the hidden file: " + i)

for root, dirs, files in os.walk(dicom_server, topdown=False):
    if os.path.join(dicom_server, 'changed_metadata_dicoms') in root:
        continue
    if root != dicom_server:
        if len(files) == 0:
            for dir_name in dirs:
                os.rmdir(join_path(root, dir_name))
                if len(os.listdir(root)) == 0:
                    os.rmdir(root)
                    print(root)
        else:
            print('CAVE: there are some files within' + root + 'or its subdirectories!')
#     else:
#         for dir_name in dirs:
            # if len(os.listdir(join_path(root, dir_name))) == 0:
            #    os.rmdir(join_path(root, dir_name))
            # else:
            #     print('CAVE: there are some files within' + join_path(root, dir_name) + 'or its subdirectories!')
print("DONE: removing empty folders")

# check that renaming and flattening worked well
print("START: checking the work...")
# check that no folders left in root directory
remaining_dirs = [i for i in os.listdir(dicom_server) if (os.path.isdir(join_path(dicom_server, i)) and
                                                          'changed_metadata_dicoms' not in i)]
no_dirs_left = len(remaining_dirs) == 0
# check that no file still has old filename
other_checks = True

# passed all checks?
worked_well = no_dirs_left and other_checks
print("DONE: checking the work")

# put new filename into DICOM tag 'ReferringPhysiciansName'
print("START: put filename into DICOM tag 'ReferringPhysiciansName'...")
if worked_well:
    # create folder to put the dicoms with the changed metadata tag 'ReferringPhysiciansName'
    changed_metadata_dicoms = dicom_server + os.sep + 'changed_metadata_dicoms'
    original_metadata_dicoms = dicom_server + os.sep + 'original_metadata_dicoms'
    if not os.path.exists(changed_metadata_dicoms):
        os.makedirs(changed_metadata_dicoms)
    if not os.path.exists(original_metadata_dicoms):
        os.makedirs(original_metadata_dicoms)
    # get all filenames
    dicom_files = [os.path.join(root, x) for root, dirs, files in os.walk(dicom_server) for x in files
                   if (root == dicom_server and not x.startswith('.'))]
    nr_of_files = len(dicom_files)
    print("OUTPUT: nr of files: ", nr_of_files)

    new_tag_list = []  # list of files where
    for i in range(0, len(dicom_files)):
        # read in dicom file
        dcm_file = pydicom.dcmread(dicom_files[i])
        # put filename into ReferringPhysicianName tag
        if (0x008, 0x090) in dcm_file:
            dcm_file.ReferringPhysicianName = os.path.splitext(os.path.split(dicom_files[i])[1])[0]
        else:
            new_tag_list = new_tag_list + [os.path.split(dicom_files[i])[1]]
            dcm_file.add_new((0x008, 0x090), 'PN', os.path.splitext(os.path.split(dicom_files[i])[1])[0])
        # save header-edited dicom file at new location
        new_image_pathfile = changed_metadata_dicoms + os.sep + os.path.split(dicom_files[i])[1]
        dcm_file.save_as(filename=str(new_image_pathfile), write_like_original=True)
        os.rename(dicom_files[i], original_metadata_dicoms + os.sep + os.path.split(dicom_files[i])[1])
    print('OUTPUT: new ReferringPhysicianName tag was created (b/c not available) for:')
    print(new_tag_list)

else:
    print("worked well = " + str(worked_well) +
          ". Therefore filename was not put into DICOM tag 'ReferringPhysicianName'")

print("DONE: put filename into DICOM tag 'ReferringPhysiciansName'")

print(datetime.datetime.now())
