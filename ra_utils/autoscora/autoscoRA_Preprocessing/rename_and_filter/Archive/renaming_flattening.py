import SimpleITK as sitk
# import pydicom
# import numpy as np
import shutil
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
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


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
dicom_server = "/data/dataset/hand/2018_MUW_RALarge/dicom"
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
# ...

# passed all checks?
worked_well = no_dirs_left and ...
print("DONE: checking the work")

# put new filename into DICOM tag 'ReferringPhysiciansName'
print("START: put filename into DICOM tag 'ReferringPhysiciansName'...")
if worked_well:
    # create folder to put the dicoms with the changed metadata tag 'ReferringPhysiciansName'
    changed_metadata_dicoms = dicom_server + os.sep + 'changed_metadata_dicoms'
    if not os.path.exists(changed_metadata_dicoms):
        os.makedirs(changed_metadata_dicoms)
    # get all filenames
    dicom_files = [os.path.join(root, x) for root, dirs, files in os.walk(dicom_server) for x in files
                   if root == dicom_server]
    nr_of_files = len(dicom_files)
    print("DONE: reading dicom files")
    print("OUTPUT: nr of files: ", nr_of_files)

# OPTION 1: pydicom
########################################################
    for i in range(0, len(dicom_files)):
        dcm_file = pydicom.dcmread(dicom_files[i])
        dcm_file.ReferringPhysicianName = os.path.split(dicom_files[i])[1]
        if dcm_file.RescaleType != "US":
            dcm_file.PhotometricInterpretation = 'MONOCHROME2'
            dcm_file.RescaleType = "US"
            dcm_file.RescaleSlope = 1
            dcm_file.RescaleIntercept = 0
            dcm_file.__delattr__(name='LossyImageCompression')
            x = dcm_file.pixel_array
            plt.imshow(x, cmap="gray")
        # dcm_file.LossyImageCompression = ''
        new_image_pathfile = changed_metadata_dicoms + os.sep + 'new' + os.path.split(dicom_files[i])[1]
        dcm_file.save_as(filename=str(new_image_pathfile), write_like_original=True)
        different_dcm = pydicom.dcmread(new_image_pathfile)
        xx = different_dcm.pixel_array
        plt.imshow(xx, cmap="gray")
        a = sitk.ReadImage(dicom_files[i])
        writer = sitk.ImageFileWriter()
        # writer.KeepOriginalImageUIDOn()
        writer.SetFileName(new_image_pathfile)
        writer.Execute(a)  # .__invert__())
        sitk.WriteImage(a, new_image_pathfile)
        sitk.Show(a)
        dcm_file_sitk = pydicom.dcmread(new_image_pathfile)

# check which tags change when saving with sitk! -> probably just photometricinterpretation! If that's the case,
# # pydicom does not change anything.
# # sitk changes PhotometricInterpretation, Bits Stored becomes Bits Allocated - 1
    # deletes some tags and adds others, changes format from XXE-2 to 0.0XX
# check which tags are different when you save without writer.KeepOriginalImageUIDOn()
    # prob not much difference (series/study/frame of reference codes?)
# SOLUTION:
    # for now, pydicom:
    #   check if pydicom works on server and use it (don't change raw data headers for now) +
    #   put 'inv'/'noninv' into filename
    # later, use sitk:
        # simply check whether it is MONOCHROME1 and if yes: invert image and change
        # to MONOCHROME2 (automatically in sitk?)
        # Reading in pixels is always same in pydicom, sitk but interpretation of those always same pixels
        # in a dicom viewer (Fiji, Horos) differs depending on header! And sitk saves as MONOCHROME2 --> wrongly read!

# OPTION 2: sitk
########################################################
    # for each file, put filename into 'series_description_code'
    for i in range(0, len(dicom_files)):  # loop through image files
        # create reader object for current dicom file
        reader = sitk.ImageFileReader()
        reader.SetFileName(dicom_files[i])
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        tags = reader.GetMetaDataKeys()
        image = reader.Execute()
        tags2 = image.GetMetaDataKeys()
        new_image_pathfile = changed_metadata_dicoms + os.sep + os.path.split(dicom_files[i])[1]
        if ('0028|1054' in tags2) or ('0008|0090' in tags2):
            if ('0028|1054' in tags2) and (image.GetMetaData('0028|1054') != 'US'):
                image.SetMetaData('0028|1054', 'US')
            if '0008|0090' in tags2:
                image.SetMetaData('0008|0090', os.path.split(dicom_files[i])[1])
                print(i)
                print(image.GetMetaData('0008|0090'))
            writer = sitk.ImageFileWriter()
            writer.KeepOriginalImageUIDOn()
            writer.SetFileName(new_image_pathfile)
            writer.Execute(image)  # .__invert__())
        else:
            shutil.copy2(dicom_files[i], new_image_pathfile)
            # os.rename(dicom_files[i], new_image_pathfile)
else:
    print("worked well = " + str(worked_well) +
          ". Therefore filename was not put into DICOM tag 'ReferringPhysicianName'")

"""
if worked_well:
    # get all filenames
    dicom_files = [os.path.join(root, x) for root, dirs, files in os.walk(dicom_server) for x in files]
    nr_of_files = len(dicom_files)
    print("DONE: reading dicom files")
    print("OUTPUT: nr of files: ", nr_of_files)

    # for each file, put filename into 'series_description_code'
    for i in range(0, len(dicom_files)):  # loop through image files
        # create reader object for current dicom file
        reader = sitk.ImageFileReader()
        reader.SetFileName(dicom_files[i])
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        tags = reader.GetMetaDataKeys()
        if '0008|0030' in tags:
            reader.SetMetaData('0008|0030', os.path.split(dicom_files[i])[1])
            print(i)
            print(reader.GetMetaData('0008|0030'))
else:
    print("worked well = " + str(worked_well) +
          ". Therefore filename was not put into DICOM tag 'study_time'")
"""
print("DONE: put filename into DICOM tag 'series_description_code'")
