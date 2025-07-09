import SimpleITK as sitk
import pydicom
import numpy as np
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
import ntpath
import matplotlib
matplotlib.use("TkAgg")  # slurm gives error still, but calling python via bash script works
import matplotlib.pyplot as plt


inverted_files_root = "/mnthome/autoscoRA/autoscoRA_Preprocessing/sample_xrays/changed_metadata_dicoms/inverted"
not_inverted_files_root = "/mnthome/autoscoRA/autoscoRA_Preprocessing/sample_xrays/changed_metadata_dicoms/not_inverted"

# get all filenames
dicom_files_inverted = [os.path.join(root, x) for root, dirs, files in os.walk(inverted_files_root) for x in files
                        if root == inverted_files_root]
dicom_files_not_inverted = [os.path.join(root, x) for root, dirs, files in os.walk(not_inverted_files_root) for x in files
                            if root == not_inverted_files_root]


# Read meta data of files into a dataframe
print("START: creating dataframe of datafiles' metadata...")
# inverted

tags_dict_df = {}

if 'pat_df_inv' in globals():
    del pat_df_inv

for i in range(0, len(dicom_files_inverted)):  # loop through image files
    # create reader object for current dicom file
    reader = sitk.ImageFileReader()
    reader.SetFileName(dicom_files_inverted[i])
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    tags = reader.GetMetaDataKeys()

    # read metadata into dictionary
    # dict_pat = {md_name: reader.GetMetaData(md_tag).encode("utf-8", "replace")
    #             for md_name, md_tag in tags_df.items() if md_tag in tags}
    # dict_pat = {md_name: reader.GetMetaData(md_tag).encode("utf-8", "surrogateescape").decode("utf-8", "replace")
    #             for md_name, md_tag in tags_df.items() if md_tag in tags}
    # dict_pat = {md_name: (reader.GetMetaData(md_tag).encode("utf-8", "replace").decode("utf-8", "replace"))
    #             for md_name, md_tag in tags_df.items() if md_tag in tags}
    dict_pat = {md_tag: reader.GetMetaData(md_tag).encode("utf-8", "replace").decode("utf-8", "replace").strip()
                for md_tag in tags}  # strip() deletes leading and trailing white spaces
    # include old file name in dict
    dict_pat['filepathname_old'] = dicom_files_inverted[i]
    dict_pat['filepath_old'] = ntpath.split(dicom_files_inverted[i])[0]
    dict_pat['filename_old'] = ntpath.split(dicom_files_inverted[i])[1]

    # turn dict into df
    i_df = pd.DataFrame(dict_pat, index=[i], columns=dict_pat.keys())

    # append df for this image file to df for all image files
    if 'pat_df_inv' not in globals():
        pat_df_inv = i_df
    else:
        pat_df_inv = pat_df_inv.append(i_df)

    if i == len(dicom_files_inverted)-1:
        print("DONE: creating dataframe of datafiles' metadata")

# check that all old filenames are unique
unique_entries_filename = len(list(dict.fromkeys(pat_df_inv['filename_old'])))
if unique_entries_filename == pat_df_inv.shape[0]:
    print('GOOD: all old filenames seem to be unique!')
else:
    print('BAD: not all old filenames seem to be unique!')

# not inverted
tags_dict_df = {}

if 'pat_df_not_inv' in globals():
    del pat_df_not_inv
# inverted
for i in range(0, len(dicom_files_not_inverted)):  # loop through image files
    # create reader object for current dicom file
    reader = sitk.ImageFileReader()
    reader.SetFileName(dicom_files_not_inverted[i])
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    tags = reader.GetMetaDataKeys()

    # read metadata into dictionary
    # dict_pat = {md_name: reader.GetMetaData(md_tag).encode("utf-8", "replace")
    #             for md_name, md_tag in tags_df.items() if md_tag in tags}
    # dict_pat = {md_name: reader.GetMetaData(md_tag).encode("utf-8", "surrogateescape").decode("utf-8", "replace")
    #             for md_name, md_tag in tags_df.items() if md_tag in tags}
    # dict_pat = {md_name: (reader.GetMetaData(md_tag).encode("utf-8", "replace").decode("utf-8", "replace"))
    #             for md_name, md_tag in tags_df.items() if md_tag in tags}
    dict_pat = {md_tag: reader.GetMetaData(md_tag).encode("utf-8", "replace").decode("utf-8", "replace").strip()
                for md_tag in tags}  # strip() deletes leading and trailing white spaces
    # include old file name in dict
    dict_pat['filepathname_old'] = dicom_files_not_inverted[i]
    dict_pat['filepath_old'] = ntpath.split(dicom_files_not_inverted[i])[0]
    dict_pat['filename_old'] = ntpath.split(dicom_files_not_inverted[i])[1]

    # turn dict into df
    i_df = pd.DataFrame(dict_pat, index=[i], columns=dict_pat.keys())

    # append df for this image file to df for all image files
    if 'pat_df_not_inv' not in globals():
        pat_df_not_inv = i_df
    else:
        pat_df_not_inv = pat_df_not_inv.append(i_df)

    if i == len(dicom_files_not_inverted)-1:
        print("DONE: creating dataframe of datafiles' metadata")

# check that all old filenames are unique
unique_entries_filename = len(list(dict.fromkeys(pat_df_not_inv['filename_old'])))
if unique_entries_filename == pat_df_not_inv.shape[0]:
    print('GOOD: all old filenames seem to be unique!')
else:
    print('BAD: not all old filenames seem to be unique!')


# from old renaming_flattening checks
worked_well = True
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
