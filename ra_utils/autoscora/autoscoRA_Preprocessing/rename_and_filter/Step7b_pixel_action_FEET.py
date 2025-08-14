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

dicom_server = "/project/autoscora/autoscoRA_images/pixel_action_finished_dicoms_FEET"
# old_images_server = "/project/autoscora/autoscoRA_images/changed_metadata_dicoms_copy"
# output_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/output"
output_folder = "/mnthome2/autoscoRA/autoscoRA_Preprocessing/output"
# pat_df_manual_version = "pat_df_manual_2019-04-08_10-12-47_Step7a_2019-04-08_13-11-42.csv"

# feet
# dicom_server = "/Volumes/Research/Diplomarbeiten/Diplomand_innen_Thomas_Deimel/Kanta_Chrysa/images/F_Dicoms"
output_folder = "/Volumes/Research/Diplomarbeiten/Diplomand_innen_Thomas_Deimel/Kanta_Chrysa/spreadsheets"
pat_df_manual_version = "pat_df_manual_2020-11-06_17-43-03.csv"
pat_df_manual_addon = "pat_df_feet_check/pat_df_manual_2021-04-20_16-37-27.csv"

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
pat_df_manual_addon_path = output_folder + os.sep + pat_df_manual_addon

# pat_df_manual_path = filedialog.askopenfilename()
pat_df_manual = pd.read_csv(pat_df_manual_path)
pat_df_manual_addon = pd.read_csv(pat_df_manual_addon_path)

# create backup copy of pat_df_manual
pat_df_manual_original = pat_df_manual.copy()
pat_df_manual_addon_original = pat_df_manual_addon.copy()

# joint the dataframes
pat_df_manual_minus_addon = pat_df_manual[[i not in pat_df_manual_addon['filename_new_dupl'].to_list()
                                           for i in pat_df_manual['filename_new_dupl']]]
pat_df_manual_incl_addon = pd.concat([pat_df_manual_minus_addon, pat_df_manual_addon])
pat_df_manual = pat_df_manual_incl_addon
pat_df_manual_feet = pat_df_manual[pat_df_manual["bodypart_manual"] == "F"]
# manual columns
manual_columns_wo_comment = ['bodypart_manual', 'laterality_manual', 'view_position_manual',
                             'inverted_manual', 'rotated_manual', 'black_manual', 'operated_manual',
                             'inappropriate_manual', 'category_manual', 'filename_manual',
                             'splitpoint_manual']


if not os.path.isdir(dicom_server):
    os.mkdir(dicom_server)
else:
    print("dir already exists: " + dicom_server)

#######################
# classes & functions #
#######################


class XRay:
    def __init__(self, filepath, donotload=False):
        self.filepath = filepath
        self.error = []
        if donotload:
            self.img = 'not_loaded'
        else:
            self.img = self.read_xray()
            # self.reload_xray()

    def reload_xray(self, reduce_sitk_dims=True):
        """sets self.img to a pydicom image"""
        self.img = pydicom.dcmread(self.filepath)
        try:
            pixels = self.img.pixel_array
        except ValueError:
            self.error = self.error + [sys.exc_info()]
            sitkim = sitk.ReadImage(self.filepath)
            sitk_pixel = sitk.GetArrayFromImage(sitkim)
            if reduce_sitk_dims:
                sitk_pixel = sitk_pixel.squeeze()
            self.img.PixelData = sitk_pixel.tobytes()

    def read_xray(self, reduce_sitk_dims=True):
        """returns a pydicom image"""
        img = pydicom.dcmread(self.filepath)
        try:
            pixels = img.pixel_array
        except ValueError:
            self.error = self.error + [sys.exc_info()]
            sitkim = sitk.ReadImage(self.filepath)
            sitk_pixel = sitk.GetArrayFromImage(sitkim)
            if reduce_sitk_dims:
                sitk_pixel = sitk_pixel.squeeze()
            img.PixelData = sitk_pixel.tobytes()
        finally:
            return img

    def check_dim(self):
        if self.img == "not_loaded":
            raise Exception("Please load xray first using 'yourimage.read_xray()'.")
        else:
            array_shape = np.shape(self.img.pixel_array)
            header_shape = (self.img.Rows, self.img.Columns)
            return array_shape == header_shape

    def close_xray(self):
        self.img = 'not_loaded'

    def pixel_monochrome(self, mono=2, maximum='maxbit'):
        """
        returns an inverted version of the pixel array of self.img but does not change self.img
        inverting alternatives:
        manual
        numpy.invert(close_img)
        from skimage import util
        img = data.camera()
        inverted_img = util.invert(img)
        cv2
        PIL
        """
        if maximum == 'maxpresent':
            max_pixel_present = np.max(self.img.pixel_array)
            max_pixel = max_pixel_present
        elif maximum == 'maxbit':
            stored_bits = self.img.BitsStored
            max_pixel = 2 ** stored_bits - 1
        else:
            raise Exception("maximum should be either 'maxpixel' or 'maxbit'")
        if mono == 2 and self.img.PhotometricInterpretation == "MONOCHROME1":
            inv_pixel = max_pixel - self.img.pixel_array
        elif mono == 1 and self.img.PhotometricInterpretation == "MONOCHROME2":
            inv_pixel = max_pixel - self.img.pixel_array
        else:
            inv_pixel = self.img.pixel_array
        return inv_pixel

    def img_monochrome(self, mono=2, maximum='maxbit', change_photometric_interpretation=False):
        """inverts pixel array of self.img"""
        self.img.PixelData = self.pixel_monochrome(mono=mono, maximum=maximum).tobytes()
        if change_photometric_interpretation:
            if mono == 2:
                new_mono = 'MONOCHROME2'
            elif mono == 1:
                new_mono = 'MONOCHROME1'
            else:
                new_mono = self.img.PhotometricInterpretation
            self.img.PhotometricInterpretation = new_mono

    def plot_img(self):
        if self.img == "not_loaded":
            raise Exception("Please load xray first using 'yourimage.read_xray()'.")
        else:
            plt.figure()
            plt.imshow(self.img.pixel_array, cmap="gray")

    def pixel_blacken(self, roi, percentile=10, mono=2):
        """colors region of interest in intensity of percentile-th pixel value percentile
        roi = nested list of upper left and lower right endpoints of
        diagonal through rectangular roi [[x1, x2], [y1, y2]]"""
        flattened_pixels = self.img.pixel_array.flatten()
        if self.img.PhotometricInterpretation == "MONOCHROME1":
            percentile_value = np.percentile(flattened_pixels, 100-percentile)
            perc_print = 100-percentile
        else:
            percentile_value = np.percentile(flattened_pixels, percentile)
            perc_print = percentile
        print(perc_print, "-th percentile pixel value: ", percentile_value)
        x1 = roi[0][0]
        x2 = roi[0][1]
        y1 = roi[1][0]
        y2 = roi[1][1]
        assert x1 < x2
        assert y1 < y2
        blackened_pixels = self.img.pixel_array
        blackened_pixels[y1:y2, x1:x2] = percentile_value
        return blackened_pixels

    def img_blacken(self, roi, percentile=10, mono=2):
        """blackens out roi in pixel_array of self.img"""
        self.img.PixelData = self.pixel_blacken(roi=roi, percentile=percentile, mono=mono).tobytes()

    def pixel_rotate(self, angle=180):
        """returns a (counter-clockwise) rotated version of the pixel array of self.img
        angle can be 90, 180, -90"""
        if self.img == "not_loaded":
            raise Exception("Please load xray first using 'yourimage.read_xray()'.")
        else:
            rot_mult = int(angle/90)
            rot_pixel = np.rot90(self.img.pixel_array, k=rot_mult)
            return rot_pixel

    def img_rotate(self, angle=180):
        """rotates pixel array of self.img"""
        rot_pixel = self.pixel_rotate(angle=angle)
        self.img.Rows = np.shape(rot_pixel)[0]
        self.img.Columns = np.shape(rot_pixel)[1]
        self.img.PixelData = rot_pixel.tobytes()

    def pixel_split(self, split_x, split_y=None, plot=False):
        """splits pixel array of self.img at x and optionally y.
        Plots if plot=True.
        Returns the image (not just the pixel array) with original image's tags if return_img=True"""

        if self.img == "not_loaded":
            raise Exception("Please load xray first using 'yourimage.read_xray()'.")
        else:
            if split_y is None:
                split_pixel_left = self.img.pixel_array[:, :split_x]
                split_pixel_right = self.img.pixel_array[:, split_x + 1:]
                return_dict = {"left": split_pixel_left,
                               "right": split_pixel_right}
            else:
                split_pixel_topleft = self.img.pixel_array[:split_y, :split_x]
                split_pixel_topright = self.img.pixel_array[:split_y, split_x + 1:]
                split_pixel_bottomleft = self.img.pixel_array[split_y + 1:, :split_x]
                split_pixel_bottomright = self.img.pixel_array[split_y + 1:, split_x + 1:]
                return_dict = {"topleft": split_pixel_topleft,
                               "topright": split_pixel_topright,
                               "bottomleft": split_pixel_bottomleft,
                               "bottomright": split_pixel_bottomright}
            if plot:
                fig = plt.figure()
                grid = plt.GridSpec(2, 4, )
                big = fig.add_subplot(grid[:2, :2])
                big.imshow(self.img.pixel_array, cmap="gray")
                if split_y is None:
                    left = fig.add_subplot(grid[:2, 2])
                    right = fig.add_subplot(grid[:2, 3])
                    left.imshow(split_pixel_left, cmap="gray")
                    right.imshow(split_pixel_right, cmap="gray")
                else:
                    topleft = fig.add_subplot(grid[0, 2])
                    topright = fig.add_subplot(grid[0, 3])
                    bottomleft = fig.add_subplot(grid[1, 2])
                    bottomright = fig.add_subplot(grid[1, 3])
                    topleft.imshow(split_pixel_topleft, cmap="gray")
                    topright.imshow(split_pixel_topright, cmap="gray")
                    bottomleft.imshow(split_pixel_bottomleft, cmap="gray")
                    bottomright.imshow(split_pixel_bottomright, cmap="gray")
                fig.show()
            return return_dict

    def img_split(self, split_x, split_y=None, plot=False):
        """return split images with new rows, cols set but tags the same as original"""
        split_pixel = self.pixel_split(split_x=split_x, split_y=split_y, plot=plot)
        img_dict = {}
        for panel, pixels in split_pixel.items():
            try:
                new_xray = deepcopy(self)
            except TypeError:
                print(sys.exc_info())
                new_xray = copy(self)
            new_xray.img.Rows = np.shape(pixels)[0]
            new_xray.img.Columns = np.shape(pixels)[1]
            new_xray.img.PixelData = pixels.tobytes()
            img_dict[panel] = new_xray
        return img_dict

    def img_write(self, filename, write_like_original):
        self.img.save_as(filename=str(filename), write_like_original=write_like_original)


# ------ anonym_prob ------ #
# indices
anonym_prob_idx = [i for i, x in enumerate(pat_df_manual['comment'])
                   if bool(re.search("anonym_prob", x)) and pat_df_manual.iloc[i, ]["bodypart_manual"] == "F"]
# images
anonym_prob_all_images = list(pat_df_manual.loc[anonym_prob_idx, 'sub_sub_dir'] +
                              os.sep +
                              pat_df_manual.loc[anonym_prob_idx, 'filename_new_dupl'] +
                              '.dcm')
anonym_prob_idx_img_dict = dict(zip(anonym_prob_idx, anonym_prob_all_images))

# directory
anonym_prob_all_dir = dicom_server + os.sep + "anonym_prob_all_fixed_dicoms_FEET"
if not os.path.isdir(anonym_prob_all_dir):
    os.mkdir(anonym_prob_all_dir)
else:
    print("dir already exists: " + anonym_prob_all_dir)

# read in left upper and right lower corners of areas to blacken out
anonym_prob_dict = {key: {"img": value, "corners": []} for key, value in anonym_prob_idx_img_dict.items()}
corners = {"42_20001211_H_NaN_NaN_MTwo_1": [[0, 1976], [6256, 6672]],
           "42_20001211_H_NaN_NaN_MTwo_2": [[272, 1752], [6184, 6576]],
           "260_20010307_H_NaN_NaN_MOne_1": [[604, 1018], [1294, 1400]]
           }
for idx, values in anonym_prob_dict.items():
    img_name = os.path.splitext(os.path.basename(values["img"]))[0]
    anonym_prob_dict[idx]["corners"] = corners[img_name]

# set pixels in those areas to 5th percentile of pixel intensity (or mean pixel intensity?)
anonym_prob_error_dict = {}
for idx, values in anonym_prob_dict.items():
    try:
        xray = XRay(values["img"])
    except:
        anonym_prob_error_dict[idx] = {"type": "other", "info": sys.exc_info()}
        print(idx, values["img"], sys.exc_info())
    else:
        xray.img_blacken(roi=values["corners"], percentile=10, mono=2)
        plt.figure()
        plt.imshow(xray.pixel_monochrome(mono=2), cmap="gray")
        xray.img_write(filename=anonym_prob_all_dir + os.sep + os.path.basename(values["img"]),
                       write_like_original=True)


# ------ > 45° rotated images ------ #
# angle (degrees)
rotation_E = 90
rotation_SE = 180
rotation_S = 180
rotation_SW = 180
rotation_W = -90

# indices
Rot_E_idx = [i for i, x in enumerate(pat_df_manual_feet['filename_new_dupl'])
             if bool(re.search("Rot_E", pat_df_manual_feet.iloc[i, ]['comment']))]
Rot_SE_idx = [i for i, x in enumerate(pat_df_manual_feet['filename_new_dupl'])
              if bool(re.search("Rot_SE", pat_df_manual_feet.iloc[i, ]['comment']))]
Rot_S_idx = [i for i, x in enumerate(pat_df_manual_feet['filename_new_dupl'])
             if bool(re.search("Rot_S", pat_df_manual_feet.iloc[i, ]['comment']))
             and not bool(re.search("Rot_SW", pat_df_manual_feet.iloc[i, ]['comment']))
             and not bool(re.search("Rot_SE", pat_df_manual_feet.iloc[i, ]['comment']))]
Rot_SW_idx = [i for i, x in enumerate(pat_df_manual_feet['filename_new_dupl'])
              if bool(re.search("Rot_SW", pat_df_manual_feet.iloc[i, ]['comment']))]
Rot_W_idx = [i for i, x in enumerate(pat_df_manual_feet['filename_new_dupl'])
             if bool(re.search("Rot_W", pat_df_manual_feet.iloc[i, ]['comment']))]

# put angles, indices in a dict
Rot_idx_dict = {"Rot_E": {"idx": Rot_E_idx, "img": [], "angle": rotation_E},
                "Rot_SE": {"idx": Rot_SE_idx, "img": [], "angle": rotation_SE},
                "Rot_S": {"idx": Rot_S_idx, "img": [], "angle": rotation_S},
                "Rot_SW": {"idx": Rot_SW_idx, "img": [], "angle": rotation_SW},
                "Rot_W": {"idx": Rot_W_idx, "img": [], "angle": rotation_W}}
# add image paths to dict
for Rot_idx_keys, Rot_idx_values in Rot_idx_dict.items():
    Rot_idx_values["img"] = list(pat_df_manual_feet.iloc[Rot_idx_values["idx"], ]['sub_sub_dir'] +
                                 os.sep +
                                 pat_df_manual_feet.iloc[Rot_idx_values["idx"], ]['filename_new_dupl'] +
                                 '.dcm')

# directory
rotated_all_dir = dicom_server + os.sep + "rotated_fixed_dicoms"
if not os.path.isdir(rotated_all_dir):
    os.mkdir(rotated_all_dir)
else:
    print("dir already exists: " + rotated_all_dir)

# load images and rotate them + save rotated images
# fig_it = "A"
# pos_xy = "+100+100"
rotated_error_dict = {}
for orientation_type, orientation_dict in Rot_idx_dict.items():
    angle = orientation_dict["angle"]
    for image in orientation_dict["img"]:
        try:
            xray = XRay(image)
        except:
            rotated_error_dict[os.path.basename(image)] = {"type": "other", "info": sys.exc_info()}
            print(os.path.basename(image), sys.exc_info())
        else:
            xray.img_rotate(angle=angle)
            # print(os.path.splitext(os.path.basename(image))[0], " -- ", angle)
            # fig_name = os.path.splitext(os.path.basename(image))[0]
            # plt.figure(fig_name)
            # thismanager = plt.get_current_fig_manager()
            # thismanager.window.wm_geometry(pos_xy)
            # plt.imshow(xray.pixel_monochrome(mono=2), cmap="gray")
            # plt.show()
            xray.img_write(filename=rotated_all_dir + os.sep + os.path.basename(image),
                           write_like_original=True)
            # if "fig_name_old" in locals():
            #     plt.close(fig_name_old)
            # fig_name_old = fig_name
            # if fig_it == "A":
            #     fig_it = "B"
            #     pos_xy = "+700+100"
            # elif fig_it == "B":
            #     fig_it = "A"
            #     pos_xy = "+100+100"

# # replace by new path for changed images (not necessary for split images b/c already new path in subsubdir)
rot_idx = []
for idx, values in Rot_idx_dict.items():
    rot_idx = rot_idx + values["idx"]
rot_idx = sorted(rot_idx)
pat_df_manual_feet.iloc[rot_idx, pat_df_manual_feet.columns.get_loc("pixel_action_filepath")] = \
    rotated_all_dir + os.sep + pat_df_manual_feet.iloc[rot_idx, ]["filename_new_dupl"] + ".dcm"

pat_df_manual_feet.iloc[anonym_prob_idx, pat_df_manual_feet.columns.get_loc("pixel_action_filepath")] = \
    anonym_prob_all_dir + os.sep + pat_df_manual_feet.iloc[anonym_prob_idx, ]["filename_new_dupl"] + ".dcm"

# save the new pat_df
pat_df_manual_nofeet = pat_df_manual_incl_addon[[i not in pat_df_manual_feet['filename_new_dupl'].to_list()
                                                 for i in pat_df_manual_incl_addon['filename_new_dupl']]]
pat_df_manual_save = pd.concat([pat_df_manual_nofeet, pat_df_manual_feet])
current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
new_pat_df_filename = output_folder + os.sep + os.path.splitext(pat_df_manual_version)[0] + "_" + \
                      os.path.splitext(os.path.basename(pat_df_manual_addon_path))[0] + \
                      "_Step7b_" + current_datetime + ".csv"
pat_df_manual_save.to_csv(new_pat_df_filename)


# ------ splitpoint_manual splitting ------ #

# TODO: CONTINUE HERE: LOAD THE PAT_DF that includes the new categorization (splitpoint!) for rotated images
#  (and where no addon df nötig, weil schon inkludiert)
dicom_server = "/project/autoscora/autoscoRA_images/pixel_action_finished_dicoms_FEET"
output_folder = "/Volumes/Research/Diplomarbeiten/Diplomand_innen_Thomas_Deimel/Kanta_Chrysa/spreadsheets"
pat_df_manual_version = "pat_df_rot/pat_df_manual_2020-11-06_17-43-03_rotsplitpoint.csv"
pat_df_manual_addon = "pat_df_feet_check/pat_df_manual_2021-04-20_16-37-27.csv"

# read in pat_df_manual
pat_df_manual_path = output_folder + os.sep + pat_df_manual_version
pat_df_manual_addon_path = output_folder + os.sep + pat_df_manual_addon

# pat_df_manual_path = filedialog.askopenfilename()
pat_df_manual = pd.read_csv(pat_df_manual_path)
pat_df_manual_addon = pd.read_csv(pat_df_manual_addon_path)

# create backup copy of pat_df_manual
pat_df_manual_original = pat_df_manual.copy()
pat_df_manual_addon_original = pat_df_manual_addon.copy()

# joint the dataframes
pat_df_manual_minus_addon = pat_df_manual[[i not in pat_df_manual_addon['filename_new_dupl'].to_list()
                                           for i in pat_df_manual['filename_new_dupl']]]
pat_df_manual_incl_addon = pd.concat([pat_df_manual_minus_addon, pat_df_manual_addon], sort=False)
pat_df_manual = pat_df_manual_incl_addon
pat_df_manual_feet = pat_df_manual[pat_df_manual["bodypart_manual"] == "F"]

# manual columns
manual_columns_wo_comment = ['bodypart_manual', 'laterality_manual', 'view_position_manual',
                             'inverted_manual', 'rotated_manual', 'black_manual', 'operated_manual',
                             'inappropriate_manual', 'category_manual', 'filename_manual',
                             'splitpoint_manual']


if not os.path.isdir(dicom_server):
    os.mkdir(dicom_server)
else:
    print("dir already exists: " + dicom_server)
# indices
splitpoint_idx = [i for i, x in enumerate(pat_df_manual_feet['filename_new_dupl'])
                  if pat_df_manual_feet.iloc[i, ]['splitpoint_manual'] not in ["none", "skipped", "TBD"]
                  and pat_df_manual_feet.iloc[i, ]['bodypart_manual'] == "F"
                  and pat_df_manual_feet.iloc[i, ]['view_position_manual'] != "lat"
                  and pat_df_manual_feet.iloc[i, ]['inappropriate_manual'] != "inapp"
                  and not bool(re.search("four_limbs", pat_df_manual_feet.iloc[i, ]['comment']))]

# put splitpoints, indices, img in a dict
splitpoint_dict = {idx: {"img": pat_df_manual_feet.iloc[idx, ]['sub_sub_dir'] +
                                os.sep +
                                pat_df_manual_feet.iloc[idx, ]['filename_new_dupl'] + ".dcm",
                         "split_x": pat_df_manual_feet.iloc[idx, ]['splitpoint_manual']}
                   for idx in splitpoint_idx}

# directory
splitpoint_fixed_dir = dicom_server + os.sep + "splitpoint_incl_rotsplit_fixed_dicoms"
if not os.path.isdir(splitpoint_fixed_dir):
    os.mkdir(splitpoint_fixed_dir)
else:
    print("dir already exists: " + splitpoint_fixed_dir)

# split them and save them in splitpoint_fixed_dir
for idx, values in splitpoint_dict.items():
    print(str(idx) + ": " + os.path.basename(values["img"]) + " -- " + values["img"])
    # instantiate original image
    xray = XRay(values["img"])
    split_images = xray.img_split(split_x=int(values["split_x"]), plot=False)
    for panel, panel_xray in split_images.items():
        filename_panel = splitpoint_fixed_dir + os.sep + \
                         os.path.splitext(os.path.basename(xray.filepath))[0] + "_" + \
                         panel + ".dcm"
        print(filename_panel)
        if not os.path.isfile(filename_panel):
            panel_xray.img_write(filename=filename_panel,
                                 write_like_original=True)
        else:
            print("file already exists: " + filename_panel)
        # create 2 new pat_df_manual_feet entries for the split images (leave old B entries in for now)
        new_row = pat_df_manual_feet.iloc[idx, :].copy()
        # # changed entries: sub_sub_dir, filename_new_dupl, all manual entries to TBD
        new_row["sub_sub_dir"] = splitpoint_fixed_dir
        new_row["filename_new_dupl"] = new_row["filename_new_dupl"] + "_" + panel
        new_row[manual_columns_wo_comment] = 'TBD'
        # # add comment to the new entries: "from_split"
        old_panel_comment = pat_df_manual_feet.iloc[idx, ]["comment"]
        if "from_split" in old_panel_comment:
            new_panel_comment = old_panel_comment
        elif old_panel_comment in ["none", "TBD"]:
            new_panel_comment = "from_split"
        else:
            new_panel_comment = old_panel_comment + "__from_split"
        new_row['comment'] = new_panel_comment
        new_row['comment_status'] = "ComYes"
        # print(new_row)
        # append new_row to pat_df_manual_feet
        new_row = new_row.rename(index=np.shape(pat_df_manual_feet)[0])
        pat_df_manual_feet = pat_df_manual_feet.append(new_row, ignore_index=True)
        # print(pat_df_manual_feet.iloc[np.shape(pat_df_manual_feet)[0]-1, :])
pat_df_manual_feet = pat_df_manual_feet.astype(pat_df_manual_addon.dtypes)

# set pat_df manual comment to "split_done" for original image
old_original_comment = pat_df_manual_feet.iloc[splitpoint_idx, ]["comment"]
new_original_comment = [x if "split_done" in x
                        else "split_done" if x in ["none", "TBD"] else x + "__split_done"
                        for x in old_original_comment]
pat_df_manual_feet.iloc[splitpoint_idx, pat_df_manual_feet.columns.get_loc("comment")] = new_original_comment
pat_df_manual_feet.iloc[splitpoint_idx, pat_df_manual_feet.columns.get_loc("comment_status")] = "ComYes"


# ------ 4 limbs splitting ------ # copied from above and changed, check that changed everything to four_limbs
# indices
four_limbs_idx = [i for i, x in enumerate(pat_df_manual_feet['filename_new_dupl'])
                  if pat_df_manual_feet.iloc[i, ]['splitpoint_manual'] not in ["none", "skipped", "TBD"]
                  and pat_df_manual_feet.iloc[i, ]['bodypart_manual'] == "F"
                  and pat_df_manual_feet.iloc[i, ]['view_position_manual'] != "lat"
                  and pat_df_manual_feet.iloc[i, ]['inappropriate_manual'] != "inapp"
                  and bool(re.search("four_limbs", pat_df_manual_feet.iloc[i, ]['comment']))]

# put splitpoints, indices, img in a dict - DONE MANUALLY
four_limbs_dict = {idx: {"img": pat_df_manual_feet.iloc[idx, ]['sub_sub_dir'] +
                                os.sep +
                                pat_df_manual_feet.iloc[idx, ]['filename_new_dupl'] + ".dcm",
                         "split_x": 2088,
                         "split_y": 2472}
                   for idx in four_limbs_idx}

# directory
four_limbs_fixed_dir = dicom_server + os.sep + "four_limbs_fixed_dicoms"
if not os.path.isdir(four_limbs_fixed_dir):
    os.mkdir(four_limbs_fixed_dir)
else:
    print("dir already exists: " + four_limbs_fixed_dir)

# split them and save them in four_limbs_fixed_dir
for idx, values in four_limbs_dict.items():
    print(str(idx) + ": " + os.path.basename(values["img"]) + " -- " + values["img"])
    xray = XRay(values["img"])
    split_images = xray.img_split(split_x=int(values["split_x"]), split_y=int(values["split_y"]), plot=True)
    for panel, panel_xray in split_images.items():
        filename_panel = four_limbs_fixed_dir + os.sep + \
                         os.path.splitext(os.path.basename(xray.filepath))[0] + "_" + \
                         panel + ".dcm"
        print(filename_panel)
        if not os.path.isfile(filename_panel):
            panel_xray.img_write(filename=filename_panel,
                                 write_like_original=True)
        else:
            print("file already exists: " + filename_panel)
        # create 2 new pat_df_manual entries for the split images (leave old B entries in for now)
        new_row = pat_df_manual_feet.iloc[idx, :].copy()
        # # changed entries: sub_sub_dir, filename_new_dupl, all manual entries to TBD
        new_row["sub_sub_dir"] = four_limbs_fixed_dir
        new_row["filename_new_dupl"] = new_row["filename_new_dupl"] + "_" + panel
        new_row[manual_columns_wo_comment] = 'TBD'
        # # add comment to the new entries: "from_split"
        old_panel_comment = pat_df_manual_feet.iloc[idx, ]["comment"]
        if "from_split" in old_panel_comment:
            new_panel_comment = old_panel_comment
        elif old_panel_comment in ["none", "TBD"]:
            new_panel_comment = "from_split"
        else:
            new_panel_comment = old_panel_comment + "__from_split"
        new_row['comment'] = new_panel_comment
        new_row['comment_status'] = "ComYes"
        # print(new_row)
        # append new_row to pat_df_manual_feet
        new_row = new_row.rename(index=np.shape(pat_df_manual_feet)[0])
        pat_df_manual_feet = pat_df_manual_feet.append(new_row, ignore_index=True)
        print(pat_df_manual_feet.iloc[np.shape(pat_df_manual_feet)[0] - 1, :])
pat_df_manual_feet = pat_df_manual_feet.astype(pat_df_manual_addon.dtypes)

# set pat_df manual comment to "split_done" for original image
old_original_comment = pat_df_manual_feet.iloc[four_limbs_idx, ]["comment"]
new_original_comment = [x if "split_done" in x
                        else "split_done" if x in ["none", "TBD"] else x + "__split_done"
                        for x in old_original_comment]
pat_df_manual_feet.iloc[four_limbs_idx, pat_df_manual_feet.columns.get_loc("comment")] = new_original_comment
pat_df_manual_feet.iloc[four_limbs_idx, pat_df_manual_feet.columns.get_loc("comment_status")] = "ComYes"


# ------ common final path ------ #

# in pat_df manual, add a column containing the new path of the changed images
# # copy old path for all images
pat_df_manual_feet["pixel_action_filepath"] = pat_df_manual_feet["sub_sub_dir"] + os.sep + \
                                              pat_df_manual_feet["filename_new_dupl"] + ".dcm"

# save the new pat_df
pat_df_manual_nofeet = pat_df_manual[[i not in pat_df_manual_feet['filename_new_dupl'].to_list()
                                      for i in pat_df_manual['filename_new_dupl']]]
pat_df_manual_save = pd.concat([pat_df_manual_nofeet, pat_df_manual_feet])
current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
new_pat_df_filename = output_folder + os.sep + os.path.splitext(pat_df_manual_version)[0] + "_" + \
                      os.path.splitext(os.path.basename(pat_df_manual_addon_path))[0] + \
                      "_Step7b_" + current_datetime + ".csv"
pat_df_manual_save.to_csv(new_pat_df_filename)

# go through splitpoint images and categorize with Fiji - DONE

# TODO: CAVE: some of the rotated images also need splitting (or were split before rotation in the split part)
# just copy the split rot images to the split folder and add according lines for the split images (if not done already) to the df