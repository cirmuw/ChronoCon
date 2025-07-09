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

dicom_server = "/project/autoscora/autoscoRA_images/H_images_of_interest_1_dicoms"
duplicate_server = "/project/autoscora/autoscoRA_images/H_dp_duplicate_1_dicoms"
# output_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/output"
output_folder = "/mnthome2/autoscoRA/autoscoRA_Preprocessing/output"
pat_df_manual_version = "pat_df_manual_2019-04-14_20-46-23.csv"

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

# true duplicates (according to manual check)
true_duplicate_1_names = ["13_20050126_H_NaN_dp_MTwo_1",
                          "13_20050126_H_NaN_dp_MTwo_3",
                          "44_20000810_H_NaN_NaN_MOne_1",
                          "44_20000810_H_NaN_NaN_MOne_2",
                          "44_20040618_H_NaN_NaN_MOne_1",
                          "44_20040618_H_NaN_NaN_MOne_2",
                          "50_20040720_H_NaN_NaN_MOne_1",
                          "50_20040720_H_NaN_NaN_MOne_2",
                          "50_20070711_H_R_dp_MTwo_1",
                          "50_20080703_H_R_dp_MOne_1",
                          "50_20110314_H_R_dp_MTwo_1",
                          "69_20120314_H_L_dp_MTwo_1",
                          "90_20080910_H_R_dp_MOne_1",
                          "102_20040514_H_NaN_dp_MTwo_1",
                          "102_20040514_H_NaN_dp_MTwo_2",
                          "104_20030825_H_NaN_NaN_MOne_1",
                          "104_20030825_H_NaN_NaN_MOne_2",
                          "104_20061114_H_L_dp_MTwo_0",
                          "104_20061114_H_R_dp_MTwo_0",
                          "111_20020424_H_NaN_NaN_MOne_1",
                          "111_20020424_H_NaN_NaN_MOne_2",
                          "117_20071123_H_B_dp_MOne_2",
                          "155_20090518_H_NaN_NaN_MOne_1_left",
                          "155_20090518_H_NaN_NaN_MOne_1_right",
                          "158_20070917_H_L_dp_MOne_1",
                          "158_20070917_H_R_dp_MOne_1",
                          "170_20071005_H_R_dp_MOne_1",
                          "170_20090122_H_R_dp_MOne_1",
                          "172_20020305_H_NaN_NaN_MOne_1",
                          "172_20020305_H_NaN_NaN_MOne_2",
                          "179_20070228_H_L_dp_MOne_1",
                          "181_20020910_H_NaN_NaN_MOne_1",
                          "181_20020910_H_NaN_NaN_MOne_2",
                          "214_20050810_H_NaN_NaN_MOne_2",
                          "214_20050810_H_NaN_NaN_MOne_4",
                          "219_20100723_H_L_dp_MTwo_2",
                          "219_20100723_H_R_dp_MTwo_2",
                          "224_20110523_H_L_dp_MTwo_0",
                          "224_20110523_H_R_dp_MTwo_1",
                          "238_20090812_H_L_dp_MTwo_0",
                          "238_20090812_H_R_dp_MTwo_0",
                          "255_20110808_H_R_dp_MTwo_0",
                          "256_20060921_H_L_dp_MOne_1",
                          "275_20030403_H_NaN_dp_MTwo_1",
                          "275_20030403_H_NaN_dp_MTwo_4",
                          "312_20130218_H_R_dp_MTwo_0",
                          "312_20130218_H_R_lat_MTwo_1",
                          "329_20111128_H_L_dp_MTwo_1",
                          "329_20111128_H_R_dp_MTwo_1",
                          "332_20140725_H_L_dp_MTwo_1",
                          "332_20140725_H_L_dp_MTwo_2",
                          "340_20140723_H_R_dp_MTwo_1",
                          "340_20140723_H_R_dp_MTwo_2",
                          "361_20050804_H_NaN_NaN_MOne_3",
                          "361_20050804_H_NaN_NaN_MOne_7",
                          "386_20030227_H_NaN_dp_MTwo_1",
                          "386_20030227_H_NaN_dp_MTwo_3",
                          "388_20071128_H_R_dp_MOne_1",
                          "401_20080403_H_R_dp_MOne_2",
                          "411_20060706_H_R_dp_MOne_1",
                          "412_20080627_H_R_dp_MOne_1",
                          "418_20050315_H_NaN_dp_MTwo_1",
                          "418_20050315_H_NaN_dp_MTwo_3",
                          "418_20060515_H_L_dp_MOne_1",
                          "434_20050318_H_NaN_dp_MTwo_1",
                          "434_20050318_H_NaN_dp_MTwo_3",
                          "440_20040712_H_NaN_NaN_MOne_1",
                          "440_20040712_H_NaN_NaN_MOne_2",
                          "450_20100531_H_L_dp_MOne_0",
                          "450_20100531_H_R_dp_MOne_0",
                          "455_20090312_H_R_dp_MOne_1",
                          "489_20050907_H_L_dp_MOne_2",
                          "489_20090922_H_L_dp_MTwo_2",
                          "489_20110301_H_L_dp_MTwo_2",
                          "489_20131223_H_L_dp_MTwo_1",
                          "495_20051014_H_L_dp_MOne_2",
                          "495_20051014_H_R_dp_MOne_2",
                          "496_20091019_H_L_dp_MTwo_1",
                          "498_20051012_H_L_dp_MOne_1",
                          "536_20100809_H_L_dp_MTwo_2",
                          "537_20160513_H_R_dp_MTwo_1",
                          "537_20160513_H_R_dp_MTwo_2",
                          "539_20080508_H_R_dp_MOne_1",
                          "543_20070507_H_R_dp_MOne_2",
                          "572_20080604_H_L_dp_MOne_2",
                          "583_20050413_H_NaN_NaN_MOne_3",
                          "603_20030429_H_R_dp_MOne_2",
                          "638_20080910_H_R_dp_MOne_1",
                          "640_20030409_H_NaN_dp_MTwo_1",
                          "640_20030409_H_NaN_dp_MTwo_5",
                          "645_20030226_H_NaN_dp_MTwo_1",
                          "645_20030226_H_NaN_dp_MTwo_3",
                          "654_20070208_H_NaN_NaN_MOne_1_left",
                          "655_20060524_H_R_dp_MOne_2",
                          "660_20070524_H_R_dp_MOne_1",
                          "660_20091117_H_L_dp_MTwo_1",
                          "660_20091117_H_L_dp_MTwo_2",
                          "663_20100310_H_L_dp_MTwo_1",
                          "663_20100310_H_L_dp_MTwo_2",
                          "682_20121018_H_R_dp_MTwo_0",
                          "684_20161121_H_L_dp_MTwo_1",
                          "703_20080909_H_L_dp_MOne_1",
                          "703_20080909_H_R_dp_MOne_1",
                          "719_20170201_H_L_dp_MTwo_1",
                          "728_20050905_H_L_dp_MOne_1",
                          "752_20111102_H_L_dp_MTwo_1",
                          "752_20111102_H_R_dp_MTwo_1",
                          "434_20050318_H_NaN_dp_MTwo_1",
                          "772_20040604_H_NaN_dp_MTwo_1",
                          "772_20040604_H_NaN_dp_MTwo_5",
                          "797_20091123_H_L_dp_MTwo_1",
                          "797_20091123_H_R_dp_MTwo_1",
                          ]

pat_df_manual.loc[pat_df_manual["filename_new_dupl"] == "537_20160513_H_R_lat_MTwo_1", "view_position_manual"] = "lat"


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

    def pixel_flip(self, flipped_axis=1):
        """returns the pixel array with entries in flipped_axis flipped"""
        if self.img == "not_loaded":
            raise Exception("Please load xray first using 'yourimage.read_xray()'.")
        else:
            flip_pixel = np.flip(m=self.img.pixel_array, axis=flipped_axis)
            return flip_pixel

    def img_flip(self, flipped_axis=1):
        """rotates pixel array of self.img"""
        flip_pixel = self.pixel_flip(flipped_axis=flipped_axis)
        # self.img.Rows = np.shape(rot_pixel)[0]
        # self.img.Columns = np.shape(rot_pixel)[1]
        self.img.PixelData = flip_pixel.tobytes()

    def img_write(self, filename, write_like_original):
        self.img.save_as(filename=str(filename), write_like_original=write_like_original)


########################
# fix duplicate issues #
########################

flipped_images = ["104_20030825_H_NaN_NaN_MOne_1",
                  "44_20040618_H_NaN_NaN_MOne_2",
                  "111_20020424_H_NaN_NaN_MOne_1",
                  "172_20020305_H_NaN_NaN_MOne_1",
                  "181_20020910_H_NaN_NaN_MOne_2"]

flipped_images_path = [dicom_server + os.sep + i + ".dcm" for i in flipped_images]

for img in flipped_images_path:
    xray = XRay(img)
    xray.img_flip(flipped_axis=1)
    save_name = dicom_server + os.sep + os.path.basename(xray.filepath)
    # save_name = "/Users/Tommi/Desktop/duplicates remaining" + os.sep + os.path.basename(xray.filepath)
    if os.path.isfile(save_name):
        xray_idx = [i for i, x in enumerate(pat_df_manual["filename_new_dupl"])
                    if x == os.path.splitext(os.path.basename(xray.filepath))[0]]
        if len(xray_idx) > 1:
            print("more than one xray_idx for " + os.path.basename(xray.filepath))
        elif len(xray_idx) == 0:
            print("no xray_idx found for " + os.path.basename(xray.filepath))
        else:
            if "".join(list(pat_df_manual.loc[xray_idx, "laterality_manual"])) == "L":
                print("set to R")
                pat_df_manual.loc[xray_idx, "laterality_manual"] = "R"
                # xray.img_write(filename=save_name, write_like_original=True)
            elif "".join(list(pat_df_manual.loc[xray_idx, "laterality_manual"])) == "R":
                print("set to L")
                pat_df_manual.loc[xray_idx, "laterality_manual"] = "L"
                # xray.img_write(filename=save_name, write_like_original=True)
            else:
                print("original laterality_manual neither 'R', nor 'L'")
    else:
        print("no file of name " + os.path.basename(xray.filepath) + " in dir " + dicom_server)

# check that worked:
# for img in flipped_images_path:
#     xray = XRay(img)
#     xray_idx = [i for i, x in enumerate(pat_df_manual["filename_new_dupl"])
#                 if x == os.path.splitext(os.path.basename(xray.filepath))[0]]
#     print("".join(list(pat_df_manual.loc[xray_idx, "laterality_manual"])))
#     xray.plot_img()


#########################
# set "img_of_interest" #
#########################

# check that all images in "images_of_interest_1" dir have img_of_interest==1
images_of_interest_1_files = os.listdir(dicom_server)
images_of_interest_1_names = [re.sub(".dcm", "", i) for i in images_of_interest_1_files]

all(pat_df_manual.loc[[x in images_of_interest_1_names for x in pat_df_manual["filename_new_dupl"]],
                      "img_of_interest"] == 1)

# check that no other images have img_of_interest==1
all(pat_df_manual.loc[[x not in images_of_interest_1_names for x in pat_df_manual["filename_new_dupl"]],
                      "img_of_interest"] != 1)

# set img_of_interest = 0 for the duplicates you want to exclude
duplicate_1_files = os.listdir(duplicate_server)
duplicate_1_names = [re.sub(".dcm", "", i) for i in duplicate_1_files]
wrong_duplicate_1_idx = [i for i, x in enumerate(pat_df_manual["filename_new_dupl"])
                         if x in duplicate_1_names
                         and x not in true_duplicate_1_names]
pat_df_manual.loc[wrong_duplicate_1_idx, "img_of_interest"] = 0

# set img_of_interest = 2 for the appriopriate images
interest_2_images_idx = [i for i, x in enumerate(pat_df_manual["img_of_interest"])
                         if x == 1
                         and pat_df_manual.loc[i, 'bodypart_manual'] == "H"
                         and pat_df_manual.loc[i, 'laterality_manual'] in ["R", "L"]
                         and pat_df_manual.loc[i, 'view_position_manual'] == "dp"
                         and pat_df_manual.loc[i, 'black_manual'] == "BOk"
                         and pat_df_manual.loc[i, 'inappropriate_manual'] == "app"
                         and not bool(re.search("split_done", pat_df_manual.loc[i, 'comment']))]

pat_df_manual.loc[interest_2_images_idx, "img_of_interest"] = 2

# check that img_of_interest == 1 does not include wrong images
any(pat_df_manual.loc[pat_df_manual["view_position_manual"] == "lat", "img_of_interest"] == 2)

# check that no img_of_interest = 1 for inapp images
any(pat_df_manual.loc[pat_df_manual["inappropriate_manual"] != "app", "img_of_interest"] == 2)


#######################################################
# check that no more duplicates or missing candidates #
#######################################################

pat_df_manual_2 = pat_df_manual.copy()
pat_df_manual_2['body_laterality_view_manual'] = pat_df_manual_2['bodypart_manual'].astype(str) + '_' \
                                                 + pat_df_manual_2['laterality_manual'] + '_' \
                                                 + pat_df_manual_2['view_position_manual']

pat_df_manual_2['id_date'] = pat_df_manual_2['pat_id'].astype(str) + '_' + pat_df_manual_2['study_date'].astype(str)
unique_id_dates = pat_df_manual_2.id_date.unique()

# only look at those with img_of_interest == 2
pat_df_manual_2 = pat_df_manual_2.loc[pat_df_manual_2["img_of_interest"] == 2, :]


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

# exclude inapp
excluded_inapp = []
duplicates_df_app = pd.DataFrame(columns=['filename_new_dupl', 'sub_sub_dir',
                                          'pixel_action_filepath',
                                          'filename_manual'])
for i in duplicates_df.iterrows():
    if pat_df_manual.loc[i[0], "inappropriate_manual"] != "app":
        excluded_inapp = excluded_inapp + [i[0]]
    else:
        if pat_df_manual.loc[i[0], "img_of_interest"] != 2:
            excluded_inapp = excluded_inapp + [i[0]]
        else:
            duplicates_df_app = duplicates_df_app.append(duplicates_df.loc[i[0], :])
            print(pat_df_manual.loc[i[0], "img_of_interest"])

# check that no candidates or duplicates
print(len(candidates_df) == 0)
print(len(duplicates_df) == 0)
print(len(duplicates_df_app) == 0)

####################################
# move img_of_interest to a folder #
####################################

# find all images with img_of_interest == 2
img_of_interest_2_idx = [i for i, x in enumerate(pat_df_manual["img_of_interest"])
                         if x == 2]

# add a column with the new filepath
img_of_interest_2_dir = re.sub(os.path.split(dicom_server)[1] + "$", "H_images_of_interest_2_dicoms", dicom_server)

pat_df_manual["img_of_interest_2_filepath"] = "none"
pat_df_manual.loc[img_of_interest_2_idx,
                  "img_of_interest_2_filepath"] = img_of_interest_2_dir + os.sep + \
                                                  pat_df_manual.loc[img_of_interest_2_idx, "filename_new_dupl"] + \
                                                  ".dcm"

# save pat_df_manual as "img_of_interest_2"
current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
new_pat_df_filename = output_folder + os.sep + os.path.splitext(pat_df_manual_version)[0] + \
                      "_img_of_interest2_" + current_datetime + ".csv"
pat_df_manual.to_csv(new_pat_df_filename)

# check that they're all in img_of_interst_1 dir
all(i + ".dcm" in images_of_interest_1_files for i in
    pat_df_manual.loc[img_of_interest_2_idx, "filename_new_dupl"])

# create dir
img_of_interest_2_dir = re.sub(os.path.split(dicom_server)[1] + "$", "H_images_of_interest_2_dicoms", dicom_server)
if not os.path.isdir(img_of_interest_2_dir):
    os.mkdir(img_of_interest_2_dir)
else:
    print("dir already exists: " + img_of_interest_2_dir)

# copy images to new folder
old_paths = dicom_server + os.sep + pat_df_manual.loc[img_of_interest_2_idx, "filename_new_dupl"] + ".dcm"
for file_of_interest in old_paths:
    if os.path.exists(file_of_interest):
        new_path = img_of_interest_2_dir + os.sep + os.path.basename(file_of_interest)
        if not os.path.exists(new_path):
            shutil.copy(file_of_interest, new_path)
            print("copied")
    else:
        print("not a file: " + file_of_interest)
