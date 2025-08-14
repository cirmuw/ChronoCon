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
# import shutil
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

dicom_server = "/project/autoscora/autoscoRA_images/H_images_of_interest_2_renamed_mirrored_dicoms"
# old_server = "/project/autoscora/autoscoRA_images/H_images_of_interest_1_dicoms"
output_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/output"
# output_folder = "/mnthome2/autoscoRA/autoscoRA_Preprocessing/output"
pat_df_manual_version = \
    "pat_df_manual_2019-04-14_20-46-23_img_of_interest2_2019-04-15_12-53-41_NameChange_2019-04-16_08-29-46.csv"

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


########################################
# take all right hands + mirror + save #
########################################

# filepaths
filepaths = [dicom_server + os.sep + i for i in os.listdir(dicom_server) if not i.startswith(".")]

# select R and mirror and save
R_hands = []
for file_i in filepaths:
    xray = XRay(file_i)
    if xray.img.Laterality == "R":
        R_hands = R_hands + [file_i]
        xray.img_flip(flipped_axis=1)
        xray.img_write(filename=file_i, write_like_original=True)
    else:
        pass
print("No. of R hands" + str(len(R_hands)))

print("end of script")
