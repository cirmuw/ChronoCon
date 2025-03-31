import input.constants.input_constants as const
import os
import sys
import numpy as np
import pandas as pd
import re
import warnings
import SimpleITK as sitk
import pydicom
from copy import copy, deepcopy
import matplotlib
matplotlib.use(const.MATPLOTLIB_BACKEND)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D


class XRay:
    def __init__(self, filepath, donotload=False, datatype='dcm'):
        from autoscora_utils.logging_class import Logger
        self.logger = Logger.sub_logger('evaluation_class.' + type(self).__name__)
        self.filepath = filepath
        self.name = os.path.splitext(os.path.basename(self.filepath))[0]
        self.img_id = self.img_id_from_name()
        self.error = []
        self.datatype = datatype
        if donotload:
            self.img = 'not_loaded'
        else:
            self.img = self.read_xray()
            # self.reload_xray()

    def img_id_from_name(self):
        img_id_return = re.sub("_(MOne|MTwo)_.*$", "", self.name)
        return img_id_return

    def reload_xray(self, reduce_sitk_dims=True):
        """sets self.img to a pydicom image"""
        if self.datatype == 'dcm':
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
        elif self.datatype == 'nii':
            pass
        else:
            raise ValueError("self.datatype should be 'dcm' or 'nii', not: " + str(self.datatype))

    def read_xray(self, reduce_sitk_dims=True):
        """returns a pydicom image"""
        if self.datatype == 'dcm':
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
        elif self.datatype == 'nii':
            pass
        else:
            raise ValueError("self.datatype should be 'dcm' or 'nii', not: " + str(self.datatype))

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
        self.logger.info(perc_print, "-th percentile pixel value: ", percentile_value)
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
                self.logger.info(sys.exc_info())
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

    def get_px_to_mm(self, warn_if_tag_not_available=True, nan_type=None):
        if (0x0028, 0x0030) in list(self.img.keys()):
            ps = self.img[(0x0028, 0x0030)].value
        elif (0x0018, 0x1164) in list(self.img.keys()):
            ps = self.img[(0x0018, 0x1164)].value
        elif (0x0018, 0x2010) in list(self.img.keys()):
            ps = self.img[(0x0018, 0x1164)].value
        elif (0x3002, 0x0011) in list(self.img.keys()):
            ps = self.img[(0x0018, 0x1164)].value
        elif (0x300A, 0x00E9) in list(self.img.keys()):
            ps = self.img[(0x0018, 0x1164)].value
        elif (0x0018, 0x7022) in list(self.img.keys()):
            ps = self.img[(0x0018, 0x1164)].value
        elif (0x0070, 0x0101) in list(self.img.keys()):
            ps = self.img[(0x0018, 0x1164)].value
        elif (0x2010, 0x0376) in list(self.img.keys()):
            ps = self.img[(0x0018, 0x1164)].value
        elif (0x0018, 0x9404) in list(self.img.keys()):
            ps = self.img[(0x0018, 0x1164)].value
        else:
            if warn_if_tag_not_available:
                warnings.warn("no PixelSpacing-like tag found for " + self.name + ". Returning None")
            if nan_type == 'pandas':
                return pd.NA
            elif nan_type == 'numpy':
                return np.nan
            else:
                return None

        ps = [float(i) for i in ps]
        assert len(list(set(ps))) == 1  # check that pixel spacing is identical for rows and columns

        return ps[0]

    def get_institution(self, warn_if_tag_not_available=True, nan_type='pandas'):
        if (0x0008, 0x0070) in list(self.img.keys()):
            ps = self.img[(0x0008, 0x0070)].value
            """
            elif (0x0008, 0x0080) in list(xray.img.keys()):
                ps = xray.img[(0x0008, 0x0080)].value
            elif (0x0018, 0x2010) in list(xray.img.keys()):
                ps = xray.img[(0x0018, 0x1164)].value
            elif (0x3002, 0x0011) in list(xray.img.keys()):
                ps = xray.img[(0x0018, 0x1164)].value
            elif (0x300A, 0x00E9) in list(xray.img.keys()):
                ps = xray.img[(0x0018, 0x1164)].value
            elif (0x0018, 0x7022) in list(xray.img.keys()):
                ps = xray.img[(0x0018, 0x1164)].value
            elif (0x0070, 0x0101) in list(xray.img.keys()):
                ps = xray.img[(0x0018, 0x1164)].value
            elif (0x2010, 0x0376) in list(xray.img.keys()):
                ps = xray.img[(0x0018, 0x1164)].value
            elif (0x0018, 0x9404) in list(xray.img.keys()):
                ps = xray.img[(0x0018, 0x1164)].value
            """
        else:
            if warn_if_tag_not_available:
                warnings.warn("no Institution-like tag found for " + self.name + ". Returning NaN")
            if nan_type == 'pandas':
                return pd.NA
            elif nan_type == 'numpy':
                return np.nan
            else:
                return None
        return ps

#
