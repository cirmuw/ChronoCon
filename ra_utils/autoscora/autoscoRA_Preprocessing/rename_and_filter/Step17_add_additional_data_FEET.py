import os
import sys
import numpy as np
import pandas as pd
from collections import OrderedDict
import re
from distutils.dir_util import copy_tree
import shutil
from copy import copy, deepcopy
import dicom
import pydicom
import SimpleITK as sitk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from datetime import datetime

RUN_LOCALLY = True

if RUN_LOCALLY:
    HOME_DIR_SEP = "/mnthome2" + os.sep
    PROJECT_DIR_SEP = "/project" + os.sep
    MATPLOTLIB_BACKEND = 'TkAgg'
else:
    HOME_DIR_SEP = "/home/cir/tdeimel" + os.sep
    PROJECT_DIR_SEP = "/project" + os.sep
    MATPLOTLIB_BACKEND = 'agg'

dicom_server = "/project/autoscora/autoscoRA_images/F_images_of_interest_2_renamed_mirrored_inverted_dicoms"
# TODO: THIS SCRIPT HAS NOT BEEN RUN AS I DID NOT GET FEET DATA FROM RUHRGEBIET!!!!

PREPROCESSING_OUTPUT_DIR = HOME_DIR_SEP + "autoscoRA/autoscoRA_Preprocessing/output/pat_df_manual_FEET"
PAT_DF_MANUAL_VERSION = \
    "data_split" + os.sep + \
    "pat_df_medstream_manual_F_dp_img_of_int_2_summary_cols_segm_sets_RL_" \
    "stratmean_split6543_chosen345_2021-05-25_13-45-01.csv"
PAT_DF_MANUAL_PATH = PREPROCESSING_OUTPUT_DIR + os.sep + PAT_DF_MANUAL_VERSION

TEMPLATE_DF_MANUAL_PATH = PREPROCESSING_OUTPUT_DIR + os.sep + 'additional_data' + os.sep + \
    'ruhrgebiet_scores.csv'

NEW_DF_MANUAL_PATH = PREPROCESSING_OUTPUT_DIR + os.sep + 'additional_data' + os.sep + \
    'Ruhrgebiet_hands_df.csv'

NEW_IMAGES_PATH = '/Users/Tommi/Documents/STUDIUM/Mac-Medizin/Rheumatologie/' \
                  'Rheumatologie AKH Med III Sommer 2017/Data/Ruhrgebiet/Ruhrgebiet_Data'

PSEUD_IMAGES_OUTPUT = '/Users/Tommi/Documents/STUDIUM/Mac-Medizin/Rheumatologie/Rheumatologie AKH Med III Sommer 2017/' \
                      'Data/Ruhrgebiet'
PSEUD_IMAGES_PATH = '/Users/Tommi/Documents/STUDIUM/Mac-Medizin/Rheumatologie/Rheumatologie AKH Med III Sommer 2017/' \
                    'Data/Ruhrgebiet/Ruhrgebiet_Data_pseudonym'


# functions, classes


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


def dicom_dataset_to_dict(dicom_header):
    dicom_dict = {}
    repr(dicom_header)
    for dicom_value in dicom_header.values():
        if dicom_value.tag == (0x7fe0, 0x0010):
            # discard pixel data
            continue
        if type(dicom_value.value) == pydicom.dataset.Dataset:
            dicom_dict[dicom_value.tag] = dicom_dataset_to_dict(dicom_value.value)
        else:
            v = _convert_value(dicom_value.value)
            dicom_dict[dicom_value.tag] = v
    return dicom_dict


def _sanitise_unicode(s):
    return s.replace(u"\u0000", "").strip()


def _convert_value(v):
    t = type(v)
    if t in (list, int, float):
        cv = v
    elif t == str:
        cv = _sanitise_unicode(v)
    elif t == bytes:
        s = v.decode('ascii', 'replace')
        cv = _sanitise_unicode(s)
    elif t == dicom.valuerep.DSfloat:
        cv = float(v)
    elif t == dicom.valuerep.IS:
        cv = int(v)
    elif t == dicom.valuerep.PersonName3:
        cv = str(v)
    else:
        cv = repr(v)
    return cv


def rec_apply(func, n):
    if n > 1:
        rec_func = rec_apply(func, n - 1)
        return lambda x: func(rec_func(x))
    return func


# read in template_df
# template_df = pd.read_csv(TEMPLATE_DF_MANUAL_PATH, header=0, sep=';')
# pat_df = pd.read_csv(PAT_DF_MANUAL_PATH)
# empty_odict = OrderedDict.fromkeys(pat_df.keys())

# save pseudonymization list and rename folder names
name_list = [f for f in os.listdir(NEW_IMAGES_PATH) if not f.startswith('.') and not f.startswith('~$')]
name_list.sort()
last_name = [re.split(',', i)[0].strip() for i in name_list]
first_name = [re.split(',', i)[1].strip() for i in name_list]

pseudo_ids = [1001 + i for i in range(len(name_list))]
pseudo_odict = OrderedDict([('LASTNAME', last_name), ('FIRSTNAME', first_name), ('OLD_NAMES', name_list)])
pseudo_df = pd.DataFrame.from_dict(pseudo_odict).sort_values('LASTNAME').reset_index(drop=True)
pseudo_df['id_nr'] = [str(i) for i in pseudo_ids]
pseudo_df = pseudo_df[['id_nr', 'LASTNAME', 'FIRSTNAME', 'OLD_NAMES']]

# copy and rename image folders"
"""
os.mkdir(PSEUD_IMAGES_PATH)
for row_nr in range(len(pseudo_df)):
    pat_folder = NEW_IMAGES_PATH + os.sep + pseudo_df['OLD_NAMES'][row_nr]
    new_pat_folder = PSEUD_IMAGES_PATH + os.sep + str(pseudo_df['id_nr'][row_nr])
    copy_tree(pat_folder, new_pat_folder)
"""

# read in image paths
img_paths = []
for root, dirs, files in os.walk(PSEUD_IMAGES_PATH):
    for file in files:
        if (not file.startswith('.')) and file.endswith('.dcm'):
            img_paths = img_paths + [root + os.sep + file]

# deletion of lateral images/paths
img_paths_dp = [i for i in img_paths if not i.__contains__('SE000002')]
img_paths_lat = [i for i in img_paths if bool(re.search('SE000002', i))]
for lat_file in img_paths_lat:
    lat_file_folder = os.path.split(lat_file)[0]
    shutil.rmtree(lat_file_folder)

# combine pseudo_df with image_paths and infos from the path name
img_paths_dp_df = pd.DataFrame.from_dict({'img_path': img_paths_dp})

img_paths_dp_df['id_nr'] = [os.path.split(rec_apply(lambda x: os.path.split(x)[0], 3)(i))[1]
                                          for i in img_paths_dp_df['img_path']]
img_paths_dp_df['bodypart'] = ['H' if i.__contains__('Hand') else 'F' if i.__contains__('Fuss') else 'O'
                               for i in img_paths_dp_df['img_path']]
img_paths_dp_df['laterality'] = ['R' if i.__contains__('rechts') else 'L' if i.__contains__('links') else 'O'
                                 for i in img_paths_dp_df['img_path']]
img_paths_dp_df = img_paths_dp_df[['id_nr', 'bodypart', 'laterality', 'img_path']]

# extr_pseudo_df = pd.concat([pseudo_df]*2, ignore_index=True).sort_values('id_nr').reset_index(drop=True)
pseudo_path_df = pd.merge(pseudo_df, img_paths_dp_df, on='id_nr', how='left')

# get dicom tags
all_unique_tags = []
dicom_tags_dict = {}.fromkeys(pseudo_path_df['img_path'])
dicom_names_dict = {}.fromkeys(pseudo_path_df['img_path'])
for row_nr in range(len(pseudo_path_df)):
    path_i = pseudo_path_df['img_path'][row_nr]
    xray_i = XRay(path_i, donotload=False)
    xray_i.img.keys()
    dicom_tags = dicom_dataset_to_dict(xray_i.img)
    all_unique_tags = list(set(all_unique_tags + [i for i in dicom_tags.keys()]))
    # dicom_tags_tuple = [(k, pydicom.datadict.keyword_for_tag(k), v) for k, v in dicom_tags.items()]
    dicom_tags_dict[path_i] = dicom_tags
    dicom_names_dict[path_i] = {pydicom.datadict.keyword_for_tag(k): v for k, v in dicom_tags.items()}
    # tag_df = pd.DataFrame.from_dict(dicom_tags)

all_unique_names = [pydicom.datadict.keyword_for_tag(i) if len(pydicom.datadict.keyword_for_tag(i))>0 else str(i)
                    for i in all_unique_tags]
tag_name_dict = {k: pydicom.datadict.keyword_for_tag(k) for k in all_unique_tags}
tag_name_list = [(str(k), v) for k, v in tag_name_dict.items()]
tag_name_df = pd.DataFrame(tag_name_list, columns=['tag', 'tag_name'])
# tag_name_df.to_csv(PSEUD_IMAGES_OUTPUT + os.sep + 'tag_name.csv')

for name in all_unique_names:
    pseudo_path_df[name] = [v[name] if name in v.keys() else None for k, v in dicom_names_dict.items()]

# id_nr, RO_datum, id_RO_nr, key_id, LASTNAME, FIRSTNAME, birthdate, sex
pseudo_path_df['sex'] = pseudo_path_df['PatientSex']
pseudo_path_df['birthdate'] = [datetime.strptime(i, "%Y%m%d").date() for i in pseudo_path_df['PatientBirthDate']]
pseudo_path_df['RO_datum'] = [datetime.strptime(i, "%Y%m%d").date() for i in pseudo_path_df['StudyDate']]
pseudo_path_df['age'] = [round(int(i.days)/365.25) for i in pseudo_path_df['RO_datum'] - pseudo_path_df['birthdate']]

pseudo_path_df['id_RO_nr'] = pseudo_path_df['id_nr'] + '_' + [str(i) for i in pseudo_path_df['RO_datum']]
pseudo_path_df['view_position'] = 'dp'
pseudo_path_df['photometric_interpretation_new'] = \
    ['MTwo' if i == 'MONOCHROME2' else 'MOne' if i == 'MONOCHROME1' else 'NaN'
     for i in pseudo_path_df['PhotometricInterpretation']]
pseudo_path_df['id_RO_nr_no_hyphan'] = [re.sub('-', '', i) for i in pseudo_path_df['id_RO_nr']]
pseudo_path_df['filename_new_dupl'] = \
    pseudo_path_df['id_RO_nr_no_hyphan'] + '_' + \
    pseudo_path_df['bodypart'] + '_' + \
    pseudo_path_df['laterality'] + '_' + \
    pseudo_path_df['view_position'] + '_' + \
    pseudo_path_df['photometric_interpretation_new'] + '_0'
pseudo_path_df['img_id'] = \
    pseudo_path_df['id_RO_nr_no_hyphan'] + '_' + \
    pseudo_path_df['bodypart'] + '_' + \
    pseudo_path_df['laterality'] + '_' + \
    pseudo_path_df['view_position']
pseudo_path_df.drop(columns=['id_RO_nr_no_hyphan'], inplace=True)
# check that img_ids are unique: len(list(set(list(pseudo_path_df['img_id'])))) == len(pseudo_path_df['img_id'])

# add remaining cols from PAT_DF_MANUAL
cols = pseudo_path_df.keys()
pat_df_manual = pd.read_csv(PAT_DF_MANUAL_PATH)
pat_df_manual_keys = pat_df_manual.keys()
for k in pat_df_manual_keys:
    if k not in cols and not str(k).startswith('r_'):
        if str(k) in ['filename_manual', 'bodypart_manual', 'laterality_manual', 'view_position_manual',
                      'inverted_manual', 'rotated_manual', 'black_manual',
                      'operated_manual', 'inappropriate_manual', 'comment_status', 'comment']:
            pseudo_path_df[k] = 'TBD'
        else:
            pseudo_path_df[k] = None

# save pseudo_pat_df
first_cols = ['LASTNAME', 'FIRSTNAME', 'OLD_NAMES', 'birthdate', 'id_nr', 'id_RO_nr', 'img_id',
              'RO_datum', 'bodypart', 'laterality', 'view_position', 'img_path',
              'sex', 'age', 'filename_manual',
              'rf_pos', 'ccp_pos', 'Beschwerdebeginn', 'Erstdiagnose',
              'bodypart_manual', 'laterality_manual', 'view_position_manual',
              'photometric_interpretation_new',
              'inverted_manual', 'rotated_manual', 'black_manual', 'operated_manual',
              'inappropriate_manual', 'img_of_interest', 'comment_status', 'comment',
              'filename_new_dupl',
              'Groesse', 'Gewicht', 'PatDiagnose_id', 'PatDiagnose_text',
              'DiagToSympt', 'SymptToDiag', 'StatusLevel', 'StatusAvg', 'set', 'segm_set', 'segm_chosen']

pseudo_path_df_reord = pseudo_path_df[first_cols + [i for i in cols if i not in first_cols]]
# pseudo_path_df_reord.to_csv(PSEUD_IMAGES_OUTPUT + os.sep + 'pseudo_path_df_v2.csv')

# long version: decide which tags to keep and which to discard + rename and restructure files (to pre_manual name)
NO_TAGS_DIR = "/Users/Tommi/Documents/STUDIUM/Mac-Medizin/Rheumatologie/" \
              "Rheumatologie AKH Med III Sommer 2017/Data/Ruhrgebiet/Ruhrgebiet_Data_TIFF"

# DICOM De-Identification: https://support.qmenta.com/hc/en-us/articles/209558109-What-is-DICOM-anonymization-
anonym_tags = pd.read_excel('/Users/Tommi/Documents/STUDIUM/Mac-Medizin/Rheumatologie/'
                            'Rheumatologie AKH Med III Sommer 2017/Data/Ruhrgebiet/anonymization_tags.xlsx',
                            header=0)

anonym_tags_remove = anonym_tags[anonym_tags['Anonymization'] == 'X']['Tag Hex Code'].reset_index(drop=True)
anonym_tags_zero = anonym_tags[anonym_tags['Anonymization'] == 'Z']['Tag Hex Code'].reset_index(drop=True)
anonym_tags_dummy = anonym_tags[anonym_tags['Anonymization'] == 'D']['Tag Hex Code'].reset_index(drop=True)
anonym_tags_pseudo = anonym_tags[anonym_tags['Anonymization'] == 'P']['Name'].reset_index(drop=True)
anonym_tags_keep = anonym_tags[anonym_tags['Anonymization'] == 'K']['Name'].reset_index(drop=True)

# [pydicom.datadict.keyword_for_tag(k) for k in mandatory_tags]
# mandatory_defaults = ['keep', ]

for row_nr in range(len(pseudo_path_df_reord)):
    old_path = pseudo_path_df_reord['img_path'][row_nr]
    new_path = NO_TAGS_DIR + os.sep + pseudo_path_df_reord['filename_new_dupl'][row_nr] + '.dcm'
    xray_i = pydicom.dcmread(old_path)
    for t in anonym_tags_remove:
        tag_i = (int(t.split(sep=',')[0], base=16), int(t.split(sep=',')[1], base=16))
        if tag_i in [i for i in xray_i.keys()]:
            print('YES:', t, ' in keys for', old_path)
            xray_i[(int(t.split(sep=',')[0], base=16), int(t.split(sep=',')[1], base=16))].value = ''
        else:
            print(t, 'not in keys for', old_path)
    for t in anonym_tags_zero:
        tag_i = (int(t.split(sep=',')[0], base=16), int(t.split(sep=',')[1], base=16))
        if tag_i in [i for i in xray_i.keys()]:
            print('YES:', t, ' in keys for', old_path)
            xray_i[(int(t.split(sep=',')[0], base=16), int(t.split(sep=',')[1], base=16))].value = ''
        else:
            print(t, 'not in keys for', old_path)
    for t in anonym_tags_dummy:
        tag_i = (int(t.split(sep=',')[0], base=16), int(t.split(sep=',')[1], base=16))
        if tag_i in [i for i in xray_i.keys()]:
            print('YES:', t, ' in keys for', old_path)
            xray_i[tag_i].value = 'anonymous'
        else:
            print(t, 'not in keys for', old_path)

    # Pseudonymization Tags
    xray_i.PatientName = pseudo_path_df_reord['id_nr'][row_nr]
    xray_i.PatientID = pseudo_path_df_reord['id_nr'][row_nr]
    xray_i.IssuerOfPatientID = 'PLACE_HOLDER_FOR_MANUAL_NAME'
    xray_i.ReferringPhysicianName = pseudo_path_df_reord['filename_new_dupl'][row_nr]
    xray_i.StudyID = pseudo_path_df_reord['filename_new_dupl'][row_nr]

    xray_i.SeriesNumber = row_nr
    xray_i.SOPInstanceUID = str(row_nr)
    xray_i.InstanceNumber = row_nr

    # keep tags: do nothing
    # xray_i.StudyDate = pseudo_path_df_reord['id_nr'][row_nr]
    # xray_i.PhotometricInterpretation
    # xray_i.Modality =

    # Kategorisierungs-Tags
    xray_i.BodyPartExamined = pseudo_path_df_reord['bodypart'][row_nr]
    xray_i.Laterality = pseudo_path_df_reord['laterality'][row_nr]
    xray_i.ViewPosition = pseudo_path_df_reord['view_position'][row_nr]

    # save xray with changed metadata and changed name
    save_all_to_main_dir = False
    img_file_extension = '.tiff'

    if not os.path.exists(NO_TAGS_DIR):
        os.mkdir(NO_TAGS_DIR)
    if save_all_to_main_dir:
        id_RO_dir = NO_TAGS_DIR
    else:
        id_RO_dir = NO_TAGS_DIR + os.sep + pseudo_path_df_reord['id_RO_nr'][row_nr]
    if not os.path.exists(id_RO_dir):
        os.mkdir(id_RO_dir)

    if img_file_extension == '.dcm':
        xray_i.save_as(id_RO_dir + os.sep + pseudo_path_df_reord['filename_new_dupl'][row_nr] + '.dcm',
                       write_like_original=True)
    else:
        im_array = xray_i.pixel_array
        shape = im_array.shape
        # Convert to float to avoid overflow or underflow losses.
        image_2d = im_array.astype(float)
        # Rescaling grey scale between 0-255
        image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0
        # Convert to uint
        image_2d_scaled = np.uint8(image_2d_scaled)
        matplotlib.image.imsave(id_RO_dir + os.sep + pseudo_path_df_reord['filename_new_dupl'][row_nr] +
                                img_file_extension,
                                image_2d_scaled, cmap='gray')


# think about use in train/val/test set --> where do I put them? Maybe decide after Gabi scored them & you know distrib.

#
