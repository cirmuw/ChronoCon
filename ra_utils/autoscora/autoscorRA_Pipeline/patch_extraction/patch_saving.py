import input.constants.input_constants as const
import input.constants.augmentation_constants as augm
import patch_extraction.io_patch_extraction as iop
import patch_extraction.patch_extraction_func as pe
from img_processing.dcm_to_npy import XRay
import os
import sys
import datetime
import gc
import pandas as pd
import math
import numpy as np
import json
# from copy import copy, deepcopy
import re
import cv2
import imutils
import warnings
import matplotlib
matplotlib.use(const.MATPLOTLIB_BACKEND)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
print("running patch_saving.py at " + start_time)

extremity = "foot"  # "foot"  # "hand"
img_format = ".npy"  # '.npy'  # '.dcm'
augment = False

if extremity == "hand":
    gt_rois_file = const.ROIS_PATH_GT_100
    gt_joints_file = const.JOINTS_PATH_GT_100
    pred_joints_file = const.JOINTS_PATH_PRED_REST100
    array_dir = const.ARRAY_DIR
    dcm_dir = const.IMAGE_DIR
    corners_list = ['RD', 'RP', 'UP', 'UD']
    finger_setup = iop.finger_joints_setup_hands
    joints_to_rois_parameters_path = const.JOINT_TO_ROIS_PARAMETERS_PATH
    if augment:
        patch_dir = const.AUGM_PATCH_DIR
        modifications = augm.AUGM_LIST_OF_DICTS
    else:
        patch_dir = const.PATCH_DIR
elif extremity == "foot":
    gt_rois_file = const.F_ROIS_PATH_GT_100
    gt_joints_file = const.F_JOINTS_PATH_GT_100
    pred_joints_file = const.F_JOINTS_PATH_PRED_REST100
    array_dir = const.F_ARRAY_DIR
    dcm_dir = const.F_IMAGE_DIR
    corners_list = ['TD', 'TP', 'FP', 'FD']
    finger_setup = iop.finger_joints_setup_feet
    joints_to_rois_parameters_path = const.F_JOINT_TO_ROIS_PARAMETERS_PATH
    if augment:
        patch_dir = const.F_AUGM_PATCH_DIR
        modifications = augm.AUGM_LIST_OF_DICTS  # try same as for hand
    else:
        patch_dir = const.F_PATCH_DIR
else:
    raise ValueError("bodypart must be either 'hand' or 'foot'")

if img_format == ".dcm":
    img_dir = dcm_dir
elif img_format == ".npy":
    img_dir = array_dir
else:
    raise ValueError("img_format must be either '.dcm' or '.npy'")

# use rois for the gt 100 and predict rois from joint coords for the rest
gt_rois = iop.import_rois_from_csv_to_dict(file=gt_rois_file, img_colname="img",
                                           corners=['RD', 'RP', 'UP', 'UD'])
# use joint coord gt for the gt 100 and pred for rest (want the best possible patches for training the SCORING net)
gt_joints = iop.import_joints_from_csv_to_dict(file=gt_joints_file, img_colname="img")
pred_joints_rest = iop.import_joints_from_csv_to_dict(file=pred_joints_file, img_colname="img")

# if the 100 img are also included in the pred_joints df, then the pred values overwrite the gt values
joints = {**gt_joints, **pred_joints_rest}

# get points for patch extraction
finger_points = iop.joints_to_points_dict_for_patch_extraction(joints=joints,
                                                               setup=finger_setup)

if extremity == "hand":
    wrist_points = iop.joints_to_points_dict_for_patch_extraction(joints=joints,
                                                                  setup=iop.wrist_by_center_setup)
    cmcgroup_points = {}  # {'SCD': None}

# roi extraction settings from patch_fit
# Opening JSON file
with open(joints_to_rois_parameters_path) as json_file:
    params = json.load(json_file)

# augmentation modifications
if augment and False:
    modifications_df = pd.DataFrame.from_dict({i: modifications[i] for i in range(len(modifications))}, orient='index')
    modifications_df.loc[[i is None for i in modifications_df['resize_DP_by']], 'resize_DP_by'] = \
        modifications_df.loc[[i is None for i in modifications_df['resize_DP_by']], 'resize_UR_by']
    modifications_df['nr'] = list(range(0, len(modifications)))
    modifications_df.to_csv('/mnthome2/autoscoRA/autoscoRA_Slurm/augm_patch_saving_slurm/augmentation_coding.csv',
                            header=True, index=None, sep=';', mode='w')
if not augment:
    modifications = [{'rotate_degrees': 0,
                      'shift_UR_by': 0,
                      'shift_DP_by': 0,
                      'resize_UR_by': 0,
                      'resize_DP_by': None}]

if not os.path.isdir(patch_dir):
    os.mkdir(patch_dir)

# iterator elements
dir_img_names = iop.get_img_basenames_from_array_dir(array_dir=img_dir)

img_names = [i for i in joints.keys() if i in dir_img_names]
img_not_in_dir = [i for i in joints.keys() if i not in dir_img_names]
img_not_in_joints = [i for i in dir_img_names if i not in joints.keys()]
img_gt_rois = list(gt_rois.keys())
roi_names = list(finger_setup.keys())
roi_finger = list(finger_setup.keys())
if extremity == 'hand':
    # + list(cmcgroup_points[img_names[0]].keys())
    roi_wrist = list(iop.wrist_by_center_setup.keys())
    roi_cmcgroup = []
    roi_names = roi_names + roi_wrist + roi_cmcgroup
else:
    roi_wrist = []
    roi_cmcgroup = []

if len(sys.argv) > 2:
    lo = int(sys.argv[1])
    hi = int(sys.argv[2])
elif len(sys.argv) == 2:
    lo = int(sys.argv[1])
    hi = len(img_names)
else:
    lo = 0
    hi = len(img_names)

print("images", lo, "to", hi)

for img in img_names[lo:hi]:
    print(img)

    if img_format == ".npy":
        array = np.load(img_dir + os.sep + img + ".npy")
    elif img_format == ".dcm":
        xray = XRay(img_dir + os.sep + img + ".dcm")
        array = xray.img.pixel_array
    else:
        raise ValueError("img_format must be either '.dcm' or '.npy'")

    extremity_ref = pe.ref_size_from_bone_lengths(joints=joints[img], extremity=extremity)

    for roi in roi_names:
        # print(roi)
        if img in img_gt_rois:
            raw_corners = gt_rois[img][roi]
            corners = pe.any_corners_to_rectangle_corners(corners=raw_corners, square=False,
                                                          orientation_type="degree", returned_values="corners_dict"
                                                          )
        else:
            ref_to_length_DP = params[roi]['length_DP'][1]
            rel_shift_DP = -params[roi]['center_shift_DP'][1]/ref_to_length_DP

            if roi in roi_finger:
                corners = pe.finger_centers_to_corners(points=finger_points[img][roi], ref_size=extremity_ref,
                                                       ref_to_size_factor=ref_to_length_DP, center_shift=rel_shift_DP)
            elif roi in roi_wrist:
                corners = pe.finger_centers_to_corners(points=wrist_points[img][roi], ref_size=extremity_ref,
                                                       ref_to_size_factor=ref_to_length_DP, center_shift=rel_shift_DP)
            elif roi in roi_cmcgroup:
                corners = {}

            else:
                raise Exception(roi + " not in roi_finger, roi_wrist, or roi_cmcgroup")

        # add corners modification/data augm here (e.g., as a for loop over modifications)
        for i in range(len(modifications)):
            modification = modifications[i]
            # print(str(i) + ":", modification)
            modified_corners = pe.roi_modifier(rectangle_measures=None, rectangle_corners=corners,
                                               modification_order=['rotate',
                                                                   'shift_UR', 'shift_DP',
                                                                   'resize'],
                                               rotate_degrees=modification['rotate_degrees'],
                                               shift_by_UR=modification['shift_UR_by'],
                                               shift_by_DP=modification['shift_DP_by'],
                                               resize_UR_by=modification['resize_UR_by'],
                                               resize_DP_by=modification['resize_DP_by'],
                                               rectangle=True, square=True)

            patch = pe.patch_cutter(img=array, rectangle_measures=None, rectangle_corners=modified_corners,
                                    square=True, resize_patch=np.array([128, 128]),
                                    padd_patch=None, base_crop=int(3),
                                    plot=False, show_steps=True, print_log=False)

            # save patch
            if augment:
                save_path = patch_dir + os.sep + img + "_" + roi + "_augm" + str(i) + ".npy"
            else:
                save_path = patch_dir + os.sep + img + "_" + roi + ".npy"

            np.save(save_path, patch)

            del modified_corners
            del patch
            gc.collect()

        del corners
        gc.collect()

    del extremity_ref
    del array
    gc.collect()
    print("DONE")


"""
# check output
img_names = [re.sub("_SMP2$", "", os.path.splitext(i)[0]) for i in os.listdir(patch_dir)]
for img in img_names:
    patches = {}
    for roi in roi_names:
        patches[roi] = np.load(patch_dir + os.sep + img + "_" + roi + ".npy")
    pe.plot_patches(gt_patches=patches)

for img in img_names[155:156]:
    for roi in roi_names[:]:
        patches = {}
        for i in range(len(modifications)):
            patches[str(i)] = np.load(patch_dir + os.sep + img + "_" + roi + "_augm" + str(i) + ".npy")
        pe.plot_patches(gt_patches=patches)

"""

# delete _augm0 from not augmented patches from feet (saved with augm0 in name initially. For hands everything ok)
"""
import os
import input.constants.input_constants as const
import re
patch_path_list = [i for i in os.listdir(const.F_PATCH_DIR)]
for j in patch_path_list:
    old_path = const.F_PATCH_DIR + os.sep + j
    new_path = const.F_PATCH_DIR + os.sep + re.sub("_augm0", "", j)
    # print(old_path)
    # print(new_path)
    os.rename(old_path, new_path)
"""


#
