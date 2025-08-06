
# import ra_utils.autoscora.autoscorRA_Pipeline.input.constants.input_constants_cw_dev as const
#import ra_utils.autoscora.autoscorRA_Pipeline.input.constants.augmentation_constants as augm
import ra_utils.autoscora.autoscorRA_Pipeline.patch_extraction.io_patch_extraction as iop
import ra_utils.autoscora.autoscorRA_Pipeline.patch_extraction.patch_extraction_func as pe
from ra_utils.autoscora.autoscorRA_Pipeline.img_processing.dcm_to_npy import XRay
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

from importlib import resources
from tqdm import tqdm
import ra_utils
import ra_utils.utils.config_parser


# roi extraction settings from patch_fit
def load_package_parameters(filename: str, package_resource="ra_utils.resources.patch_extraction"):
    with resources.files(package_resource).joinpath(filename).open("r") as f:
        return json.load(f)

start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
print("running patch_saving.py at " + start_time)


config = ra_utils.utils.config_parser.load_config(
    default_config="/home/cwatzenboeck/code/RA/ra_utils/runs/config_patches/F_patch_extraction_580_LOCAL.yml",  # for debugging
    debugging_in_jupyter_nb=False, silencium=False)

import matplotlib
matplotlib_backend = config.get("matplotlib_backend", "agg")  # agg, "TkAgg"
matplotlib.use(matplotlib_backend)
import matplotlib.pyplot as plt


extremity = config["extremity"]  # "foot"  # "hand"
img_format_input = config["img_format_input"]#".dcm"  # '.npy'  # '.dcm'
img_dir = config["image_dir"]
pred_joints_file = config["landmarks_csv"]
patch_dir_root = config["output_dir"]
seperate_output_folder_structure_by_roi = config.get("seperate_output_folder_structure_by_roi", False)


def input_checks(config: dict):
    if config["extremity"] not in ["hand", "foot"]:
        raise ValueError("extremity must be either 'hand' or 'foot'")
    if config["img_format_input"] not in [".dcm", ".npy"]:
        raise ValueError("img_format_input must be either '.dcm' or '.npy'")


input_checks(config)

if extremity == "hand":
    corners_list = ['RD', 'RP', 'UP', 'UD']
    finger_setup = iop.finger_joints_setup_hands
elif extremity == "foot":
    corners_list = ['TD', 'TP', 'FP', 'FD']
    finger_setup = iop.finger_joints_setup_feet


joints = iop.import_joints_from_csv_to_dict(file=pred_joints_file, img_colname="img")
finger_points = iop.joints_to_points_dict_for_patch_extraction(joints=joints, setup=finger_setup)

if extremity == "hand":
    wrist_points = iop.joints_to_points_dict_for_patch_extraction(joints=joints,
                                                                  setup=iop.wrist_by_center_setup)
    cmcgroup_points = {}  # {'SCD': None}


# load parameters to map reference size to length of DP
params = load_package_parameters("patch_fit_parameters_H.json") if extremity == "hand" else \
         load_package_parameters("patch_fit_parameters_F.json")


# augmentation modifications
modification = {'rotate_degrees': 0,
                'shift_UR_by': 0,
                'shift_DP_by': 0,
                'resize_UR_by': config.get("resize_UR_by", 0),
                'resize_DP_by': None}



if not os.path.isdir(patch_dir_root):
    os.makedirs(patch_dir_root, exist_ok=True)
    print("created directory:", patch_dir_root)

# iterator elements
dir_img_names = iop.get_img_basenames_from_array_dir(array_dir=img_dir)

img_names = [i for i in joints.keys() if i in dir_img_names]
img_not_in_dir = [i for i in joints.keys() if i not in dir_img_names]
img_not_in_joints = [i for i in dir_img_names if i not in joints.keys()]
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

# if len(sys.argv) > 2:
#     lo = int(sys.argv[1])
#     hi = int(sys.argv[2])
# elif len(sys.argv) == 2:
#     lo = int(sys.argv[1])
#     hi = len(img_names)
# else:
#     lo = 0
#     hi = len(img_names)

lo = config.get("image_index_start", 0)
hi = config.get("image_index_end", len(img_names))
if lo < 0 or hi > len(img_names):
    raise ValueError(f"lo and hi must be between 0 and {len(img_names)}")
if lo > hi:
    raise ValueError(f"lo must be less than hi. {lo} > {hi}")
if lo == hi:
    raise ValueError(f"lo and hi must be different. {lo} == {hi}")
if hi > len(img_names):
    raise ValueError(f"hi must be less than {len(img_names)}. {hi} > {len(img_names)}")


print("images", lo, "to", hi)

for img in tqdm(img_names[lo:hi]):
    print(img)

    if img_format_input == ".npy":
        array = np.load(img_dir + os.sep + img + ".npy")
    elif img_format_input == ".dcm":
        xray = XRay(img_dir + os.sep + img + ".dcm")
        array = xray.img.pixel_array
    else:
        raise ValueError("img_format must be either '.dcm' or '.npy'")

    extremity_ref = pe.ref_size_from_bone_lengths(joints=joints[img], extremity=extremity)

    roi_indexes_to_use = config.get("roi_indexes_to_use", np.arange(0, len(roi_names)))
    for roi_i, roi in enumerate(roi_names):
        if roi_i not in roi_indexes_to_use:
            print("skipping roi", roi)
            continue
        # print(roi)
        ref_to_length_DP = params[roi]['length_DP'][1]
        rel_shift_DP = -params[roi]['center_shift_DP'][1]/ref_to_length_DP

        if roi in roi_finger:
            corners, extra_points_for_debugging = pe.finger_centers_to_corners(points=finger_points[img][roi], ref_size=extremity_ref,
                                                    ref_to_size_factor=ref_to_length_DP, center_shift=rel_shift_DP, 
                                                    return_extra_points=True)
        elif roi in roi_wrist:
            corners, extra_points_for_debugging = pe.finger_centers_to_corners(points=wrist_points[img][roi], ref_size=extremity_ref,
                                                    ref_to_size_factor=ref_to_length_DP, center_shift=rel_shift_DP,
                                                    return_extra_points=True)
        elif roi in roi_cmcgroup:
            corners = {}
            extra_points_for_debugging = None

        else:
            raise Exception(roi + " not in roi_finger, roi_wrist, or roi_cmcgroup")
        
        corners_original = corners.copy()
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
        
        out_dim = np.array(config.get("out_dim", [128, 128]))
        patch = pe.patch_cutter(img=array, rectangle_measures=None, rectangle_corners=modified_corners,
                                square=True, resize_patch=out_dim,
                                padd_patch=None, base_crop=int(3),
                                plot=config.get("plot", False), 
                                plot_type = config.get("plot_type", "show_steps"),
                                print_log=False, 
                                other_corners_to_plot=corners_original if config.get("plot_other_corners_as_well", False) else None,
                                crop_back_border=config.get("crop_back_border", True),
                                extra_points_for_debugging = extra_points_for_debugging, 
                                roi_name = roi
                                )

        # save patch
        if seperate_output_folder_structure_by_roi: 
            patch_dir = os.path.join(patch_dir_root, roi)
            if not os.path.isdir(patch_dir):
                print("creating directory:", patch_dir)
                os.mkdir(patch_dir)
        else: 
            patch_dir = patch_dir_root

        save_path = patch_dir + os.sep + img + "_" + roi + ".npy"
        np.save(save_path, patch)
        del modified_corners
        del patch
        gc.collect()

        del corners
        gc.collect()
    #exit(0) # DEBUGGING

    del extremity_ref
    del array
    gc.collect()


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
