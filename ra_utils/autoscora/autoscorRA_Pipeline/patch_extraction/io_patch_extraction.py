import numpy as np
import pandas as pd
import input.constants.input_constants_cw as const
import os
import re

# nomenclature
# joints = MCP1-D, etc.
# points = center, axis-prox, etc.
# corners = patch corners
# rois = SMP1, etc.

finger_joints_setup_master_hands = {
    "SCDÄ": {"center": "CMCÄ-D",
             "axis_prox": "MCBÄ",
             "axis_dist": "MCPÄ-P"},
    "SMPÄ": {"center": "MCPÄ-P",
             "axis_prox": "MCBÄ",
             "axis_dist": "MCPÄ-P"},
    "SMDÄ": {"center": "MCPÄ-D",
             "axis_prox": "MCPÄ-D",
             "axis_dist": "PIPÄ-P"},
    "SPPÄ": {"center": "PIPÄ-P",
             "axis_prox": "MCPÄ-D",
             "axis_dist": "PIPÄ-P"},
    "SPDÄ": {"center": "PIPÄ-D",
             "axis_prox": "PIPÄ-D",
             "axis_dist": "DIPÄ-P"},
}

finger_joints_setup_master_feet = {
    "STPÄ": {"center": "MTPÄ-P",
             "axis_prox": "MTBÄ",
             "axis_dist": "MTPÄ-P"},
    "STDÄ": {"center": "MTPÄ-D",
             "axis_prox": "MTPÄ-D",
             "axis_dist": "FIPÄ-P"},
    "SIPÄ": {"center": "FIPÄ-P",
             "axis_prox": "MTPÄ-D",
             "axis_dist": "FIPÄ-P"},
    "SIDÄ": {"center": "FIPÄ-D",
             "axis_prox": "FIPÄ-D",
             "axis_dist": "FDPÄ-P"}
}


finger_nr_hands = {'SCDÄ': [1, 3, 4, 5],
                   'SMPÄ': [1, 2, 3, 4, 5], 'SMDÄ': [1, 2, 3, 4, 5],
                   'SPPÄ': [1, 2, 3, 4, 5], 'SPDÄ': [1, 2, 3, 4, 5]
                   }

finger_nr_feet = {'STPÄ': [1, 2, 3, 4, 5], 'STDÄ': [1, 2, 3, 4, 5],
                  'SIPÄ': [1], 'SIDÄ': [1]}


wrist_by_center_setup = {"SWR": {"center": "WRC",
                                 "axis_prox": "MCB3",
                                 "axis_dist": "MCP3-P"}}


def finger_joints_master_to_finger_joints_setup(setup_master, finger_nr, placeholder='Ä'):
    """
    :param setup_master: dict of finger/toe_joint_setup
    :param finger_nr: dict of lists, for each key of setup, which fingers (1-5) are relevant
    :param placeholder: placeholder used instead of finger nr in setup
    :return: joint_setup_long dict, with
                                    each actual joint name as keys and
                                    {center: c, axis_prox: p, axis_dist: d} as values
    """
    return_dict = {}
    for roi, points in setup_master.items():
        for n in finger_nr[roi]:
            return_dict[re.sub(placeholder, str(n), roi)] = {k: re.sub(placeholder, str(n), p)
                                                             for k, p in points.items()}
    return return_dict


finger_joints_setup_hands = finger_joints_master_to_finger_joints_setup(setup_master=finger_joints_setup_master_hands,
                                                                        finger_nr=finger_nr_hands,
                                                                        placeholder='Ä')

finger_joints_setup_feet = finger_joints_master_to_finger_joints_setup(setup_master=finger_joints_setup_master_feet,
                                                                       finger_nr=finger_nr_feet,
                                                                       placeholder='Ä')


def import_joints_from_csv_to_dict(file=const.JOINTS_PATH_GT_100, img_colname="img", sep=',', header='infer'):
    """
    :param file: str, path to csv containing joint coords,
                      with columns img_column, joint1-X, joint1-Y, joint2-X, ...
                      and 1 row for each image
    :param img_colname: str, name of column that contains image names
    :param sep: see pd.read_csv
    :param header: see pd.read_csv
    :return: dict, {img1:
                         {joint1: np.array([joint1-X, joint1-Y]),
                          joint2: np.array([joint2-X, joint2-Y]),
                          ...},
                    img2:
                         {joint1: np.array([joint1-X, joint1-Y]),
                          joint2: np.array([joint2-X, joint2-Y]),
                          ...},
                    ...}
    """
    # load csv
    joints_df = pd.read_csv(file, sep=sep, header=header)
    # use img_colname column as row index
    joints_df.set_index(img_colname, inplace=True)
    # get names of joints (in csv one column key per coordinate (e.g, MCP1-P-X, MCP1-Y,...) but want one key per point)
    joint_names = list(set([re.sub("((-X)|(-Y))$", '', i) for i in joints_df.keys()]))
    # dict with row index as key and rows in {col: cell} dicts as values: {img_name: {col: cell, ...}, ...}
    joints_dict_raw = joints_df.to_dict(orient='index')
    # reformat dict to {img_name: {joint: np.array([x,y]), ...}, ...}
    joints_dict = {im: {j: np.array([values[j + "-X"], values[j + "-Y"]]) for j in joint_names}
                   for im, values in joints_dict_raw.items()}
    return joints_dict


def joints_to_points_dict_for_patch_extraction(joints, setup):
    """
    :param joints: dict, output from import_joints_from_csv_to_dict()
    :param setup:
    :return: {img1: {roi1:  {point1: np-array[X, Y],
                             ...}
                     ...},
              ...}
    """
    points = {img: {roi: {point: joint_coords[joint_name] for point, joint_name in points.items()}
                    for roi, points in setup.items()}
              for img, joint_coords in joints.items()}
    return points


def import_rois_from_csv_to_dict(file=const.ROIS_PATH_GT_100, img_colname="img",
                                 corners=['RD', 'RP', 'UP', 'UD']):
    """
    :param file: str, path to csv containing roi coords,
                      with columns img_column, roi1-RD-X, roi1-RD-Y, roi1-RP-X, ..., roi2-RD-X, roi2-RD-Y,...
                      and 1 row for each image
    :param img_colname: str, name of column that contains image names

    :return: dict, {img1: {roi1: {'corner_RD': np.array([roi1-RD-X, roi1-RD-Y]),
                                  'corner_RP': np.array([roi1-RP-X, roi1-RP-Y]),
                                  'corner_UP': np.array([roi1-UP-X, roi1-UP-Y]),
                                  'corner_UD': np.array([roi1-UD-X, roi1-UD-Y])},

                          {roi2: {'corner_RD': np.array([roi2-RD-X, roi2-RD-Y]),
                                  'corner_RP': np.array([roi2-RP-X, roi2-RP-Y]),
                                  'corner_UP': np.array([roi2-UP-X, roi2-UP-Y]),
                                  'corner_UD': np.array([roi2-UD-X, roi2-UD-Y])},
                          ...
                          }
                    img2: {...},
                    ...
                    }
    """
    # load csv
    rois_df = pd.read_csv(file)
    # use img_colname column as row index
    rois_df.set_index(img_colname, inplace=True)
    # get names of roi corners (in csv one col key per coordinate (e.g, SMP1-RD-X, SMP1-RD-Y,...) but want 1 key/corner)
    rois_corners_names = list(set([re.sub("((-X)|(-Y))$", '', i) for i in rois_df.keys()]))
    # get names of rois
    re_string = '((-' + ')|(-'.join(corners) + '))'
    rois_names = list(set([re.sub(re_string + "$", '', i) for i in rois_corners_names]))
    # dict with row index as key and rows in {col: cell} dicts as values: {img_name: {col: cell, ...}, ...}
    rois_dict_raw = rois_df.to_dict(orient='index')
    # reformat dict to {img_name: {roi: {corner-RD: np.array([x,y]), ...}, ...}, ...}
    rois_dict = {im: {roi: {"corner_" + corner: np.array([values[roi + "-" + corner + "-X"],
                                                          values[roi + "-" + corner + "-Y"]])
                            for corner in corners}
                      for roi in rois_names}
                 for im, values in rois_dict_raw.items()}
    return rois_dict


def get_img_basenames_from_array_dir(array_dir=const.ARRAY_DIR):
    array_basenames = [os.path.splitext(i)[0] for i in os.listdir(array_dir) if not i.startswith(".")]
    return array_basenames


#
