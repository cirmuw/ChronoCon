import ra_utils.autoscora.autoscorRA_Pipeline.input.constants.input_constants_cw as const
import ra_utils.autoscora.autoscorRA_Pipeline.patch_extraction.io_patch_extraction as iop
import os
import datetime
# import pandas as pd
import math
import numpy as np
# from copy import copy, deepcopy
import re
import random
import cv2
import imutils
import warnings
import json
from copy import deepcopy
#matplotlib.use(const.MATPLOTLIB_BACKEND)
import matplotlib
matplotlib.use("TkAgg")  # Or "Qt5Agg"
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

print("running patch_extraction.py at " + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


def degrees_to_unit_vector_and_back(orientation_UR=None, zero_thresh=10 ** (-15)):
    if np.shape(orientation_UR) == ():
        old_angle = orientation_UR
        unit_vector = np.array([math.cos(math.radians(old_angle)),
                                math.sin(math.radians(old_angle))])
        unit_vector[abs(unit_vector) < zero_thresh] = 0
        new_orientation = unit_vector
    elif np.shape(orientation_UR) == (2,):
        old_unit_vector = orientation_UR / np.linalg.norm(orientation_UR)
        cos_angle = np.sum(np.multiply(np.array([1, 0]), old_unit_vector))
        if old_unit_vector[1] > 0:
            angle = math.degrees(math.acos(cos_angle))  # angle in degrees
        elif old_unit_vector[1] < 0:
            angle = - math.degrees(math.acos(cos_angle))  # angle in degrees
        else:
            angle = 0
        new_orientation = angle
    else:
        raise Exception('specify only degrees XOR unit_vector!')
    return new_orientation


def vector_rotator(vector, rotation_degrees_pos_x_to_pos_y, zero_thresh=10 ** (-15)):
    angle = rotation_degrees_pos_x_to_pos_y
    x_component = vector[0] * np.array([math.cos(math.radians(angle)),
                                        math.sin(math.radians(angle))])
    y_component = vector[1] * np.array([- math.sin(math.radians(angle)),
                                        math.cos(math.radians(angle))])
    rotated_vector = x_component + y_component
    rotated_vector[abs(rotated_vector) < zero_thresh] = 0
    return rotated_vector


def rectangle_measures_to_corners(measures, returned_values='corners_dict'):
    """
    :param measures: dict with keys [center, length_UR, length_DP, orientation_UR (degrees or unit_vector]
    :param returned_values: ...
    :return: corners_dict or cor
    """
    orientation_UR = measures['orientation_UR']
    if np.shape(orientation_UR) == ():
        orientation_UR = degrees_to_unit_vector_and_back(orientation_UR=orientation_UR)
    orientation_UR = orientation_UR / np.linalg.norm(orientation_UR)

    orientation_DP = vector_rotator(orientation_UR, 90)
    orientation_DP = orientation_DP / np.linalg.norm(orientation_DP)

    length_UR = measures['length_UR']
    length_DP = measures['length_DP']
    center = measures['center']

    length_vector_UR = orientation_UR * length_UR
    length_vector_DP = orientation_DP * length_DP

    corner_RD = center + length_vector_UR / 2 - length_vector_DP / 2
    corner_RP = center + length_vector_UR / 2 + length_vector_DP / 2
    corner_UP = center - length_vector_UR / 2 + length_vector_DP / 2
    corner_UD = center - length_vector_UR / 2 - length_vector_DP / 2

    # return prep
    return_object = {"corner_RD": corner_RD,  # np.int32()
                     "corner_RP": corner_RP,
                     "corner_UP": corner_UP,
                     "corner_UD": corner_UD
                     }

    if returned_values == 'corners_dict':
        pass
    elif returned_values == 'corners_list':
        return_object = [i for key, i in return_object.items()]
    elif returned_values == 'corners_array':
        return_object = np.array([i for key, i in return_object.items()]
                                 # , dtype=np.int32
                                 )
    else:
        raise Exception("returned_values should be one out of "
                        "['corners_dict', 'corners_array', 'corners_list']")

    return return_object


def rectangle_corners_to_measures(corners, returned_values='measures_odict', orientation_type='degree'):
    """
    :param corners: dict
    :param returned_values: measures_odict or measures_list
    :param orientation_type: degree, rad, vector
    :return: measures (data structure corresponding to returned_values)
    """
    corner_RD = corners['corner_RD']
    corner_RP = corners['corner_RP']
    corner_UP = corners['corner_UP']
    corner_UD = corners['corner_UD']

    # check that UR lengths are the same, and that DP lengths are the same
    assert math.isclose(np.linalg.norm(corner_RD - corner_RP),
                        np.linalg.norm(corner_UP - corner_UD),
                        abs_tol=0.001)
    assert math.isclose(np.linalg.norm(corner_RP - corner_RD),
                        np.linalg.norm(corner_UP - corner_UD),
                        abs_tol=0.001)

    # center
    vector_RD_to_RP = corner_RP - corner_RD  # (A - B) zeigt von B nach A
    vector_RD_to_UD = corner_UD - corner_RD
    center = corner_RD + vector_RD_to_RP / 2 + vector_RD_to_UD / 2

    # length
    length_UR = np.mean(np.array([np.linalg.norm(corner_RP - corner_UP),
                                  np.linalg.norm(corner_RD - corner_UD)
                                  ]))
    length_DP = np.mean(np.array([np.linalg.norm(corner_RP - corner_RD),
                                  np.linalg.norm(corner_UP - corner_UD)
                                  ]))

    # orientation_UR
    # unit vector
    unit_vector_UR_P = (corner_RP - corner_UP) / np.linalg.norm(corner_RP - corner_UP)
    unit_vector_UR_D = (corner_RD - corner_UD) / np.linalg.norm(corner_RD - corner_UD)
    unit_vector_DP_R = (corner_RP - corner_RD) / np.linalg.norm(corner_RP - corner_RD)
    unit_vector_DP_U = (corner_UP - corner_UD) / np.linalg.norm(corner_UP - corner_UD)

    unit_vector_UR = np.mean(np.array([unit_vector_UR_P, unit_vector_UR_D]), axis=0)
    unit_vector_DP = np.mean(np.array([unit_vector_DP_R, unit_vector_DP_U]), axis=0)

    # angle (angle between pos x-axis and ulnar-radial axis, counted in direction from the pos x to neg y axis)
    # CAVE: from the pos x to pos y axis = counter clockwise in a coord system
    # = clockwise in an image !! Because y from top to bottom!
    cos_angle = np.sum(np.multiply(np.array([1, 0]), unit_vector_UR))

    if unit_vector_UR[1] > 0:
        angle = math.degrees(math.acos(cos_angle))  # angle in degrees
    elif unit_vector_UR[1] < 0:
        angle = - math.degrees(math.acos(cos_angle))  # angle in degrees
    else:
        angle = 0

    # return prep
    if orientation_type == 'vector':
        orientation_UR = unit_vector_DP
    elif orientation_type == 'degree':
        orientation_UR = angle
    elif orientation_type == 'rad':
        orientation_UR = math.radians(angle)
    else:
        raise Exception("orientation_type should be one out of "
                        "['vector', 'degree', 'rad']")

    if returned_values == 'measures_list':
        return_object = [center, length_UR, length_DP, orientation_UR]
    elif returned_values == 'measures_odict':
        return_object = {"center": center,
                         "length_UR": length_UR,
                         "length_DP": length_DP,
                         "orientation_UR": orientation_UR
                         }
    else:
        raise Exception("returned_values should be 'measures_list' or 'measures_odict'")

    return return_object


def any_corners_to_rectangle_corners(corners=None, square=False,
                                     orientation_type="degree", returned_values="corners_dict"
                                     ):
    """
    APPLY THIS TO THE MANUALLY ANNOTATED ROIS (BECAUSE THESE ARE NOT PERFECT RECTANGLES!!!!)
    THE TRAFO OF CORNERS TO RECTANGLE CORNERS IS NOT IMPLEMENTED YET, JUST PREPARED!
    returns float
    1. square=True: turns rectangle into square with same center and mean side lengths
    2. predicted=True: returns rectangle for the predicted coords (not the annotated ground truth)
    """

    # TODO: already fixed? Fix trafo of any 4 corners to a rectangle --> center correct but corners not
    # (starting with center as Schwerpunkt? Or directly starting from the corners/sidelenghts?)

    # sample data
    # corner_RD = joint.coords_odict[joint.roi_names_odict['SMD4'][0]]
    # corner_RP = joint.coords_odict[joint.roi_names_odict['SMD4'][1]]
    # corner_UP = joint.coords_odict[joint.roi_names_odict['SMD4'][2]]
    # corner_UD = joint.coords_odict[joint.roi_names_odict['SMD4'][3]]
    # rect_corner_RD = corner_RD
    # rect_corner_RP = corner_RP
    # rect_corner_UP = corner_UP
    # rect_corner_UD = corner_UD

    # start
    return_object = None

    corner_RD = corners['corner_RD']
    corner_RP = corners['corner_RP']
    corner_UP = corners['corner_UP']
    corner_UD = corners['corner_UD']

    # corners to rectangle corners
    rect_center = np.array([corner_RD, corner_RP, corner_UP, corner_UD]).mean(axis=0)

    # new unit vectors
    unit_vector_UR_P = (corner_RP - corner_UP) / np.linalg.norm(corner_RP - corner_UP)
    unit_vector_UR_D = (corner_RD - corner_UD) / np.linalg.norm(corner_RD - corner_UD)
    unit_vector_DP_R = (corner_RP - corner_RD) / np.linalg.norm(corner_RP - corner_RD)
    unit_vector_DP_U = (corner_UP - corner_UD) / np.linalg.norm(corner_UP - corner_UD)

    unit_vector_UR = np.mean(np.array([unit_vector_UR_P, unit_vector_UR_D]), axis=0) / \
                     np.linalg.norm(np.mean(np.array([unit_vector_UR_P, unit_vector_UR_D]), axis=0))
    unit_vector_DP = np.mean(np.array([unit_vector_DP_R, unit_vector_DP_U]), axis=0) / \
                     np.linalg.norm(np.mean(np.array([unit_vector_DP_R, unit_vector_DP_U]), axis=0))

    # new lenghts
    length_UR = np.mean(np.array([np.linalg.norm(corner_RP - corner_UP),
                                  np.linalg.norm(corner_RD - corner_UD)
                                  ]))
    length_DP = np.mean(np.array([np.linalg.norm(corner_RP - corner_RD),
                                  np.linalg.norm(corner_UP - corner_UD)
                                  ]))

    # new vectors
    vector_UR = unit_vector_UR * length_UR
    vector_DP = unit_vector_DP * length_DP

    # new corners
    rect_corner_RD = rect_center + vector_UR / 2 - vector_DP / 2
    rect_corner_RP = rect_center + vector_UR / 2 + vector_DP / 2
    rect_corner_UP = rect_center - vector_UR / 2 + vector_DP / 2
    rect_corner_UD = rect_center - vector_UR / 2 - vector_DP / 2

    if square:
        # find center of rectangle from read in coordinates --> is also center of new square
        vector_RD_to_RP = rect_corner_RP - rect_corner_RD  # (A - B) zeigt von B nach A
        vector_RD_to_UD = rect_corner_UD - rect_corner_RD
        center = rect_corner_RD + vector_RD_to_RP / 2 + vector_RD_to_UD / 2

        # new unit vectors
        square_unit_vector_UR_P = (rect_corner_RP - rect_corner_UP) / np.linalg.norm(rect_corner_RP - rect_corner_UP)
        square_unit_vector_UR_D = (rect_corner_RD - rect_corner_UD) / np.linalg.norm(rect_corner_RD - rect_corner_UD)
        square_unit_vector_DP_R = (rect_corner_RP - rect_corner_RD) / np.linalg.norm(rect_corner_RP - rect_corner_RD)
        square_unit_vector_DP_U = (rect_corner_UP - rect_corner_UD) / np.linalg.norm(rect_corner_UP - rect_corner_UD)

        square_unit_vector_UR = np.mean(np.array([square_unit_vector_UR_P, square_unit_vector_UR_D]), axis=0) / \
                                np.linalg.norm(np.mean(np.array([square_unit_vector_UR_P, square_unit_vector_UR_D]),
                                                       axis=0))
        square_unit_vector_DP = np.mean(np.array([square_unit_vector_DP_R, square_unit_vector_DP_U]), axis=0) / \
                                np.linalg.norm(np.mean(np.array([square_unit_vector_DP_R, square_unit_vector_DP_U]),
                                                       axis=0))

        # new lenghts
        square_length_UR = np.mean(np.array([np.linalg.norm(rect_corner_RP - rect_corner_UP),
                                             np.linalg.norm(rect_corner_RD - rect_corner_UD)
                                             ]))
        square_length_DP = np.mean(np.array([np.linalg.norm(rect_corner_RP - rect_corner_RD),
                                             np.linalg.norm(rect_corner_UP - rect_corner_UD)
                                             ]))
        square_length = np.mean(np.array([square_length_UR, square_length_DP]))

        # new vectors
        square_vector_UR = square_unit_vector_UR * square_length
        square_vector_DP = square_unit_vector_DP * square_length

        # new corners
        square_corner_RD = center + square_vector_UR / 2 - square_vector_DP / 2
        square_corner_RP = center + square_vector_UR / 2 + square_vector_DP / 2
        square_corner_UP = center - square_vector_UR / 2 + square_vector_DP / 2
        square_corner_UD = center - square_vector_UR / 2 - square_vector_DP / 2

        new_corner_RD = square_corner_RD
        new_corner_RP = square_corner_RP
        new_corner_UP = square_corner_UP
        new_corner_UD = square_corner_UD

    else:
        new_corner_RD = rect_corner_RD
        new_corner_RP = rect_corner_RP
        new_corner_UP = rect_corner_UP
        new_corner_UD = rect_corner_UD

    if returned_values in ['measures_list', 'measures_odict']:
        new_corners_dict = {"corner_RD": new_corner_RD,  # np.int32()
                            "corner_RP": new_corner_RP,
                            "corner_UP": new_corner_UP,
                            "corner_UD": new_corner_UD
                            }
        return_object = rectangle_corners_to_measures(corners=new_corners_dict,
                                                      returned_values=returned_values,
                                                      orientation_type=orientation_type)

    if returned_values in ['measures_list', 'measures_odict']:
        return_object = return_object
    elif returned_values == 'corners_list':
        return_object = np.array([new_corner_RD, new_corner_RP, new_corner_UP, new_corner_UD]
                                 # , dtype=np.int32
                                 ).flatten().tolist()
    elif returned_values == 'corners_array':
        return_object = np.array([new_corner_RD, new_corner_RP, new_corner_UP, new_corner_UD]
                                 # , dtype=np.int32
                                 )
    elif returned_values == 'corners_dict':
        return_object = {"corner_RD": new_corner_RD,  # np.int32()
                         "corner_RP": new_corner_RP,
                         "corner_UP": new_corner_UP,
                         "corner_UD": new_corner_UD}
    else:
        raise Exception("returned_values should be one out of "
                        "['measures_list', 'measures_odict', 'corners_list', 'corners_array', 'corners_dict']")
    return return_object


def get_shift(bone_edge_center, roi_center, orientation_UR_in_degrees):
    center_to_bone_edge = bone_edge_center - roi_center
    unitvec_UR = degrees_to_unit_vector_and_back(orientation_UR=orientation_UR_in_degrees,
                                                 zero_thresh=10 ** (-15))

    unitvec_UR = unitvec_UR / np.linalg.norm(unitvec_UR)
    unitvec_DP = vector_rotator(vector=unitvec_UR,
                                rotation_degrees_pos_x_to_pos_y=90,
                                zero_thresh=10 ** (-15))
    unitvec_DP = unitvec_DP / np.linalg.norm(unitvec_DP)

    # project onto unitvec_UR/DP
    center_shift_UR = np.dot(center_to_bone_edge, unitvec_UR)
    center_shift_DP = np.dot(center_to_bone_edge, unitvec_DP)

    return [center_shift_UR, center_shift_DP]


def ref_size_from_bone_lengths(joints, extremity="hand"):

    """
    calculates ref_size to input in finger_centers_to_corners() from joint centers via bone lengths
    :param joints: a dict w/ coords of both ends of each bone (metacarpal, proximal phalanx, middle phalanx of dig II-V
    for either hand or foot)

    {'CMC2-D': np.array([CMC2-D-X, CMC2-D-Y]),
     'MCP2-P': np.array([MCP2-P-X, MCP2-P-Y]),
     'MCP2-D': np.array([MCP2-D-X, MCP2-D-Y]),
     'PIP2-P': np.array([PIP2-P-X, PIP2-P-Y]),
     'PIP2-D': np.array([PIP2-D-X, PIP2-D-Y]),
     'DIP2-P': np.array([DIP2-P-X, DIP2-P-Y]),
     'CMC3-D': np.array([CMC3-D-X, CMC3-D-Y]),
     .
     .
     .
     'DIP5-P': np.array([DIP5-P-X, DIP5-P-Y])
     }

    (for foot, replace CMC by TMT and MCP by MTP)
    (if the dict includes other keys, as well, those will be ignored)

    :param extremity: str, "hand" or "foot"
    :return: a scalar, the ref_size for the given extremity
    """
    if extremity == "hand":
        size_bones = {'MC': {'prox_joint': 'CMC', 'dist_joint': 'MCP', 'prox_end': '-D', 'dist_end': '-P',
                             'bone_nr': ['2', '3', '4', '5']},
                      'PP': {'prox_joint': 'MCP', 'dist_joint': 'PIP', 'prox_end': '-D', 'dist_end': '-P',
                             'bone_nr': ['2', '3', '4', '5']},
                      'MP': {'prox_joint': 'PIP', 'dist_joint': 'DIP', 'prox_end': '-D', 'dist_end': '-P',
                             'bone_nr': ['2', '3', '4', '5']},
                      }
    elif extremity == "foot":
        size_bones = {'MT': {'prox_joint': 'TMT', 'dist_joint': 'MTP', 'prox_end': '-D', 'dist_end': '-P',
                             'bone_nr': ['1', '2', '3', '4', '5']},
                      'PP': {'prox_joint': 'MTP', 'dist_joint': 'FIP', 'prox_end': '-D', 'dist_end': '-P',
                             'bone_nr': ['1', '2', '3', '4', '5']},
                      'MP': {'prox_joint': 'FIP', 'dist_joint': 'FDP', 'prox_end': '-D', 'dist_end': '-P',
                             'bone_nr': ['1']},
                      }
    else:
        raise ValueError("extremity must be either 'hand' or 'foot'")

    for bone_key, bone_values in size_bones.items():
        size_bones[bone_key]['bone_lens'] = \
            {bone_key + y: {'prox_joint': bone_values['prox_joint'] + y + bone_values['prox_end'],
                            'dist_joint': bone_values['dist_joint'] + y + bone_values['dist_end'],
                            'bone_len': np.linalg.norm(joints[bone_values['prox_joint'] + y +
                                                              bone_values['prox_end']] -
                                                       joints[bone_values['dist_joint'] + y +
                                                              bone_values['dist_end']])
                            }
             for y in bone_values['bone_nr']}
    size_bones['all_bone_lens'] = {bone_j: size_bones[type_i]['bone_lens'][bone_j]['bone_len']
                                   for type_i in size_bones.keys()
                                   for bone_j in size_bones[type_i]['bone_lens'].keys()}

    ref_size = np.median([bone_length for bone_length in size_bones['all_bone_lens'].values()])

    return ref_size

from copy import copy, deepcopy

def finger_centers_to_corners(points, ref_size, ref_to_size_factor=float(1), center_shift=float(0), 
                              return_extra_points=False
                              ):

    """
    turns finger joint coords to patch corner coords
    :param points: a dict of x/y-coords arrays {"center": np.array([center_of_joint_x, center_of_joint_y]),
                                                "axis_prox": np.array([prox_end_of_axis_x, prox_end_of_axis_y]),
                                                "axis_dist": np.array([dist_end_of_axis_x, dist_end_of_axis_y])
                                                }
    :param ref_size: float, hand/foot reference size (derived from bone lengths)
    :param ref_to_size_factor: float, ref_size * ref_to_size_factor = patch side length
    :param center_shift: float, shift of center of joint
    :param return_extra_points: bool, default=false, return centers and orientation points
    along orientation axis (distal -> proximal (DP)) to reach center of patch
    CAVE: Sign of center_shift matters -> + = DP or - = PD
    + vs. - depends on patch: distal patch of a joint: shift towards distal = neg sign
                              proximal patch of a joint: shift towards proximal = pos sign
    :return: a dict {"corner_RD": corner_RD,  # np.int32()
                     "corner_RP": corner_RP,
                     "corner_UP": corner_UP,
                     "corner_UD": corner_UD
                     }
    """

    # get coords of centers to use to calculate corners
    center = points["center"]
    center_original = copy(center)
    axis_prox = points["axis_prox"]
    axis_dist = points["axis_dist"]

    # side length
    length = ref_size * ref_to_size_factor

    # orientation (DP direction) -- note: (A - B) zeigt von B nach A
    vector_DP = axis_prox - axis_dist
    length_DP = np.linalg.norm(vector_DP)
    unit_vector_DP = vector_DP / length_DP
    orientation = unit_vector_DP

    # center new (shift by center_shift of length along DP orientation, i.e. DP)
    center = center + orientation * length * center_shift

    # calculate corners
    length_vector_DP = orientation * length
    length_vector_UR = np.array([orientation[1], - orientation[0]]) * length
    corner_RD = center + length_vector_UR / 2 - length_vector_DP / 2
    corner_RP = center + length_vector_UR / 2 + length_vector_DP / 2
    corner_UP = center - length_vector_UR / 2 + length_vector_DP / 2
    corner_UD = center - length_vector_UR / 2 - length_vector_DP / 2

    # return prep
    return_object = {"corner_RD": corner_RD,  # np.int32()
                     "corner_RP": corner_RP,
                     "corner_UP": corner_UP,
                     "corner_UD": corner_UD
                     }

    if return_extra_points:
        extra_points = {
            "center_original": center_original, 
            "center": center,
            "axis_prox": axis_prox, 
            "axis_dist": axis_dist,
        } 
        return return_object, extra_points

    return return_object


def roi_modifier(rectangle_measures=None, rectangle_corners=None, modification_order=['rotate',
                                                                                      'shift_UR', 'shift_DP',
                                                                                      'resize'],
                 rotate_degrees=10, shift_by_UR=0.1, shift_by_DP=0.1, resize_UR_by=0.1, resize_DP_by=0.1,
                 rectangle=True, square=False):
    """
    :param rectangle_measures:
    :param rectangle_corners:
    :param modification_order: order of application of transformations
    :param rotate_degrees: increases orientation_UR by rotate_degrees°
    :param shift_by_UR: float, shifts roi by shift_by_UR * 100 % of length_UR along UR axis
    :param shift_by_DP: float, shifts roi by shift_by_DP * 100 % of length_DP along DP axis
    :param resize_UR_by: float, resizes length_UR by resize_UR_by * 100 % of length_UR along UR axis
    :param resize_DP_by: float, resizes length_DP by resize_DP_by * 100 % of length_DP along DP axis
    :return: new corners dict
    """
    
    if rectangle_measures is None and rectangle_corners is None:
        raise ValueError("rectangle_measures is None AND rectangle_corners is None: only one allowed to be None")

    if rectangle_measures is None:
        rectangle_measures = rectangle_corners_to_measures(rectangle_corners,
                                                           returned_values='measures_odict',
                                                           orientation_type='degree')
    current_measures = rectangle_measures
    
    # dict of operations
    tasks = {}

    # assigns a function with decorator task to the dictionary tasks and calls it
    task = lambda f: tasks.setdefault(f.__name__, f)

    # decorated operations
    @task
    def rotate(measures, degrees=rotate_degrees):
        new_degrees = measures['orientation_UR'] + degrees
        new_measures = deepcopy(measures)
        new_measures['orientation_UR'] = new_degrees
        # nonloncal current_measures
        # current_measures['orientation_UR'] = new_degrees
        return new_measures

    @task
    def shift_UR(measures, shift_by=shift_by_UR):
        old_center = measures['center']
        old_unit_vector = degrees_to_unit_vector_and_back(orientation_UR=measures['orientation_UR'])
        shift_unit_vector = old_unit_vector
        old_length = measures['length_UR']

        shift_unit_vector = shift_unit_vector/np.linalg.norm(shift_unit_vector)
        shift_vector = shift_unit_vector * old_length * shift_by
        new_center = old_center + shift_vector
        new_measures = deepcopy(measures)
        new_measures['center'] = new_center
        return new_measures
    
    @task
    def shift_DP(measures, shift_by=shift_by_DP):
        old_center = measures['center']
        old_unit_vector = degrees_to_unit_vector_and_back(orientation_UR=measures['orientation_UR'])
        shift_unit_vector = vector_rotator(vector=old_unit_vector,
                                           rotation_degrees_pos_x_to_pos_y=90)
        old_length = measures['length_DP']

        shift_unit_vector = shift_unit_vector/np.linalg.norm(shift_unit_vector)
        shift_vector = shift_unit_vector * old_length * shift_by
        new_center = old_center + shift_vector
        new_measures = deepcopy(measures)
        new_measures['center'] = new_center
        return new_measures

    @task
    def resize(measures, resize_UR_by=resize_UR_by, resize_DP_by=resize_DP_by):
        if resize_UR_by is None:
            resize_UR_by = resize_DP_by
        if resize_DP_by is None:
            resize_DP_by = resize_UR_by
        if resize_DP_by is None and resize_UR_by is None:
            resize_UR_by = 0
            resize_DP_by = 0
        new_length_UR = measures['length_UR'] * (1 + resize_UR_by)
        new_length_DP = measures['length_DP'] * (1 + resize_DP_by)
        new_measures = deepcopy(measures)
        new_measures['length_UR'] = new_length_UR
        new_measures['length_DP'] = new_length_DP
        return new_measures

    # runs the function "key" by selecting it via the task decorator
    def my_main(key, measures):
        new_measures = tasks[key](measures=measures)
        return new_measures

    for i in range(0, len(modification_order)):
        current_measures = my_main(modification_order[i], measures=current_measures)

    # return prep
    return_measures = current_measures
    return_corners = rectangle_measures_to_corners(measures=return_measures,
                                                   returned_values='corners_dict')
    if rectangle or square:
        return_corners = any_corners_to_rectangle_corners(corners=return_corners, square=square,
                                                          orientation_type="degree", returned_values="corners_dict")
    return return_corners

from typing import Literal, Optional
def patch_cutter(
    img, 
    rectangle_measures=None, 
    rectangle_corners=None,
    square=False, 
    resize_patch=None, 
    padd_patch=None, 
    base_crop=int(3),
    plot=False, 
    plot_type: Literal["show_steps", "show_input_and_output", "show_only_output"] = "show_steps",
    print_log=False, 
    other_corners_to_plot=None,  # plot additional corners
    crop_back_border=True,
    extra_points_for_debugging: Optional[dict] = None, 
    roi_name = ""
    ):
    """
    :param img: numpy array
    :param rectangle_measures:
    :param rectangle_corners:
    :param square:
    :param resize_patch:
    :param padd_patch: target shape after padding np.array([heigth, width])
    :param base_crop:
    :param plot:
    :param show_steps:
    :param print_log: bool, if False suppresses print() output
    :return:
    """
    
    
    
    if rectangle_measures is None and rectangle_corners is None:
        raise ValueError("rectangle_measures is None AND rectangle_corners is None: only one allowed to be None")

    if rectangle_measures is None:
        rectangle_measures = rectangle_corners_to_measures(rectangle_corners,
                                                           returned_values='measures_odict',
                                                           orientation_type='degree')
    if rectangle_corners is None:
        rectangle_corners = rectangle_measures_to_corners(rectangle_measures,
                                                          returned_values='corners_dict')

    # create mask rectangle described by the calculated corner points
    mask_inv = np.ones(img.shape)  # (height, width)

    angle = - rectangle_measures['orientation_UR']  # rotate the image in opposite direction to get patch upright

    roi = [tuple(coordinates) for corner, coordinates in rectangle_corners.items()]
    """
    roi = [tuple(rectangle_corners['corner_RD']),
           tuple(rectangle_corners['corner_RP']),
           tuple(rectangle_corners['corner_UP']),
           tuple(rectangle_corners['corner_UD'])]
    """
    # [tuple(self.coords_odict[i]) for i in self.roi_names_odict[roi_name]] # (x, y)
    mask = cv2.fillPoly(mask_inv, [np.array(roi, dtype=np.int32)], 0) == 0

    # rotate original image and masked image so that borders of patch are vertical and horizontal, respectively
    rotated = imutils.rotate_bound(img, angle)
    masked_img = np.multiply(img, mask)
    rotated_masked = imutils.rotate_bound(masked_img, angle)

    # crop everything except patch
    rotated_mask = rotated_masked > 0
    pixel_dtype = img.__getattribute__('dtype')
    pixel_max = np.iinfo(pixel_dtype).max
    rotated_mask = np.array(np.multiply(rotated_mask, pixel_max), dtype=pixel_dtype)

    y1 = np.min(np.where(rotated_mask == pixel_max)[0])
    x1 = np.min(np.where(rotated_mask == pixel_max)[1])
    y2 = np.max(np.where(rotated_mask == pixel_max)[0])
    x2 = np.max(np.where(rotated_mask == pixel_max)[1])

    raw_patch = rotated_masked[y1:y2, x1:x2]
    # plt.imshow(raw_patch, cmap="gray")
    # plt.title("raw patch before cropping")
    # plt.show()

    # if present, crop black borders that occur due to small errors in calculating rotation
    cropped_patch = raw_patch
    if crop_back_border:
        check_0 = np.any(cropped_patch == 0)
        cropped_by = 0
        while check_0:
            cropped_patch = cropped_patch[1:-1, 1:-1]
            check_0 = np.any(np.array([np.any(cropped_patch[1, :] == 0),
                                    np.any(cropped_patch[-1, :] == 0),
                                    np.any(cropped_patch[:, 1] == 0),
                                    np.any(cropped_patch[:, -1] == 0)]))
            cropped_by += 1
        if not check_0:
            # additional cropping because had weird dotted line on all sides after rotation
            if base_crop > 0:
                cropped_patch = cropped_patch[base_crop:-base_crop, base_crop:-base_crop]
            if print_log:
                print("removed black borders --> check_0 = " + str(check_0) +
                    " (after cropping all 4 borders by " + str(cropped_by) + str(base_crop) + " )")
        if check_0:
            warnings.warn("what's happening with those black borders of the patch?")

    # plt.imshow(cropped_patch, cmap="gray")
    # plt.title("before cropped to square")
    # plt.show()

    # crop to square
    if square:  # cuts longer axis equally on both sides
        diff = abs(cropped_patch.shape[1] - cropped_patch.shape[0])
        crop_len = math.ceil(diff / 2)
        if crop_len > 0:
            if np.argmin(cropped_patch.shape) == 0:
                square_patch = cropped_patch[:, crop_len:-crop_len]
                if print_log:
                    print("cropped to square by cropping axis 0 (y) by", crop_len, "on each side")
            elif np.argmin(cropped_patch.shape) == 1:
                square_patch = cropped_patch[crop_len:-crop_len, :]
                if print_log:
                    print("cropped to square by cropping axis 1 (x) by", crop_len, "on each side")
            else:
                raise Exception()
        else:
            square_patch = cropped_patch
            if print_log:
                print("patch was already square before cropping to square: obviously not cropped any further now")
        patch = square_patch
        # plt.imshow(patch, cmap="gray")
        # plt.title("cropped to square")
        # plt.show()

    else:
        patch = cropped_patch

    # padding
    if padd_patch is not None:
        y_padding = np.max(0, np.ceil(padd_patch[0] - np.shape(patch)[0]))
        x_padding = np.max(0, np.ceil(padd_patch[1] - np.shape(patch)[1]))

        pad_y = (int(np.floor(y_padding / 2)), int(np.ceil(y_padding / 2)))
        pad_x = (int(np.floor(x_padding / 2)), int(np.ceil(x_padding / 2)))
        padding = [pad_y, pad_x]

        patch = np.pad(patch, padding, mode='constant')
        if print_log:
            print("padded by", padding, "to shape", padd_patch)

        # plt.imshow(patch, cmap="gray")
        # plt.title("padded")
        # plt.show()

    # resize
    if resize_patch is not None:
        patch = cv2.resize(patch, (resize_patch[0], resize_patch[1]), interpolation=cv2.INTER_LINEAR)
        if print_log:
            print("resized to", resize_patch)
        # plt.imshow(patch, cmap="gray")
        # plt.title("resized")
        # plt.show()

    # plot
    if plot:
        if plot_type == "show_steps": 
            # prep plot
            corner_arr = np.array([coordinates for corner, coordinates in rectangle_corners.items()])
            corner_names = [corner for corner, coordinates in rectangle_corners.items()]
            # plots
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(321)
            ax1.imshow(img, cmap="gray")
            ax1.scatter(corner_arr[:, 0], corner_arr[:, 1], s=3)
            # for i in np.arange(0, len(corner_names)):
            #     ax1.annotate(corner_names[i], (corner_arr[i, 0], corner_arr[i, 1]), color='red',
            #                  fontsize='x-small')
            
            # Add the centers and other points of interrest to the plot: 

            if extra_points_for_debugging != None: 
                arr = np.array([coordinates for _, coordinates in extra_points_for_debugging.items()])
                names = [n for n, _ in extra_points_for_debugging.items()]
                for i in np.arange(0, len(extra_points_for_debugging)):
                    # ax1.annotate(names[i], 
                    #              (arr[i, 0], arr[i, 1]), 
                    #              color='red', fontsize='x-small')
                    ax1.scatter(arr[i, 0], arr[i, 1], marker='x', color='red')
                
            ax1.add_patch(patches.Polygon(corner_arr, closed=True, edgecolor='green', fill=False, linewidth=3))
            if other_corners_to_plot is not None:
                corner_arr = np.array(list(other_corners_to_plot.values()))
                ax1.add_patch(patches.Polygon(corner_arr, closed=True, fill=False, edgecolor='blue', linewidth=3))

            # ax1.set_ylim(ax1.get_ylim()[::-1])
            ax2 = fig1.add_subplot(322)
            ax2.imshow(np.multiply(img, mask), cmap="gray")
            ax3 = fig1.add_subplot(323)
            ax3.imshow(rotated, cmap="gray")
            ax4 = fig1.add_subplot(324)
            ax4.imshow(rotated_masked, cmap="gray")
            ax5 = fig1.add_subplot(325)
            ax5.imshow(raw_patch, cmap="gray")
            ax5.set_title("raw patch")
            ax6 = fig1.add_subplot(326)
            ax6.imshow(patch, cmap="gray")
            patch_title_insert = "patch cropped"
            if square:
                patch_title_insert = patch_title_insert + ", cropped to square"
            if padd_patch is not None:
                patch_title_insert = patch_title_insert + ", padded"
            if resize_patch is not None:
                patch_title_insert = patch_title_insert + ", resized"
            ax6.set_title(patch_title_insert)

            fig1.tight_layout()

        elif plot_type == "show_only_output":
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            ax1.imshow(patch, cmap="gray")
            patch_title_insert = "patch cropped"
            if square:
                patch_title_insert = patch_title_insert + ", cropped to square"
            if padd_patch is not None:
                patch_title_insert = patch_title_insert + ", padded"
            if resize_patch is not None:
                patch_title_insert = patch_title_insert + ", resized"
            ax1.set_title(patch_title_insert)
            
        elif plot_type == "show_input_and_output":
            # prepare corner and debug point arrays
            corner_arr = np.array([coordinates for corner, coordinates in rectangle_corners.items()])
            corner_names = [corner for corner, coordinates in rectangle_corners.items()]

            fig1 = plt.figure(figsize=(10, 5))
            ax1 = fig1.add_subplot(121)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.imshow(img, cmap="gray")
            ax1.scatter(corner_arr[:, 0], corner_arr[:, 1], s=3)
            ax1.add_patch(patches.Polygon(corner_arr, closed=True, edgecolor='green', fill=False, linewidth=3))
            
            # Add the centers and other debugging points
            if extra_points_for_debugging is not None:
                arr = np.array([coordinates for _, coordinates in extra_points_for_debugging.items()])
                names = [n for n, _ in extra_points_for_debugging.items()]
                for i in np.arange(0, len(extra_points_for_debugging)):
                    ax1.scatter(arr[i, 0], arr[i, 1], marker='x', color='lightgreen', s=35)

            if other_corners_to_plot is not None:
                corner_arr_other = np.array(list(other_corners_to_plot.values()))
                ax1.add_patch(patches.Polygon(corner_arr_other, closed=True, fill=False, edgecolor='purple', linewidth=3))


            ax1.set_title(f"Input Image with ROI  {roi_name}")

            ax2 = fig1.add_subplot(122)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.imshow(patch, cmap="gray")
            patch_title_insert = "patch cropped"
            if square:
                patch_title_insert += ", cropped to square"
            if padd_patch is not None:
                patch_title_insert += ", padded"
            if resize_patch is not None:
                patch_title_insert += ", resized"
            ax2.set_title(patch_title_insert)

            fig1.tight_layout()
        plt.show()

    return patch


def reshape_with_padding(array, target_shape):

    """
    Reshapes a numpy array to target_shape. If aspect ratio of array differs from that of target_shape, approriate
    padding is applied to x or y axis. Only then resampling is done, so that the input array is not distorted.

    :param array: 2D numpy array
    :param target_shape: int, (height, width)
    :return: padded and reshaped numpy array
    """

    target_ratio = target_shape[0]/target_shape[1]
    array_shape = array.shape
    array_ratio = array_shape[0] / array_shape[1]

    if array_ratio > target_ratio:
        padded_width = round(array_ratio / target_ratio * array_shape[1], 5)
        # "round" above is used to discard some decimals to account for float precision errors
        # otherwise np.floor or np.ceil would subtract/add one int when padded_height is actually exactly an int
        pad_horiz_left = int(np.floor((padded_width - array_shape[1]) / 2))
        pad_horiz_right = int(np.ceil((padded_width - array_shape[1]) / 2))
        padded_array = np.pad(array, ((0, 0), (pad_horiz_left, pad_horiz_right)), 'constant', constant_values=0)
    elif array_ratio < target_ratio:
        padded_height = round(target_ratio / array_ratio * array_shape[0], 5)
        # "round" above is used to discard some decimals to account for float precision errors
        # otherwise np.floor or np.ceil would subtract/add one int when padded_height is actually exactly an int
        pad_vert_up = int(np.floor((padded_height - array_shape[0]) / 2))
        pad_vert_down = int(np.ceil((padded_height - array_shape[0]) / 2))
        padded_array = np.pad(array, ((pad_vert_up, pad_vert_down), (0, 0)), 'constant', constant_values=0)
    else:
        padded_array = array

    assert abs(padded_array.shape[0]/padded_array.shape[1] - target_ratio) <= 0.005

    reshaped_padded_array = cv2.resize(padded_array, (target_shape[1], target_shape[0]),
                                       interpolation=cv2.INTER_LINEAR)
    return reshaped_padded_array


def plot_joints_rois(img, gt_joints=None, pred_joints=None, gt_corners=None, pred_corners=None,
                     annot_joints=False, annot_rois=False, gt_scatter_size=3, pred_scatter_size=3,
                     gt_scatter_color='blue', pred_scatter_color='red',
                     gt_rectangle_fill=True, gt_rectangle_alpha=0.2, gt_rectangle_lw=1,
                     pred_rectangle_fill=True, pred_rectangle_alpha=0.2, pred_rectangle_lw=1,
                     display_legend=True, suppress_plot=False):
    """
    :param img: numpy array
    :param gt_joints: dict of form {joint_1: np.array(X, Y), joint_2: np.array(X, Y)}
    :param pred_joints: dict of form {joint_1: np.array(X, Y), joint_2: np.array(X, Y)}
    :param gt_corners: dict of form: {roi_name_1: {"corner_RD": corner_RD,  # np.int32()
                                                "corner_RP": corner_RP,
                                                "corner_UP": corner_UP,
                                                "corner_UD": corner_UD},
                                   .
                                   .
                                   .
                                   corner_name_n: {"corner_RD": corner_RD,  # np.int32()
                                                "corner_RP": corner_RP,
                                                "corner_UP": corner_UP,
                                                "corner_UD": corner_UD}}
    :param pred_corners: dict of same form as gt_corners
    :param annot_joints: bool, annotate jointss with keys from gt_joints/pred_joints
    :param annot_rois: bool, annotate rois with keys from gt_corners/pred_corners
    :param gt_scatter_size: float, size of scatter plot points representing gt_joints
    :param pred_scatter_size: float, size of scatter plot points representing pred_joints
    :param gt_scatter_color:
    :param pred_scatter_color
    :param gt_rectangle_fill:
    :param gt_rectangle_alpha:
    :param gt_rectangle_lw:
    :param pred_rectangle_fill:
    :param pred_rectangle_alpha:
    :param pred_rectangle_lw:
    :param display_legend:
    :param suppress_plot: bool, if True, plt.show() is not executed
    :return: None
    """

    plt.figure()

    # display the image
    plt.imshow(img, cmap="gray")

    # variables
    legend_text = []
    legend_lines = []

    # add joints and their names
    ## gt_joints
    if gt_joints is not None:
        gt_joints_names = [i for i in gt_joints.keys()]
        gt_joints_list_x = [i[0] for i in gt_joints.values()]
        gt_joints_list_y = [i[1] for i in gt_joints.values()]

        plt.scatter(gt_joints_list_x, gt_joints_list_y,
                    cmap='gray', s=gt_scatter_size, color=gt_scatter_color)
        if annot_joints:
            for i in range(0, len(gt_joints_names)):
                plt.annotate(gt_joints_names[i],
                             (gt_joints_list_x[i], gt_joints_list_y[i]),
                             color=gt_scatter_color, fontsize='x-small')

    ## pred_joints
    if pred_joints is not None:
        pred_joints_names = [i for i in pred_joints.keys()]
        pred_joints_list_x = [i[0] for i in pred_joints.values()]
        pred_joints_list_y = [i[1] for i in pred_joints.values()]

        plt.scatter(pred_joints_list_x, pred_joints_list_y,
                    cmap='gray', s=pred_scatter_size, color=pred_scatter_color)
        if annot_joints:
            for i in range(0, len(pred_joints_names)):
                plt.annotate(pred_joints_names[i],
                             (pred_joints_list_x[i], pred_joints_list_y[i]),
                             color=pred_scatter_color, fontsize='x-small')

    # add roi rectangle(s) and their names
    ## gt_corners
    if gt_corners is not None:
        for roi in gt_corners.keys():
            gt_rectangle_measures = rectangle_corners_to_measures(gt_corners[roi],
                                                                  returned_values='measures_odict',
                                                                  orientation_type='degree')
            gt_rectangle_corners = gt_corners[roi]

            gt_xy = gt_rectangle_corners['corner_UD']  # UD = low-x, low-y corner
            gt_height = gt_rectangle_measures['length_DP']
            gt_width = gt_rectangle_measures['length_UR']
            gt_angle = gt_rectangle_measures['orientation_UR']  # rotates pos x to pos y = clockwise

            ax1 = plt.gca()
            ax1.add_patch(patches.Rectangle(gt_xy, gt_width, gt_height, gt_angle,
                                            alpha=min([1, gt_rectangle_alpha]),
                                            color=gt_scatter_color,
                                            fill=gt_rectangle_fill,
                                            linewidth=gt_rectangle_lw))

            if annot_rois:
                plt.annotate(roi,
                             (gt_rectangle_corners['corner_RP'][0], gt_rectangle_corners['corner_RP'][1]),
                             color=gt_scatter_color, fontsize='x-small')

    ## pred_corners
    if pred_corners is not None:
        for roi in pred_corners.keys():
            pred_rectangle_measures = rectangle_corners_to_measures(pred_corners[roi],
                                                                  returned_values='measures_odict',
                                                                  orientation_type='degree')
            pred_rectangle_corners = pred_corners[roi]

            pred_xy = pred_rectangle_corners['corner_UD']  # UD = low-x, low-y corner
            pred_height = pred_rectangle_measures['length_DP']
            pred_width = pred_rectangle_measures['length_UR']
            pred_angle = pred_rectangle_measures['orientation_UR']  # rotates pos x to pos y = clockwise

            ax1 = plt.gca()
            ax1.add_patch(patches.Rectangle(pred_xy, pred_width, pred_height, pred_angle,
                                            alpha=min([1, pred_rectangle_alpha]),
                                            color=pred_scatter_color,
                                            fill=pred_rectangle_fill,
                                            linewidth=pred_rectangle_lw))

            if annot_rois:
                plt.annotate(roi,
                             (pred_rectangle_corners['corner_RP'][0], pred_rectangle_corners['corner_RP'][1]),
                             color=pred_scatter_color, fontsize='x-small')

    ## legend
    if display_legend:
        if gt_joints is not None or gt_corners is not None:
            legend_text = legend_text + ['ground_truth']
            legend_lines = legend_lines + [Line2D([0], [0], marker='o', color=gt_scatter_color,
                                                  markerfacecolor=gt_scatter_color,
                                                  markersize=gt_scatter_size, lw=1)]

        if pred_joints is not None or pred_corners is not None:
            legend_text = legend_text + ['predicted']
            legend_lines = legend_lines + [Line2D([0], [0], marker='o', color=pred_scatter_color,
                                                  markerfacecolor=pred_scatter_color,
                                                  markersize=pred_scatter_size, lw=1)]

        plt.legend(legend_lines, legend_text)

    # color_iteration = iter(plt.cm.rainbow(np.linspace(0, 1, len(modified_rois_odict.keys()))))
    if not suppress_plot:
        plt.show()


def plot_patches(gt_patches=None, pred_patches=None):

    if pred_patches is None:
        n_patches = len(gt_patches.keys())
        patch_names = list(gt_patches.keys())
    elif gt_patches is not None and pred_patches is not None and len(gt_patches.keys()) > len(pred_patches.keys()):
        n_patches = len(gt_patches.keys())
        patch_names = list(gt_patches.keys())
    else:
        n_patches = len(pred_patches.keys())
        patch_names = list(pred_patches.keys())

    fig = plt.figure()
    fig.suptitle('Extracted Patches from Ground Truth and Predicted ROIs', fontsize=16)
    outer_grid = gridspec.GridSpec(int(np.ceil(n_patches / 5)), 5, wspace=0.2, hspace=0.2)
    for i in range(n_patches):
        patch_grid = gridspec.GridSpecFromSubplotSpec(1, 2,
                                                      subplot_spec=outer_grid[i], wspace=0.1, hspace=0.1)
        if gt_patches is not None and patch_names[i] in gt_patches.keys() and gt_patches[patch_names[i]] is not None:
            patch_ground_truth = gt_patches[patch_names[i]]
        else:
            patch_ground_truth = np.zeros(pred_patches[patch_names[i]].shape, dtype=np.uint16)
        if pred_patches is not None and patch_names[i] in pred_patches.keys() and \
                pred_patches[patch_names[i]] is not None:
            patch_predicted = pred_patches[patch_names[i]]
        else:
            patch_predicted = np.zeros(gt_patches[patch_names[i]].shape, dtype=np.uint16)

        if patch_ground_truth is not None:
            gt_ax = plt.Subplot(fig, patch_grid[0])
            gt_ax.imshow(patch_ground_truth, cmap="gray")
            gt_ax.set_title(patch_names[i] + "_gt")
            fig.add_subplot(gt_ax)

        if patch_predicted is not None:
            pred_ax = plt.Subplot(fig, patch_grid[1])
            pred_ax.imshow(patch_predicted, cmap="gray")
            pred_ax.set_title(patch_names[i] + "_pred")
            fig.add_subplot(pred_ax)

    """
    else:
        if predicted:
            n_patches = len(self.predicted_patch_pixel_odict.keys())
            grid_rows = np.ceil(n_patches / 6)
            grid_cols = 6
            plot_nr = 1
            fig = plt.figure()
            fig.suptitle('Extracted Patches from Ground Truth ROIs', fontsize=16)
            current_patch = np.zeros((200, 200), dtype=np.uint16)
            for patch in self.predicted_patch_pixel_odict.keys():
                ax = fig.add_subplot(grid_rows, grid_cols, plot_nr)
                if self.predicted_patch_pixel_odict[patch] is None:
                    current_patch = np.zeros(current_patch.shape, dtype=np.uint16)
                else:
                    current_patch = self.predicted_patch_pixel_odict[patch]
                ax.imshow(current_patch, cmap="gray")
                ax.set_title(patch)
                plot_nr += 1
        else:
            n_patches = len(self.patch_pixel_odict.keys())
            grid_rows = np.ceil(n_patches / 6)
            grid_cols = 6
            plot_nr = 1
            fig = plt.figure()
            fig.suptitle('Extracted Patches from Predicted ROIs', fontsize=16)
            current_patch = np.zeros((200, 200), dtype=np.uint16)
            for patch in self.patch_pixel_odict.keys():
                ax = fig.add_subplot(grid_rows, grid_cols, plot_nr)
                if self.patch_pixel_odict[patch] is None:
                    current_patch = np.zeros(current_patch.shape, dtype=np.uint16)
                else:
                    current_patch = self.patch_pixel_odict[patch]
                ax.imshow(current_patch, cmap="gray")
                ax.set_title(patch)
                plot_nr += 1
    """

    plt.show()


def custom_cv_pat(img_names, splits=3, save_path_prefix=None, ran_seed=None):
    # https://stackoverflow.com/questions/24078301/custom-cross-validation-split-sklearn
    # ran_seed = 2457
    pat_img_dict = {re.sub("_.*$", "", i): [] for i in img_names}
    for i in img_names:
        pat_img_dict[re.sub("_.*$", "", i)] = pat_img_dict[re.sub("_.*$", "", i)] + [i]

    base_split_len = len(img_names) // splits
    split_lens = [base_split_len] * splits
    split_rest = len(img_names) % splits
    s = splits
    r = split_rest
    while r > 0:
        s = s - 1
        for i in range(s):
            split_lens[i] += r // s
        r = r % s
    max_n = max(split_lens)

    split_imgs = [[]] * splits

    if ran_seed is not None:
        random.seed(a=ran_seed)
    for p, i in pat_img_dict.items():
        p_len = len(i)
        ad_split = random.choice(range(splits))
        n = len(split_imgs[ad_split])
        limit_n = split_lens[ad_split]
        tried_splits = [ad_split]
        untried_splits = [i for i in range(splits) if i not in tried_splits]
        while n + p_len > limit_n:  # max_n:
            if len(untried_splits) == 0:
                raise Exception("")
            ad_split = random.choice(untried_splits)
            n = len(split_imgs[ad_split])
            limit_n = split_lens[ad_split]
            tried_splits = tried_splits + [ad_split]
            untried_splits = [i for i in range(splits) if i not in tried_splits]

        split_imgs[ad_split] = split_imgs[ad_split] + i

    assert [len(i) for i in split_imgs] == split_lens

    if save_path_prefix is not None:
        for i in split_imgs:
            with open(save_path_prefix + "_split_" + str(i), 'w') as fp:
                fp.write('\n'.join(i))

    return split_imgs


def split_img_based_on_pat_split_from_custom_cv_pat(img_names, custom_cv):
    pat_ids_split = [[re.sub("_.*", "", img_j) for img_j in split_i] for split_i in custom_cv]
    imgs_split = []
    for split_i in pat_ids_split:
        imgs_split = imgs_split + [[img_k for img_k in img_names if re.sub("_.*", "", img_k) in split_i]]
    return imgs_split


def save_custom_cv_pat(custom_cv, save_dir):
    # custom_cv = custom_cv_pat(img_names=img_names, ran_seed=2457)
    # custom_cv = split_img_based_on_pat_split_from_custom_cv_pat(img_names=img_names, custom_cv=custom_cv)
    # save_dir = "/mnthome2/autoscoRA/autoscoRA_Pipeline/output/loc_and_segm_output/localization_cv_splits/" \
    #            "cv_splits_foot_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # save_dir = "/mnthome2/autoscoRA/autoscoRA_Pipeline/output/loc_and_segm_output/localization_cv_splits/" \
    #            "cv_splits_hand_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    for i in range(len(custom_cv)):

        val_imgs = custom_cv[i]
        train_imgs = [j for k in custom_cv for j in k if j not in val_imgs]

        # save dir for split i
        split_dir = save_dir + os.sep + "split_" + str(i)
        if not os.path.isdir(split_dir):
            os.mkdir(split_dir)

        # save train
        with open(split_dir + os.sep + "train.txt", 'w') as fp:
            fp.write('\n'.join(train_imgs))

        # save val
        with open(split_dir + os.sep + "val.txt", 'w') as fp:
            fp.write('\n'.join(val_imgs))


if __name__ == "__main__":
    rois = iop.import_rois_from_csv_to_dict(file=const.ROIS_PATH_GT_100, img_colname="img",
                                            corners=['RD', 'RP', 'UP', 'UD'])
    joints = iop.import_joints_from_csv_to_dict(file=const.JOINTS_PATH_GT_100, img_colname="img")
    finger_points = iop.joints_to_points_dict_for_patch_extraction(joints=joints,
                                                                   setup=iop.finger_joints_setup_hands)
    wrist_points = iop.joints_to_points_dict_for_patch_extraction(joints=joints,
                                                                  setup=iop.wrist_by_center_setup)
    # toe_points = iop.joints_to_points_dict_for_patch_extraction(joints=joints,
    #                                                             setup=iop.finger_joints_setup_feet)

    with open(const.JOINT_TO_ROIS_PARAMETERS_PATH) as json_file:
        params = json.load(json_file)

    for img in list(joints.keys())[1:10]:
        print(img)
        array = np.load(const.ARRAY_DIR + os.sep + img + ".npy")
        hand_ref = ref_size_from_bone_lengths(joints=joints[img], extremity='hand')
        pred_corners = {}
        gt_corners = {}
        pred_corners_modified = {}
        gt_corners_modified = {}
        pred_patches = {}
        gt_patches = {}
        pred_patches_modified = {}
        gt_patches_modified = {}

        if len(wrist_points) > 0:
            finger_points[img]['SWR'] = wrist_points[img]['SWR']

        for roi in finger_points[img].keys():
            print(roi)
            ref_to_length_DP = params[roi]['length_DP'][1]
            rel_shift_DP = -params[roi]['center_shift_DP'][1] / ref_to_length_DP

            pred_corners[roi] = finger_centers_to_corners(points=finger_points[img][roi],
                                                          ref_size=hand_ref,
                                                          ref_to_size_factor=ref_to_length_DP,  # float(1)
                                                          center_shift=rel_shift_DP)  # float(0)
            gt_corners[roi] = any_corners_to_rectangle_corners(corners=rois[img][roi], square=False,
                                                               orientation_type="degree",
                                                               returned_values="corners_dict",
                                                               )

            pred_corners_modified[roi] = roi_modifier(rectangle_measures=None, rectangle_corners=pred_corners[roi],
                                                      modification_order=['rotate',
                                                                          'shift_UR', 'shift_DP',
                                                                          'resize'],
                                                      rotate_degrees=20, shift_by_UR=0.1, shift_by_DP=0.1,
                                                      resize_UR_by=0.1, resize_DP_by=0.1,
                                                      rectangle=True, square=True)

            gt_corners_modified[roi] = roi_modifier(rectangle_measures=None, rectangle_corners=gt_corners[roi],
                                                    modification_order=['rotate',
                                                                        'shift_UR', 'shift_DP',
                                                                        'resize'],
                                                    rotate_degrees=20, shift_by_UR=0.1, shift_by_DP=0.1,
                                                    resize_UR_by=0.1, resize_DP_by=0.1,
                                                    rectangle=True, square=True)

            pred_patches[roi] = patch_cutter(img=array, rectangle_measures=None, rectangle_corners=pred_corners[roi],
                                             square=False, resize_patch=None, padd_patch=None, base_crop=int(3),
                                             plot=False, show_steps=True, print_log=False)

            gt_patches[roi] = patch_cutter(img=array, rectangle_measures=None,
                                           rectangle_corners=gt_corners[roi],
                                           square=False, resize_patch=None, padd_patch=None, base_crop=int(3),
                                           plot=False, show_steps=True, print_log=False)

            pred_patches_modified[roi] = patch_cutter(img=array, rectangle_measures=None,
                                                      rectangle_corners=pred_corners_modified[roi],
                                                      square=False, resize_patch=None, padd_patch=None, base_crop=int(3),
                                                      plot=False, show_steps=True, print_log=False)

            gt_patches_modified[roi] = patch_cutter(img=array, rectangle_measures=None,
                                                    rectangle_corners=gt_corners_modified[roi],
                                                    square=False, resize_patch=None, padd_patch=None, base_crop=int(3),
                                                    plot=False, show_steps=True, print_log=False)

        plot_joints_rois(img=array,
                         gt_joints=joints[img], pred_joints=None,
                         gt_corners=pred_corners,
                         pred_corners=pred_corners_modified,
                         annot_joints=False, annot_rois=True, gt_scatter_size=3, pred_scatter_size=3,
                         gt_scatter_color='cornflowerblue', pred_scatter_color='red',
                         gt_rectangle_fill=True, gt_rectangle_alpha=0.2, gt_rectangle_lw=1,
                         pred_rectangle_fill=True, pred_rectangle_alpha=0.2, pred_rectangle_lw=1,
                         display_legend=True, suppress_plot=False)

        plot_patches(gt_patches=pred_patches, pred_patches=pred_patches_modified)

        #####################
        # loc cv data split #
        #####################

        # img names
        H_joints_file = const.JOINTS_PATH_GT_100
        H_joints = iop.import_joints_from_csv_to_dict(file=H_joints_file, img_colname="img")
        H_img_names = list(H_joints.keys())

        F_joints_file = const.F_JOINTS_PATH_GT_100
        F_joints = iop.import_joints_from_csv_to_dict(file=F_joints_file, img_colname="img")
        F_img_names = list(F_joints.keys())

        # H custom_cv
        H_custom_cv = custom_cv_pat(img_names=H_img_names, ran_seed=2457)
        # F custom_cv based on pat level split for H
        F_custom_cv = split_img_based_on_pat_split_from_custom_cv_pat(img_names=F_img_names, custom_cv=H_custom_cv)

        # save custom_cv
        dtnow = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        H_save_dir = "/mnthome2/autoscoRA/autoscoRA_Pipeline/output/loc_and_segm_output/localization_cv_splits/" \
                     "cv_splits_hand_" + dtnow
        F_save_dir = "/mnthome2/autoscoRA/autoscoRA_Pipeline/output/loc_and_segm_output/localization_cv_splits/" \
                     "cv_splits_foot_" + dtnow

        save_custom_cv_pat(H_custom_cv, H_save_dir)
        save_custom_cv_pat(F_custom_cv, F_save_dir)
#
