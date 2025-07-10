# import SimpleITK as sitk
# import pydicom
# import numpy as np
import os
# from os.path import dirname, abspath, exists, splitext  # , basename
# from os.path import join as join_path
# import itertools
# import sys
import pandas as pd
import re
# import ntpath
import datetime
# from collections import Counter
# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
# from distutils.dir_util import copy_tree
import shutil
# from itertools import compress
# import tkinter as tk
# from tkinter import filedialog

print(datetime.datetime.now())

#############
# load data #
#############

dicom_server = "/project/autoscora/autoscoRA_images/sorted_by_categ_dicoms"
# old_images_server = "/project/autoscora/autoscoRA_images/changed_metadata_dicoms_copy"
output_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/output"
# output_folder = "/mnthome2/autoscoRA/autoscoRA_Preprocessing/output"
pat_df_manual_version = "pat_df_manual_2019-03-26_23-08-33_COMMENT.csv"  # "pat_df_manual_2019-03-31_13-27-40.csv"
one_note_issues_version = "one_note_issues_v2.csv"

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

# read in one_note_issues csv
one_note_issues_path = output_folder + os.sep + one_note_issues_version

one_note_issues_df = pd.read_csv(one_note_issues_path)

one_note_issues_dict_raw = one_note_issues_df.to_dict('list')  # turn to dict
one_note_issues_dict_nonan = {key: [element for element in value if not pd.isnull(element)]
                              for key, value in one_note_issues_dict_raw.items()}  # delete nans
required_action = {key: value[0] for key, value in one_note_issues_dict_nonan.items()}
one_note_issues_dict = {key: [str.strip(i) for i in value[1:]]
                        for key, value in one_note_issues_dict_nonan.items()}  # remove white spaces at start/end


####################################################
# deal with issues/prepare for manual intervention #
####################################################

# create backup copy of pat_df_manual
pat_df_manual_original = pat_df_manual.copy()

# ------ ring ------ #

# for known rings, delete the ring entry from OP
known_ring_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl']) if x in one_note_issues_dict['ring']]
missing_known_ring = [x for x in one_note_issues_dict['ring'] if x not in list(pat_df_manual['filename_new_dupl'])]
print("missing_known_ring: " + str(len(missing_known_ring)))

# unique_op_ring = list(pat_df_manual.loc[ring_idx, "operated_manual"].unique())
new_OP_raw = [re.sub("-PIP(III|IV)Prox", "", x) for x in pat_df_manual.loc[known_ring_idx, "operated_manual"]]
new_OP = ["OPNo" if op == 'OP' else op for op in new_OP_raw]

pat_df_manual.loc[known_ring_idx, "operated_manual"] = new_OP

# for unknown rings, find them via the ring entry for OP
# unknown_ring_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl']) if x in one_note_issues_dict['ring']]
unknown_ring_idx = [i for i, x in enumerate(pat_df_manual['operated_manual'])
                    if bool(re.search("-PIP(III|IV)Prox", x))]
unknown_ring_images = list(pat_df_manual.loc[unknown_ring_idx, 'sub_sub_dir'] +
                           os.sep +
                           pat_df_manual.loc[unknown_ring_idx, 'filename_new_dupl'] +
                           '.dcm')

# copy unknown ring images into a folder to go through again and check for rings
"""
# # create target directory
unknown_ring_dir = re.sub(os.path.split(dicom_server)[1] + "$", 
                                        "notes_issues_dicoms/H_dp_unknown_ring_dicoms", dicom_server)
if not os.path.isdir(unknown_ring_dir):
    os.mkdir(unknown_ring_dir)
else:
    print("dir already exists: " + unknown_ring_dir)

# # copy images to target directory
for unknown_ring_img in unknown_ring_images:
    if os.path.exists(unknown_ring_img):
        new_path = unknown_ring_dir + os.sep + os.path.basename(unknown_ring_img)
        shutil.copy(unknown_ring_img, new_path)
    else:
        print("not a file: " + unknown_ring_img)
"""

# for unknown rings, fix issues manually in pat_df_manual or here like above
# # DONE MANUALLY:
# # # 19036    538_20130917_H_L_dp_MTwo_0 PIPIVTrans
# # # 2993     538_20100217_H_L_dp_MTwo_0 PIPIVTrans
# # DONE AUTOMATICALLY:
actual_rings = ["724_20050301_H_B_dp_MOne_1", "30_20090206_H_R_dp_MTwo_0"]
actual_rings_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl']) if x in actual_rings]
unknown_new_OP_raw = [re.sub("-PIP(III|IV)Prox", "", x) for x in pat_df_manual.loc[actual_rings_idx, "operated_manual"]]
unknown_new_OP = ["OPNo" if op == 'OP' else op for op in unknown_new_OP_raw]

pat_df_manual.loc[actual_rings_idx, "operated_manual"] = unknown_new_OP


mcpiii_prox = ["50_20110314_H_R_dp_MTwo_1", "50_20060123_H_R_dp_MTwo_0",
               "50_20080703_H_R_dp_MOne_1", "50_20070711_H_R_dp_MTwo_1",
               "50_20070711_H_R_dp_MTwo_2", "50_20091021_H_R_dp_MTwo_0"]
mcpiii_prox_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl']) if x in mcpiii_prox]
mcpiii_prox_OP_raw = [re.sub("-MCPIIITrans-PIPIIIProx", "-MCPIIIProx", x)
                      for x in pat_df_manual.loc[mcpiii_prox_idx, "operated_manual"]]
mcpiii_prox_OP = ["OPNo" if op == 'OP' else op for op in mcpiii_prox_OP_raw]

pat_df_manual.loc[mcpiii_prox_idx, "operated_manual"] = mcpiii_prox_OP

# once checked manually: for known + unknown rings, add "ring" to the comment
all_rings_idx = known_ring_idx + actual_rings_idx
old_ring_comment = pat_df_manual.loc[all_rings_idx, "comment"]
new_ring_comment = ["ring" if x in ["none", "TBD"] else x + "__ring" for x in old_ring_comment]

pat_df_manual.loc[all_rings_idx, "comment"] = new_ring_comment
pat_df_manual.loc[all_rings_idx, "comment_status"] = "ComYes"


# ------ ulna head ------ #
# find all UlnaBoth entries
ulna_both_OP_idx = [i for i, x in enumerate(pat_df_manual['operated_manual'])
                    if bool(re.search("-UlnaBoth", x))]
ulna_both_OP_images = list(pat_df_manual.loc[ulna_both_OP_idx, 'sub_sub_dir'] +
                           os.sep +
                           pat_df_manual.loc[ulna_both_OP_idx, 'filename_new_dupl'] +
                           '.dcm')
# remove UlnaBoth entries
ulna_both_OP_raw = [re.sub("-UlnaBoth", "", x)
                    for x in pat_df_manual.loc[ulna_both_OP_idx, "operated_manual"]]
ulna_both_OP = ["OPNo" if op == 'OP' else op for op in ulna_both_OP_raw]

pat_df_manual.loc[ulna_both_OP_idx, "operated_manual"] = ulna_both_OP

# add ulna_head_forgotten to ulna_both index list
ulna_both_forgotten_OP_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
                              if x in one_note_issues_dict['ulna_head_forgotten']]
missing_ulna_both_forgotten = [x for x in one_note_issues_dict['ulna_head_forgotten']
                               if x not in list(pat_df_manual['filename_new_dupl'])]
print("missing_ulna_both_forgotten: " + str(len(missing_ulna_both_forgotten)))

ulna_both_idx = sorted(list(set(ulna_both_OP_idx + ulna_both_forgotten_OP_idx)))

# check that none of the combined entries have UlnaBoth in OP
ulna_both_combined_OP_idx = [i for i, x in enumerate(pat_df_manual.loc[ulna_both_OP_idx, 'operated_manual'])
                             if bool(re.search("-UlnaBoth", x))]
print(len(ulna_both_combined_OP_idx) == 0)

# create/add comment "ulna_head"
old_ulna_both_comment = pat_df_manual.loc[ulna_both_idx, "comment"]
new_ulna_both_comment = ["ulna_head" if x in ["none", "TBD"] else x + "__ulna_head" for x in old_ulna_both_comment]

pat_df_manual.loc[ulna_both_idx, "comment"] = new_ulna_both_comment
pat_df_manual.loc[ulna_both_idx, "comment_status"] = "ComYes"


# ------ irrelevant crop ------ #
irrelev_both_crop_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
                         if x in one_note_issues_dict['irrelev_both_crop']]
missing_irrelev_both_crop = [x for x in one_note_issues_dict['irrelev_both_crop']
                             if x not in list(pat_df_manual['filename_new_dupl'])]
print("missing_irrelev_both_crop: " + str(len(missing_irrelev_both_crop)))

irrelev_dist_crop_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
                         if x in one_note_issues_dict['irrelev_dist_crop']]
missing_irrelev_dist_crop = [x for x in one_note_issues_dict['irrelev_dist_crop']
                             if x not in list(pat_df_manual['filename_new_dupl'])]
print("missing_irrelev_dist_crop: " + str(len(missing_irrelev_dist_crop)))

irrelev_prox_crop_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
                         if x in one_note_issues_dict['irrelev_prox_crop']]
missing_oirrelev_prox_crop = [x for x in one_note_issues_dict['irrelev_prox_crop']
                              if x not in list(pat_df_manual['filename_new_dupl'])]
print("missing_oirrelev_prox_crop: " + str(len(missing_oirrelev_prox_crop)))

old_irrelev_both_crop_comment = pat_df_manual.loc[irrelev_both_crop_idx, "comment"]
new_irrelev_both_crop_comment = ["irrelev_dist_crop__irrelev_prox_crop" if x in ["none", "TBD"]
                                 else x + "__irrelev_dist_crop__irrelev_prox_crop"
                                 for x in old_irrelev_both_crop_comment]
pat_df_manual.loc[irrelev_both_crop_idx, "comment"] = new_irrelev_both_crop_comment
pat_df_manual.loc[irrelev_both_crop_idx, "comment_status"] = "ComYes"

old_irrelev_dist_crop_comment = pat_df_manual.loc[irrelev_dist_crop_idx, "comment"]
new_irrelev_dist_crop_comment = ["irrelev_dist_crop" if x in ["none", "TBD"] else x + "__irrelev_dist_crop"
                                 for x in old_irrelev_dist_crop_comment]
pat_df_manual.loc[irrelev_dist_crop_idx, "comment"] = new_irrelev_dist_crop_comment
pat_df_manual.loc[irrelev_dist_crop_idx, "comment_status"] = "ComYes"

old_irrelev_prox_crop_comment = pat_df_manual.loc[irrelev_prox_crop_idx, "comment"]
new_irrelev_prox_crop_comment = ["irrelev_prox_crop" if x in ["none", "TBD"] else x + "__irrelev_prox_crop"
                                 for x in old_irrelev_prox_crop_comment]
pat_df_manual.loc[irrelev_prox_crop_idx, "comment"] = new_irrelev_prox_crop_comment
pat_df_manual.loc[irrelev_prox_crop_idx, "comment_status"] = "ComYes"


# ------ irrelevant material ------ #
irrelev_material_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
                        if x in one_note_issues_dict['irrelev_material']]
missing_irrelev_material = [x for x in one_note_issues_dict['irrelev_material']
                            if x not in list(pat_df_manual['filename_new_dupl'])]
print("missing_irrelev_material: " + str(len(missing_irrelev_material)))

old_irrelev_material_comment = pat_df_manual.loc[irrelev_material_idx, "comment"]
new_irrelev_material_comment = ["irrelev_material" if x in ["none", "TBD"] else x + "__irrelev_material"
                                for x in old_irrelev_material_comment]
pat_df_manual.loc[irrelev_material_idx, "comment"] = new_irrelev_material_comment
pat_df_manual.loc[irrelev_material_idx, "comment_status"] = "ComYes"


# ------ overlapping structures ------ #
overlapping_structures_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
                              if x in one_note_issues_dict['overlapping_structures']]
missing_overlapping_structures = [x for x in one_note_issues_dict['overlapping_structures']
                                  if x not in list(pat_df_manual['filename_new_dupl'])]
print("missing_overlapping_structures: " + str(len(missing_overlapping_structures)))

old_overlapping_structures_comment = pat_df_manual.loc[overlapping_structures_idx, "comment"]
new_overlapping_structures_comment = ["overlapping_structures" if x in ["none", "TBD"]
                                      else x + "__overlapping_structures"
                                      for x in old_overlapping_structures_comment]
pat_df_manual.loc[overlapping_structures_idx, "comment"] = new_overlapping_structures_comment
pat_df_manual.loc[overlapping_structures_idx, "comment_status"] = "ComYes"


# ------ B overlaps ------ #
B_overlaps_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
                  if x in one_note_issues_dict['B_overlaps']]
missing_B_overlaps = [x for x in one_note_issues_dict['B_overlaps']
                      if x not in list(pat_df_manual['filename_new_dupl'])]
print("missing_B_overlaps: " + str(len(missing_B_overlaps)))

old_B_overlaps_comment = pat_df_manual.loc[B_overlaps_idx, "comment"]
new_B_overlaps_comment = ["B_overlaps" if x in ["none", "TBD"] else x + "__B_overlaps"
                          for x in old_B_overlaps_comment]
pat_df_manual.loc[B_overlaps_idx, "comment"] = new_B_overlaps_comment
pat_df_manual.loc[B_overlaps_idx, "comment_status"] = "ComYes"


# ------ rotation: direction noted ------ #
Rot_NE_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
              if x in one_note_issues_dict['Rot_NE']]
missing_Rot_NE = [x for x in one_note_issues_dict['Rot_NE']
                  if x not in list(pat_df_manual['filename_new_dupl'])]
print("missing_Rot_NE: " + str(len(missing_Rot_NE)))

Rot_E_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
             if x in one_note_issues_dict['Rot_E']]
missing_Rot_E = [x for x in one_note_issues_dict['Rot_E']
                 if x not in list(pat_df_manual['filename_new_dupl'])]
print("missing_Rot_E: " + str(len(missing_Rot_E)))

Rot_SE_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
              if x in one_note_issues_dict['Rot_SE']]
missing_Rot_SE = [x for x in one_note_issues_dict['Rot_SE']
                  if x not in list(pat_df_manual['filename_new_dupl'])]
print("missing_Rot_SE: " + str(len(missing_Rot_SE)))

Rot_S_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
             if x in one_note_issues_dict['Rot_S']]
missing_Rot_S = [x for x in one_note_issues_dict['Rot_S']
                 if x not in list(pat_df_manual['filename_new_dupl'])]
print("missing_Rot_S: " + str(len(missing_Rot_S)))

Rot_SW_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
              if x in one_note_issues_dict['Rot_SW']]
missing_Rot_SW = [x for x in one_note_issues_dict['Rot_SW']
                  if x not in list(pat_df_manual['filename_new_dupl'])]
print("missing_Rot_SW: " + str(len(missing_Rot_SW)))

Rot_W_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
             if x in one_note_issues_dict['Rot_W']]
missing_Rot_W = [x for x in one_note_issues_dict['Rot_W']
                 if x not in list(pat_df_manual['filename_new_dupl'])]
print("missing_Rot_W: " + str(len(missing_Rot_W)))

Rot_NW_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
              if x in one_note_issues_dict['Rot_NW']]
missing_Rot_NW = [x for x in one_note_issues_dict['Rot_NW']
                  if x not in list(pat_df_manual['filename_new_dupl'])]
print("missing_Rot_NW: " + str(len(missing_Rot_NW)))

# set RotYes for all of them
Rot_known = Rot_NE_idx + Rot_E_idx + Rot_SE_idx + Rot_S_idx +\
            Rot_SW_idx + Rot_W_idx + Rot_NW_idx

pat_df_manual.loc[Rot_known, "rotated_manual"] = "RotYes"

# put direction into comment
Rot_orientation = ["Rot_NE"]*len(Rot_NE_idx) + ["Rot_E"]*len(Rot_E_idx) +\
                  ["Rot_SE"]*len(Rot_SE_idx) + ["Rot_S"]*len(Rot_S_idx) +\
                  ["Rot_SW"]*len(Rot_SW_idx) + ["Rot_W"]*len(Rot_W_idx) +\
                  ["Rot_NW"]*len(Rot_NW_idx)
Rot_orientation_df = pd.DataFrame({"idx": Rot_known, "orientation": Rot_orientation})

old_Rot_orientation_comment = pat_df_manual.loc[Rot_orientation_df["idx"], "comment"]
new_Rot_orientation_comment = [Rot_row[1]["orientation"]
                               if pat_df_manual.loc[Rot_row[1]["idx"], "comment"] in ["none", "TBD"]
                               else pat_df_manual.loc[Rot_row[1]["idx"], "comment"] + "__" + Rot_row[1]["orientation"]
                               for Rot_row in Rot_orientation_df.iterrows()
                               ]
pat_df_manual.loc[Rot_orientation_df["idx"], "comment"] = new_Rot_orientation_comment
pat_df_manual.loc[Rot_orientation_df["idx"], "comment_status"] = "ComYes"


# ------ rotation: direction not noted ------ #
Rot_direction_unclear_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
                             if x in one_note_issues_dict['Rot_direction_unclear']]
missing_Rot_direction_unclear = [x for x in one_note_issues_dict['Rot_direction_unclear']
                                 if x not in list(pat_df_manual['filename_new_dupl'])]
print("missing_Rot_direction_unclear: " + str(len(missing_Rot_direction_unclear)))

# set RotYes for them
pat_df_manual.loc[Rot_direction_unclear_idx, "rotated_manual"] = "RotYes"


# find all RotYes entries with no noted direction
RotYes_unknown_idx = [i for i, x in enumerate(pat_df_manual['rotated_manual'])
                      if bool(re.search("RotYes", x)) and (i not in Rot_known)]
RotYes_unknown_images = list(pat_df_manual.loc[RotYes_unknown_idx, 'sub_sub_dir'] +
                             os.sep +
                             pat_df_manual.loc[RotYes_unknown_idx, 'filename_new_dupl'] +
                             '.dcm')

# set folder to which to copy RotYes images with unknown orientation to go through again and comment in direction
unknown_rot_dir = re.sub(os.path.split(dicom_server)[1] + "$",
                         "notes_issues_dicoms/H_dp_unknown_direction_dicoms",
                         dicom_server)


# ------ wrong category ------ #
wrong_categ_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
                   if x in one_note_issues_dict['wrong_categ']]
missing_wrong_categ = [x for x in one_note_issues_dict['wrong_categ']
                       if x not in list(pat_df_manual['filename_new_dupl'])]
print("missing_wrong_categ: " + str(len(missing_wrong_categ)))

# find NaN_manual images
NaN_idx = [i for i, x in enumerate(pat_df_manual['category_manual'])
           if bool(re.search("NaN", x))]
wrong_categ_NaN_idx = sorted(wrong_categ_idx + NaN_idx)

# images
wrong_categ_NaN_images = list(pat_df_manual.loc[wrong_categ_NaN_idx, 'sub_sub_dir'] +
                              os.sep +
                              pat_df_manual.loc[wrong_categ_NaN_idx, 'filename_new_dupl'] +
                              '.dcm')

# directory
wrong_categ_NaN_dir = re.sub(os.path.split(dicom_server)[1] + "$",
                             "notes_issues_dicoms/wrong_categ_NaN_dicoms",
                             dicom_server)


# ------ severe distortion ------ #
severe_distortion_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
                         if x in one_note_issues_dict['severe_distortion']]
missing_severe_distortion = [x for x in one_note_issues_dict['severe_distortion']
                             if x not in list(pat_df_manual['filename_new_dupl'])]
print("missing_severe_distortion: " + str(len(missing_severe_distortion)))

old_severe_distortion_comment = pat_df_manual.loc[severe_distortion_idx, "comment"]
new_severe_distortion_comment = ["severe_distortion" if x in ["none", "TBD"] else x + "__severe_distortion"
                                 for x in old_severe_distortion_comment]
pat_df_manual.loc[severe_distortion_idx, "comment"] = new_severe_distortion_comment
pat_df_manual.loc[severe_distortion_idx, "comment_status"] = "ComYes"

# images
severe_distortion_images = list(pat_df_manual.loc[severe_distortion_idx, 'sub_sub_dir'] +
                                os.sep +
                                pat_df_manual.loc[severe_distortion_idx, 'filename_new_dupl'] +
                                '.dcm')

# directory
severe_distortion_dir = re.sub(os.path.split(dicom_server)[1] + "$",
                               "notes_issues_dicoms/severe_distortion_dicoms",
                               dicom_server)


# ------ amputation ------ #
amput_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
             if x in one_note_issues_dict['amput']]
missing_amput = [x for x in one_note_issues_dict['amput']
                 if x not in list(pat_df_manual['filename_new_dupl'])]
print("missing_amput: " + str(len(missing_amput)))

old_amput_comment = pat_df_manual.loc[amput_idx, "comment"]
new_amput_comment = ["amput" if x in ["none", "TBD"] else x + "__amput"
                     for x in old_amput_comment]
pat_df_manual.loc[amput_idx, "comment"] = new_amput_comment
pat_df_manual.loc[amput_idx, "comment_status"] = "ComYes"

# images
amput_images = list(pat_df_manual.loc[amput_idx, 'sub_sub_dir'] +
                    os.sep +
                    pat_df_manual.loc[amput_idx, 'filename_new_dupl'] +
                    '.dcm')

# directory
amput_dir = re.sub(os.path.split(dicom_server)[1] + "$",
                   "notes_issues_dicoms/amput_dicoms",
                   dicom_server)


# ------ unclear BProb ------ #
unclear_B_prob_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
                      if x in one_note_issues_dict['unclear_B_prob']]
missing_unclear_B_prob = [x for x in one_note_issues_dict['unclear_B_prob']
                          if x not in list(pat_df_manual['filename_new_dupl'])]
print("missing_unclear_B_prob: " + str(len(missing_unclear_B_prob)))

# images
unclear_B_prob_images = list(pat_df_manual.loc[unclear_B_prob_idx, 'sub_sub_dir'] +
                             os.sep +
                             pat_df_manual.loc[unclear_B_prob_idx, 'filename_new_dupl'] +
                             '.dcm')

# directory
unclear_B_prob_dir = re.sub(os.path.split(dicom_server)[1] + "$",
                            "notes_issues_dicoms/unclear_B_prob_dicoms",
                            dicom_server)


# ------ unclear comment ------ #
unclear_comment_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
                       if x in one_note_issues_dict['unclear_comment']]
missing_unclear_comment = [x for x in one_note_issues_dict['unclear_comment']
                           if x not in list(pat_df_manual['filename_new_dupl'])]
print("missing_unclear_comment: " + str(len(missing_unclear_comment)))

# images
unclear_comment_images = list(pat_df_manual.loc[unclear_comment_idx, 'sub_sub_dir'] +
                              os.sep +
                              pat_df_manual.loc[unclear_comment_idx, 'filename_new_dupl'] +
                              '.dcm')

# directory
unclear_comment_dir = re.sub(os.path.split(dicom_server)[1] + "$",
                             "notes_issues_dicoms/unclear_comment_dicoms",
                             dicom_server)


# ------ view unclear ------ #
view_unclear_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
                    if x in one_note_issues_dict['view_unclear']]
missing_view_unclear = [x for x in one_note_issues_dict['view_unclear']
                        if x not in list(pat_df_manual['filename_new_dupl'])]
print("missing_view_unclear: " + str(len(missing_view_unclear)))

# images
view_unclear_images = list(pat_df_manual.loc[view_unclear_idx, 'sub_sub_dir'] +
                           os.sep +
                           pat_df_manual.loc[view_unclear_idx, 'filename_new_dupl'] +
                           '.dcm')

# directory
view_unclear_dir = re.sub(os.path.split(dicom_server)[1] + "$",
                          "notes_issues_dicoms/view_unclear_dicoms",
                          dicom_server)


# ------ OP unclear ------ #
OP_unclear_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
                  if x in one_note_issues_dict['OP_unclear']]
missing_OP_unclear = [x for x in one_note_issues_dict['OP_unclear']
                      if x not in list(pat_df_manual['filename_new_dupl'])]
print("missing_OP_unclear: " + str(len(missing_OP_unclear)))

# images
OP_unclear_images = list(pat_df_manual.loc[OP_unclear_idx, 'sub_sub_dir'] +
                         os.sep +
                         pat_df_manual.loc[OP_unclear_idx, 'filename_new_dupl'] +
                         '.dcm')

# directory
OP_unclear_dir = re.sub(os.path.split(dicom_server)[1] + "$",
                        "notes_issues_dicoms/OP_unclear_dicoms",
                        dicom_server)


# ------ missing and skipped splitpoints ------ #
splitpoint_missing_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
                          if x in one_note_issues_dict['splitpoint_missing']]
missing_splitpoint_missing = [x for x in one_note_issues_dict['splitpoint_missing']
                              if x not in list(pat_df_manual['filename_new_dupl'])]
print("missing_splitpoint_missing: " + str(len(missing_splitpoint_missing)))

# find skipped splitpoint images
splitpoint_skipped_idx = [i for i, x in enumerate(pat_df_manual['splitpoint_manual'])
                          if bool(re.search("skipped", x))]
splitpoint_missing_skipped_idx = sorted(splitpoint_missing_idx + splitpoint_skipped_idx)

# images
splitpoint_missing_skipped_images = list(pat_df_manual.loc[splitpoint_missing_skipped_idx, 'sub_sub_dir'] +
                                         os.sep +
                                         pat_df_manual.loc[splitpoint_missing_skipped_idx, 'filename_new_dupl'] +
                                         '.dcm')

# directory
splitpoint_missing_skipped_dir = re.sub(os.path.split(dicom_server)[1] + "$",
                                        "notes_issues_dicoms/splitpoint_missing_skipped_dicoms",
                                        dicom_server)


# ------ H_dp_lat ------ #
H_dp_lat_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
                if x in one_note_issues_dict['H_dp_lat']]
missing_H_dp_lat = [x for x in one_note_issues_dict['H_dp_lat']
                    if x not in list(pat_df_manual['filename_new_dupl'])]
print("missing_H_dp_lat: " + str(len(missing_H_dp_lat)))

# find duplicated images only differing in _1, _2, etc. at end from the H_dp_lat image
H_dp_lat_duplicates_idx_dict = {H_dp_lat_i:
                                [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
                                 if bool(re.search(re.sub("_[0-9]$", "",
                                                          pat_df_manual.loc[H_dp_lat_i, 'filename_new_dupl']),
                                                   x))
                                 ]
                                for H_dp_lat_i in H_dp_lat_idx
                                }
# find those images that have same manual pat_id, study_date, bodypart_manual, laterality_manual
H_dp_lat_manual_duplicates_idx_dict = {H_dp_lat_i:
                                       [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
                                        if "_".join([str(i) for i in pat_df_manual.loc[i, ["pat_id", "study_date",
                                                                                           "bodypart_manual",
                                                                                           "laterality_manual"]
                                                                                       ].tolist()]) ==
                                        "_".join([str(i) for i in pat_df_manual.loc[H_dp_lat_i, ["pat_id", "study_date",
                                                                                                 "bodypart_manual",
                                                                                                 "laterality_manual"]
                                                                                    ].tolist()])
                                        ]
                                       for H_dp_lat_i in H_dp_lat_idx
                                       }


H_dp_lat_duplicates_idx = [idx for idx_list in H_dp_lat_duplicates_idx_dict.values() for idx in idx_list]
H_dp_lat_manual_duplicates_idx = [idx for idx_list in H_dp_lat_manual_duplicates_idx_dict.values() for idx in idx_list]

H_dp_lat_and_duplicates_idx = sorted(H_dp_lat_idx + H_dp_lat_duplicates_idx + H_dp_lat_manual_duplicates_idx)
deduplicated_H_dp_lat_and_duplicates_idx = [x for i, x in enumerate(H_dp_lat_and_duplicates_idx)
                                            if H_dp_lat_and_duplicates_idx.index(x) == i]

# added_H_dp_lat_duplicates_list_ind = [i not in H_dp_lat_idx
#                                       for i in deduplicated_H_dp_lat_and_duplicates_idx
#                                       ]
# H_dp_lat_duplicates_list = list(compress(deduplicated_H_dp_lat_and_duplicates_idx,
#                                          added_H_dp_lat_duplicates_list_ind))

# indices
H_dp_lat_plus_idx = deduplicated_H_dp_lat_and_duplicates_idx

# images
H_dp_lat_plus_images = list(pat_df_manual.loc[H_dp_lat_plus_idx, 'sub_sub_dir'] +
                            os.sep +
                            pat_df_manual.loc[H_dp_lat_plus_idx, 'filename_new_dupl'] +
                            '.dcm')

# directory
H_dp_lat_plus_dir = re.sub(os.path.split(dicom_server)[1] + "$",
                           "notes_issues_dicoms/H_dp_lat_dicoms",
                           dicom_server)


# ------ F_dp_lat ------ #
F_dp_lat_idx = [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
                if x in one_note_issues_dict['F_dp_lat']]
missing_F_dp_lat = [x for x in one_note_issues_dict['F_dp_lat']
                    if x not in list(pat_df_manual['filename_new_dupl'])]
print("missing_F_dp_lat: " + str(len(missing_F_dp_lat)))

# find duplicated images only differing in _1, _2, etc. at end from the F_dp_lat image
F_dp_lat_duplicates_idx_dict = {F_dp_lat_i:
                                [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
                                 if bool(re.search(re.sub("_[0-9]$", "",
                                                          pat_df_manual.loc[F_dp_lat_i, 'filename_new_dupl']),
                                                   x))
                                 ]
                                for F_dp_lat_i in F_dp_lat_idx
                                }
# find those images that have same manual pat_id, study_date, bodypart_manual, laterality_manual
F_dp_lat_manual_duplicates_idx_dict = {F_dp_lat_i:
                                       [i for i, x in enumerate(pat_df_manual['filename_new_dupl'])
                                        if "_".join([str(i) for i in pat_df_manual.loc[i, ["pat_id", "study_date",
                                                                                           "bodypart_manual",
                                                                                           "laterality_manual"]
                                                                                       ].tolist()]) ==
                                        "_".join([str(i) for i in pat_df_manual.loc[F_dp_lat_i, ["pat_id", "study_date",
                                                                                                 "bodypart_manual",
                                                                                                 "laterality_manual"]
                                                                                    ].tolist()])
                                        ]
                                       for F_dp_lat_i in F_dp_lat_idx
                                       }


F_dp_lat_duplicates_idx = [idx for idx_list in F_dp_lat_duplicates_idx_dict.values() for idx in idx_list]
F_dp_lat_manual_duplicates_idx = [idx for idx_list in F_dp_lat_manual_duplicates_idx_dict.values() for idx in idx_list]

F_dp_lat_and_duplicates_idx = sorted(F_dp_lat_idx + F_dp_lat_duplicates_idx + F_dp_lat_manual_duplicates_idx)
deduplicated_F_dp_lat_and_duplicates_idx = [x for i, x in enumerate(F_dp_lat_and_duplicates_idx)
                                            if F_dp_lat_and_duplicates_idx.index(x) == i]

# added_F_dp_lat_duplicates_list_ind = [i not in F_dp_lat_idx
#                                       for i in deduplicated_F_dp_lat_and_duplicates_idx
#                                       ]
# F_dp_lat_duplicates_list = list(compress(deduplicated_F_dp_lat_and_duplicates_idx,
#                                          added_F_dp_lat_duplicates_list_ind))

# indices
F_dp_lat_plus_idx = deduplicated_F_dp_lat_and_duplicates_idx

# images
F_dp_lat_plus_images = list(pat_df_manual.loc[F_dp_lat_plus_idx, 'sub_sub_dir'] +
                            os.sep +
                            pat_df_manual.loc[F_dp_lat_plus_idx, 'filename_new_dupl'] +
                            '.dcm')

# directory
F_dp_lat_plus_dir = re.sub(os.path.split(dicom_server)[1] + "$",
                           "notes_issues_dicoms/F_dp_lat_dicoms",
                           dicom_server)


# ------ common final path for those that I have to go through again ------ #

# NOTE: if images in two of the dirs != prob, b/c after gone through one of the dirs, that images has no TBD anymore
# and won't be displayed again

# set >= 1 TBD for those I have to go through again
go_through_again_idx = {unknown_rot_dir: {"idx": RotYes_unknown_idx, "img": RotYes_unknown_images},
                        wrong_categ_NaN_dir: {"idx": wrong_categ_NaN_idx, "img": wrong_categ_NaN_images},
                        severe_distortion_dir: {"idx": severe_distortion_idx, "img": severe_distortion_images},
                        amput_dir: {"idx": amput_idx, "img": amput_images},
                        unclear_B_prob_dir: {"idx": unclear_B_prob_idx, "img": unclear_B_prob_images},
                        unclear_comment_dir: {"idx": unclear_comment_idx, "img": unclear_comment_images},
                        view_unclear_dir: {"idx": view_unclear_idx, "img": view_unclear_images},
                        OP_unclear_dir: {"idx": OP_unclear_idx, "img": OP_unclear_images},
                        splitpoint_missing_skipped_dir: {"idx": splitpoint_missing_skipped_idx,
                                                         "img": splitpoint_missing_skipped_images},
                        H_dp_lat_plus_dir: {"idx": H_dp_lat_plus_idx, "img": H_dp_lat_plus_images},
                        F_dp_lat_plus_dir: {"idx": F_dp_lat_plus_idx, "img": F_dp_lat_plus_images}
                        }
for directory, files in go_through_again_idx.items():
    pat_df_manual.loc[files["idx"], "splitpoint_manual"] = "TBD"

# create target directories
for target_dir in go_through_again_idx.keys():
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    else:
        print("dir already exists: " + target_dir)

# copy images to target directories
for directory, files in go_through_again_idx.items():
    for img in files["img"]:
        if os.path.exists(img):
            new_path = directory + os.sep + os.path.basename(str(img))
            shutil.copy(str(img), new_path)
        else:
            print("not a file: " + str(img))


# ------ remove duplicated comment parts ------ #
deduplicated_comments = []
for comment_i in pat_df_manual.loc[:, "comment"]:
    comment_list_i = comment_i.split("__")
    deduplicated_comment_list_i = [x for i, x in enumerate(comment_list_i)
                                   if comment_list_i.index(x) == i]
    deduplicated_comment_i = '__'.join(deduplicated_comment_list_i)
    deduplicated_comments = deduplicated_comments + [deduplicated_comment_i]

# all(deduplicated_comments == pat_df_manual.loc[:, "comment"])
pat_df_manual.loc[:, "comment"] = deduplicated_comments


# ------ save pat_df to csv ------ #
current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
new_pat_df_filename = output_folder + os.sep + os.path.splitext(pat_df_manual_version)[0] + \
                      "_Step6_" + current_datetime + ".csv"
pat_df_manual.to_csv(new_pat_df_filename)
