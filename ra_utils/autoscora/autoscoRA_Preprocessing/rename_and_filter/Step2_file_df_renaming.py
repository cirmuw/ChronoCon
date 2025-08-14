# import SimpleITK as sitk
import numpy as np
import os
# import itertools
# import sys
import pandas as pd
# import re
# import ntpath
import datetime
from collections import Counter


print(datetime.datetime.now())

#################
# load the data #
#################

dicom_server = "/project/autoscora/autoscoRA_images/autoscoRA_original_images"
output_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/output"

emptystr = True

# sample data
# output_folder = "/mnthome/autoscoRA/autoscoRA_Preprocessing/output/test"
# output_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/output/test"
# dicom_server = "/mnthome/autoscoRA/autoscoRA_Preprocessing/sample_xrays"
# dicom_server = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/sample_xrays"

# read metadata dataframe
print("START: reading metadata dataframe...")
pat_df = pd.read_pickle(output_folder + os.sep + "metadata_df.pkl")
# df = pd.read_csv(output_folder + os.sep + "metadata_df.csv")
print("DONE: reading metadata dataframe")

# read count of files per unique patient+study_date combination -->
# can use this as well to identify double-hand/foot images!
print("START: reading nr of images per id+date combo...")
im_per_iddate = pd.read_pickle(output_folder + os.sep + "im_per_iddate.pkl")
# im_per_iddate = pd.read_csv(output_folder + os.sep + "im_per_iddate.csv")
print("DONE: reading nr of images per id+date combo")

# count number of unique patient+study_date combinations
iddate_count = len(im_per_iddate)
print("OUTPUT: number of id+date combinations:", iddate_count)

# read dictionary of unique metadata entries
print("START: reading df of unique metadata entries...")
pat_dict_unique_df = pd.read_pickle(output_folder + os.sep + "metadata_unique_df.pkl")
# pat_dict_unique_df = pd.read_csv(output_folder + os.sep + "metadata_unique_df.csv")
print("DONE: reading df of unique metadata entries")

#####################################
# manually created list of criteria #
#####################################

# ---- bodypart ---- #
bodypart_foot = ["FOOT", "ANKLE", "LOW_EXM"]
bodypart_hand = ["HAND", "UP_EXM"]
bodypart_other = ["CSPINE", "Vertebral Column", "KNEE", "NECK", "PELVIS", "ELBOW"]
bodypart_nonNA = bodypart_foot + bodypart_hand + bodypart_other

# ---- study_description ---- #
study_description_hand = ['KR Hand',  'Hand', 'Handgelenk', 'KR Handgelenk', 'Hand Gelenke',
                          'Hand dp Zitherstellung', 'Hand Extremit?ten', 'Hand dp Zitherstellung Extremit?ten'
                          ]
study_description_foot = ['Vorfuss',  'KR Fu?', 'Fu?',
                          'Fu?gelenk', 'Fu?gelenk Gelenke',  # those two actually Sprunggelenk?
                          'Vorfuss Extremit?ten', 'Fu? Extremit?ten'
                          ]
study_description_bodypart_nonNA = study_description_hand + study_description_foot

# ---- laterality ---- #
laterality_R = ['R']
laterality_L = ['L']
laterality_nonNA = laterality_R + laterality_L

# ---- view_position ---- #
view_position_dp = ['PA', 'AP']
view_position_lat = ['LL', 'RL', 'LLO', 'RLO']
view_position_R = ['RL', 'RLO']
view_position_L = ['LL', 'LLO']
view_position_nonNA = view_position_dp + view_position_lat
view_position_laterality_nonNA = view_position_R + view_position_L

# ---- series_description ---- #
series_description_hand = ['Hand dp L', 'Hand lat L', 'Hand dp R', 'Hand lat R', 'Hand pa',
                           'Hand lat', 'Hand links seitl.', 'Hand links ap', 'Hand rechts ap',
                           'Hand rechts seitl.', 'HAND ZITHER VGL', 'HAND DV LI', 'HAND DV RE',
                           'HAND DP RE', 'HAND DP LI', 'Hand Zitherstellung', 'HAND ZITHER BDS/X',
                           'HAND DP LI/X', 'HAND DP RE/X', 'HAND ZITHER RE/X *',
                           'T106 Hand schr?g links', 'T106 Hand a.p. links',
                           'T106 Hand a.p. rechts', 'T106 Hand schr?g rechts', 'HAND ZITHER RE',
                           'HANDGELENK STL VGL', 'HANDGELENK AP LI/X', 'HANDGELENK STL VGL/X',
                           'HANDGELENK AP RE/X', 'HAND ZITHER LI', 'HAND ZITHER BDS/X *',
                           'HAND DP LI/X *', 'HAND DP RE/X *', 'T106 Hand seitlich links',
                           'HANDGELENK AP VGL', 'HANDGELENK AP LI', 'HANDGELENK AP RE',
                           'Handgelenk dp R', 'Handgelenk lat R', 'Handgelenk dp L',
                           'Handgelenk lat L', 'Handgelenk rechts ap', 'Handgelenk rechts seitl.',
                           'HAND ZITHER LI/X', 'Handgelenk links ap', 'HAND ZITHER RE/X',
                           'HANDGELENK STL RE/X', 'Handgelenk links seitl.',
                           'Unterarm / Handgelenk pa L', 'Finger ap R'
                           ]
series_description_foot = ['Fuss dp L', 'Fuss schr?g L', 'Fuss dp R', 'Fuss schr?g R',
                           'Sprunggelenk lat', 'Sprunggelenk ap', 'Fu? schr?g', 'Fu? ap',
                           'Vorfuss rechts ap', 'Vorfuss rechts schraeg', 'Vorfuss links ap',
                           'Vorfuss links schraeg', 'Sprunggelenk rechts seitl.', 'Sprunggelenk rechts ap',
                           'Sprunggelenk links ap', 'Sprunggelenk links seitl.', 'VORFUSS schraeg VGL',
                           'VORFUSS DP VGL', 'VORFUSS SCHRAEG RE', 'Fu? dp L', 'Fu? dp R', 'Fu? schr?g R',
                           'Fu? schr?g L', 'VORFUSS SCHRAEG LI', 'Zehen rechts ap', 'T109 Vorfu? d.p. rechts',
                           'T109 Vorfu? d.p. links', 'T109 Vorfu? schr?g links', 'T109 Vorfu? schr?g rechts',
                           'Fuss lat R', 'Fu? lat', 'FUSS AP LI/T', 'FUSS STL LI/T *', 'FUSS AP RE/T',
                           'FUSS STL RE/T *', 'FUSS AP VGL/T *', 'Fu? lat L', 'Fu? lat R',
                           'VORFUSS SCHRAEG LI/T', 'VORFUSS SCHRAEG RE/T', 'VORFUSS DP VGL/T',
                           'VORFUSS DP VGL/T *', 'VORFUSS SCHRAEG RE/T *', 'VORFUSS SCHRAEG LI/T *',
                           'Fuss lat L', 'SPRUNGGELENK AP RE', 'SPRUNGGELENK STL LI', 'SPRUNGGELENK STL RE',
                           'FUSS STL LI', 'FUSS STL RE', 'VORFUSS DP LI STD', 'VORFUSS DP RE STD',
                           'Fu? ap R', 'Fu? ap L', 'FUSS AP LI', 'FUSS AP RE', 'VORFUSS DP LI',
                           'VORFUSS DP RE', 'VORFUSS DP LI/T', 'FUSS STL RE STD', 'FUSS STL LI STD',
                           'FUSS STL LI/T', 'FUSS AP VGL/T', 'FUSS STL RE/T', 'FUSS STL RE STD/W *',
                           'FUSS AP VGL', 'VF SESAMBEINE RE',
                           'VORFUSS DP VGL/W *', 'VORFUSS SCHRAEG LI/W *', 'VORFUSS SCHRAEG RE/W *'
                           ]
series_description_dp = ['Fuss dp L', 'Fuss dp R', 'Sprunggelenk ap', 'Fu? ap', 'Vorfuss rechts ap',
                         'Vorfuss links ap', 'Sprunggelenk rechts ap', 'Sprunggelenk links ap',
                         'VORFUSS DP VGL', 'Fu? dp L', 'Fu? dp R', 'Zehen rechts ap',
                         'T109 Vorfu? d.p. rechts', 'T109 Vorfu? d.p. links', 'FUSS AP LI/T',
                         'FUSS AP RE/T', 'FUSS AP VGL/T *', 'VORFUSS DP VGL/T', 'VORFUSS DP VGL/T *',
                         'SPRUNGGELENK AP RE', 'VORFUSS DP LI STD', 'VORFUSS DP RE STD', 'Fu? ap R',
                         'Fu? ap L', 'FUSS AP LI', 'FUSS AP RE', 'VORFUSS DP LI', 'VORFUSS DP RE',
                         'VORFUSS DP LI/T', 'FUSS AP VGL/T', 'FUSS AP VGL', 'VORFUSS DP VGL/W *',
                         'Hand dp L', 'Hand dp R', 'Hand pa', 'Hand links ap', 'Hand rechts ap',
                         'HAND DV LI', 'HAND DV RE', 'HAND DP RE', 'HAND DP LI', 'HAND DP LI/X',
                         'HAND DP RE/X', 'T106 Hand a.p. links', 'T106 Hand a.p. rechts',
                         'HANDGELENK AP LI/X', 'HANDGELENK AP RE/X', 'HAND DP LI/X *',
                         'HAND DP RE/X *', 'HANDGELENK AP VGL', 'HANDGELENK AP LI', 'HANDGELENK AP RE',
                         'Handgelenk dp R', 'Handgelenk dp L', 'Handgelenk rechts ap',
                         'Handgelenk links ap', 'Unterarm / Handgelenk pa L', 'Finger ap R',
                         'KNIE AP LI/W *', 'Sprunggelenk ap R', 'Sprunggelenk ap L'
                         ]
series_description_lat = ['Fuss schr?g L', 'Fuss schr?g R', 'Sprunggelenk lat', 'Fu? schr?g',
                          'Vorfuss rechts schraeg', 'Vorfuss links schraeg',
                          'Sprunggelenk rechts seitl.', 'Sprunggelenk links seitl.',
                          'VORFUSS schraeg VGL', 'VORFUSS SCHRAEG RE', 'Fu? schr?g R', 'Fu? schr?g L',
                          'VORFUSS SCHRAEG LI', 'T109 Vorfu? schr?g links', 'T109 Vorfu? schr?g rechts',
                          'Fuss lat R', 'Fu? lat', 'FUSS STL LI/T *', 'FUSS STL RE/T *', 'Fu? lat L',
                          'Fu? lat R', 'VORFUSS SCHRAEG LI/T', 'VORFUSS SCHRAEG RE/T',
                          'VORFUSS SCHRAEG RE/T *', 'VORFUSS SCHRAEG LI/T *', 'Fuss lat L',
                          'SPRUNGGELENK STL LI', 'SPRUNGGELENK STL RE', 'FUSS STL LI', 'FUSS STL RE',
                          'FUSS STL RE STD', 'FUSS STL LI STD', 'FUSS STL LI/T', 'FUSS STL RE/T',
                          'FUSS STL RE STD/W *', 'VORFUSS SCHRAEG LI/W *', 'VORFUSS SCHRAEG RE/W *',
                          'Hand lat L', 'Hand lat R', 'Hand lat', 'Hand links seitl.',
                          'Hand rechts seitl.', 'HAND ZITHER VGL', 'Hand Zitherstellung',
                          'HAND ZITHER BDS/X', 'HAND ZITHER RE/X *', 'T106 Hand schr?g links',
                          'T106 Hand schr?g rechts', 'HAND ZITHER RE', 'HANDGELENK STL VGL',
                          'HANDGELENK STL VGL/X', 'HAND ZITHER LI', 'HAND ZITHER BDS/X *',
                          'T106 Hand seitlich links', 'Handgelenk lat R', 'Handgelenk lat L',
                          'Handgelenk rechts seitl.', 'HAND ZITHER LI/X', 'HAND ZITHER RE/X',
                          'HANDGELENK STL RE/X', 'Handgelenk links seitl.', 'HWS lat', 'KNIE STL LI/T *',
                          'Fersenbein lat L', 'Fersenbein lat R', 'Sprunggelenk lat R', 'Sprunggelenk lat L'
                          ]
series_description_R = ['Fuss schr?g R', 'Vorfuss rechts schraeg', 'Sprunggelenk rechts seitl.',
                        'VORFUSS SCHRAEG RE', 'Fu? schr?g R', 'T109 Vorfu? schr?g rechts',
                        'Fuss lat R', 'FUSS STL RE/T *', 'Fu? lat R', 'VORFUSS SCHRAEG RE/T',
                        'VORFUSS SCHRAEG RE/T *', 'SPRUNGGELENK STL RE', 'FUSS STL RE',
                        'FUSS STL RE STD', 'FUSS STL RE/T', 'FUSS STL RE STD/W *',
                        'VORFUSS SCHRAEG RE/W *', 'Hand lat R', 'Hand rechts seitl.',
                        'HAND ZITHER RE/X *', 'T106 Hand schr?g rechts', 'HAND ZITHER RE',
                        'Handgelenk lat R', 'Handgelenk rechts seitl.', 'HAND ZITHER RE/X',
                        'HANDGELENK STL RE/X', 'Fersenbein lat R', 'Sprunggelenk lat R',
                        'Fuss dp R', 'Vorfuss rechts ap', 'Sprunggelenk rechts ap', 'Fu? dp R',
                        'Zehen rechts ap', 'T109 Vorfu? d.p. rechts', 'FUSS AP RE/T',
                        'SPRUNGGELENK AP RE', 'VORFUSS DP RE STD', 'Fu? ap R', 'FUSS AP RE',
                        'VORFUSS DP RE', 'Hand dp R', 'Hand rechts ap', 'HAND DV RE',
                        'HAND DP RE', 'HAND DP RE/X', 'T106 Hand a.p. rechts',
                        'HANDGELENK AP RE/X', 'HAND DP RE/X *', 'HANDGELENK AP RE',
                        'Handgelenk dp R', 'Handgelenk rechts ap', 'Finger ap R',
                        'Sprunggelenk ap R', 'VF SESAMBEINE RE'
                        ]
series_description_L = ['Fuss schr?g L', 'Vorfuss links schraeg', 'Sprunggelenk links seitl.',
                        'Fu? schr?g L', 'VORFUSS SCHRAEG LI', 'T109 Vorfu? schr?g links',
                        'FUSS STL LI/T *', 'Fu? lat L', 'VORFUSS SCHRAEG LI/T',
                        'VORFUSS SCHRAEG LI/T *', 'Fuss lat L', 'SPRUNGGELENK STL LI',
                        'FUSS STL LI', 'FUSS STL LI STD', 'FUSS STL LI/T', 'VORFUSS SCHRAEG LI/W *',
                        'Hand lat L', 'Hand links seitl.', 'T106 Hand schr?g links',
                        'HAND ZITHER LI', 'T106 Hand seitlich links', 'Handgelenk lat L',
                        'HAND ZITHER LI/X', 'Handgelenk links seitl.', 'KNIE STL LI/T *',
                        'Fersenbein lat L', 'Sprunggelenk lat L', 'Fuss dp L', 'Vorfuss links ap',
                        'Sprunggelenk links ap', 'Fu? dp L', 'T109 Vorfu? d.p. links', 'FUSS AP LI/T',
                        'VORFUSS DP LI STD', 'Fu? ap L', 'FUSS AP LI', 'VORFUSS DP LI',
                        'VORFUSS DP LI/T', 'Hand dp L', 'Hand links ap', 'HAND DV LI', 'HAND DP LI',
                        'HAND DP LI/X', 'T106 Hand a.p. links', 'HANDGELENK AP LI/X',
                        'HAND DP LI/X *', 'HANDGELENK AP LI', 'Handgelenk dp L',
                        'Handgelenk links ap', 'Unterarm / Handgelenk pa L', 'KNIE AP LI/W *',
                        'Sprunggelenk ap L', 'Fersenbein axial L'
                        ]
series_description_B = ['VORFUSS schraeg VGL', 'HAND ZITHER VGL', 'HAND ZITHER BDS/X',
                        'HANDGELENK STL VGL', 'HANDGELENK STL VGL/X', 'HAND ZITHER BDS/X *',
                        'VORFUSS DP VGL', 'FUSS AP VGL/T *', 'VORFUSS DP VGL/T',
                        'VORFUSS DP VGL/T *', 'FUSS AP VGL/T', 'FUSS AP VGL',
                        'VORFUSS DP VGL/W *', 'HANDGELENK AP VGL'
                        ]
series_description_bodypart_nonNA = series_description_hand + series_description_foot
series_description_view_position_nonNA = series_description_dp + series_description_lat
series_description_laterality_nonNA = series_description_R + series_description_L + series_description_B

# ---- photometric_interpretation ---- #
photometric_interpretation_MOne = ['MONOCHROME1']
photometric_interpretation_MTwo = ['MONOCHROME2']
photometric_interpretation_MOther = []
photometric_interpretation_nonNA = photometric_interpretation_MOne +\
                                   photometric_interpretation_MTwo +\
                                   photometric_interpretation_MOther


##################################
# algorithm to categorize images #
##################################

# ---- 1. body part ---- #
pat_df['bodypart_new'] = [np.nan]*pat_df.shape[0]
bodypart_NA = [(k not in bodypart_nonNA) for k in pat_df['body_part_examined']]

# foot
pat_df.loc[[(i in bodypart_foot) for i in pat_df['body_part_examined']], 'bodypart_new'] = 'F'  # bodypart

studydescription_foot = [(i in study_description_foot) for i in pat_df['study_description']]  # study
studydescription_foot = [a and b for a, b in zip(bodypart_NA, studydescription_foot)]
seriesdescription_foot = [(i in series_description_foot) for i in pat_df['series_description']]  # series
seriesdescription_foot = [a and b for a, b in zip(bodypart_NA, seriesdescription_foot)]

pat_df.loc[[a or b for a, b in zip(studydescription_foot, seriesdescription_foot)], 'bodypart_new'] = 'F'

# hand
pat_df.loc[[(i in bodypart_hand) for i in pat_df['body_part_examined']], 'bodypart_new'] = 'H'  # bodypart

studydescription_hand = [(i in study_description_hand) for i in pat_df['study_description']]  # study
studydescription_hand = [a and b for a, b in zip(bodypart_NA, studydescription_hand)]
seriesdescription_hand = [(i in series_description_hand) for i in pat_df['series_description']]  # series
seriesdescription_hand = [a and b for a, b in zip(bodypart_NA, seriesdescription_hand)]

pat_df.loc[[a or b for a, b in zip(studydescription_hand, seriesdescription_hand)], 'bodypart_new'] = 'H'

# other
pat_df.loc[[(i in bodypart_other) for i in pat_df['body_part_examined']], 'bodypart_new'] = 'O'

# checks
# check that no na's were part of bodypart_nonNA
set(list(pat_df.loc[pat_df['bodypart_new'].isnull(),
                    'body_part_examined'].index)).issubset(list(pat_df.loc[[(i not in bodypart_nonNA)
                                                                            for i in pat_df['body_part_examined']],
                                                                           'body_part_examined'].index))
# check that all emptystr were put in bodypart_NA at first
set(list(pat_df.loc[pat_df['body_part_examined'] == 'emptystr',
                    'body_part_examined'].index)).issubset(list(pat_df.loc[[(i not in bodypart_nonNA)
                                                                            for i in pat_df['body_part_examined']],
                                                                           'body_part_examined'].index))

# ---- 2. laterality ---- #

pat_df['laterality_new'] = [np.nan]*pat_df.shape[0]
laterality_NA = [(k not in laterality_nonNA) for k in pat_df['laterality']]

# right
pat_df.loc[[(i in laterality_R) for i in pat_df['laterality']], 'laterality_new'] = 'R'  # laterality

viewposition_R = [(i in view_position_R) for i in pat_df['view_position']]  # view position
viewposition_R = [a and b for a, b in zip(laterality_NA, viewposition_R)]
seriesdescription_R = [(i in series_description_R) for i in pat_df['series_description']]  # series
seriesdescription_R = [a and b for a, b in zip(laterality_NA, seriesdescription_R)]

pat_df.loc[[a or b for a, b in zip(viewposition_R, seriesdescription_R)], 'laterality_new'] = 'R'

# left
pat_df.loc[[(i in laterality_L) for i in pat_df['laterality']], 'laterality_new'] = 'L'  # laterality

viewposition_L = [(i in view_position_L) for i in pat_df['view_position']]  # view position
viewposition_L = [a and b for a, b in zip(laterality_NA, viewposition_L)]
seriesdescription_L = [(i in series_description_L) for i in pat_df['series_description']]  # series
seriesdescription_L = [a and b for a, b in zip(laterality_NA, seriesdescription_L)]

pat_df.loc[[a or b for a, b in zip(viewposition_L, seriesdescription_L)], 'laterality_new'] = 'L'

# both
seriesdescription_B = [(i in series_description_B) for i in pat_df['series_description']]  # series
seriesdescription_B = [a and b for a, b in zip(laterality_NA, seriesdescription_B)]

pat_df.loc[seriesdescription_B, 'laterality_new'] = 'B'

# checks
# check that no na's were part of laterality_nonNA
set(list(pat_df.loc[pat_df['laterality_new'].isnull(),
                    'laterality'].index)).issubset(list(pat_df.loc[[(i not in laterality_nonNA)
                                                                    for i in pat_df['laterality']],
                                                                   'laterality'].index))
# check that all emptystr were put in bodypart_NA at first
set(list(pat_df.loc[pat_df['laterality'] == 'emptystr',
                    'laterality'].index)).issubset(list(pat_df.loc[[(i not in bodypart_nonNA)
                                                                    for i in pat_df['laterality']],
                                                                   'laterality'].index))

# ---- 3. view position ---- #

pat_df['view_position_new'] = [np.nan]*pat_df.shape[0]
view_position_NA = [(k not in view_position_nonNA) for k in pat_df['view_position']]

# dp
pat_df.loc[[(i in view_position_dp) for i in pat_df['view_position']], 'view_position_new'] = 'dp'  # view_position

seriesdescription_dp = [(i in series_description_dp) for i in pat_df['series_description']]  # series
seriesdescription_dp = [a and b for a, b in zip(view_position_NA, seriesdescription_dp)]

pat_df.loc[seriesdescription_dp, 'view_position_new'] = 'dp'

# lat
pat_df.loc[[(i in view_position_lat) for i in pat_df['view_position']], 'view_position_new'] = 'lat'  # view_position

seriesdescription_lat = [(i in series_description_lat) for i in pat_df['series_description']]  # series
seriesdescription_lat = [a and b for a, b in zip(view_position_NA, seriesdescription_lat)]

pat_df.loc[seriesdescription_lat, 'view_position_new'] = 'lat'

# checks
# check that no na's were part of view_position_nonNA
set(list(pat_df.loc[pat_df['view_position_new'].isnull(),
                    'view_position'].index)).issubset(list(pat_df.loc[[(i not in view_position_nonNA)
                                                                       for i in pat_df['view_position']],
                                                                      'view_position'].index))
# check that all emptystr were put in bodypart_NA at first
set(list(pat_df.loc[pat_df['view_position'] == 'emptystr',
                    'view_position'].index)).issubset(list(pat_df.loc[[(i not in bodypart_nonNA)
                                                                       for i in pat_df['view_position']],
                                                                      'view_position'].index))

# ---- 4. photometric_interpretation ---- #
pat_df['photometric_interpretation_new'] = [np.nan]*pat_df.shape[0]
photometric_interpretation_NA = [(k not in photometric_interpretation_nonNA)
                                 for k in pat_df['photometric_interpretation']]

# MONOCHROME1
pat_df.loc[[(i in photometric_interpretation_MOne)  # photometric_interpretation
            for i in pat_df['photometric_interpretation']], 'photometric_interpretation_new'] = 'MOne'

# MONOCHROME2
pat_df.loc[[(i in photometric_interpretation_MTwo)  # photometric_interpretation
            for i in pat_df['photometric_interpretation']], 'photometric_interpretation_new'] = 'MTwo'

# OTHER
pat_df.loc[[(i in photometric_interpretation_MOther)  # photometric_interpretation
            for i in pat_df['photometric_interpretation']], 'photometric_interpretation_new'] = 'MOther'

# checks
# check that no na's were part of photometric_interpretation_nonNA
set(list(pat_df.loc[pat_df['photometric_interpretation_new'].isnull(),
                    'photometric_interpretation'].index)).issubset(list(pat_df.loc[[(i not in
                                                                                     photometric_interpretation_nonNA)
                                                                                    for i in
                                                                                    pat_df['photometric_interpretation']
                                                                                    ],
                                                                                   'photometric_interpretation'].index))
# check that all emptystr were put in bodypart_NA at first
set(list(pat_df.loc[pat_df['photometric_interpretation'] == 'emptystr',
                    'photometric_interpretation'].index)).issubset(list(pat_df.loc[[(i not in bodypart_nonNA)
                                                                                    for i in
                                                                                    pat_df['photometric_interpretation']
                                                                                    ],
                                                                                   'photometric_interpretation'].index))

####################
# new file name/id #
####################


# ---- basic filename based on pat_id, study_date, categories ---- #

# replace np.nan by NAN (in order to be able to concatenate new categories to one string)
pat_df = pat_df.replace(np.nan, 'NaN', regex=True)
# for i in pat_df.columns:
#    print(pat_df.loc[pat_df[i].isnull(), 'bodypart_new'])
#    print(any(pat_df[i].isnull()))


# pat_df['category_new'] = pat_df.bodypart_new.str.cat(pat_df.laterality_new, sep='_')
# pat_df['bodypart_new'] + "_" + pat_df["laterality_new"].map(str)
category_cols = ['bodypart_new', 'laterality_new', 'view_position_new', 'photometric_interpretation_new']
filename_cols = ['pat_id', 'study_date', 'bodypart_new', 'laterality_new',
                 'view_position_new', 'photometric_interpretation_new']

newcategory_df = pat_df[category_cols[0:(len(category_cols)-1)]] + '_'
newcategory_df[category_cols[len(category_cols)-1]] = pat_df[category_cols[len(category_cols)-1]]
newcategory = newcategory_df.astype(str).sum(axis=1)

newfilename_df = pat_df[filename_cols[0:(len(filename_cols)-1)]] + '_'
newfilename_df[filename_cols[len(filename_cols)-1]] = pat_df[filename_cols[len(filename_cols)-1]]
newfilename = newfilename_df.astype(str).sum(axis=1)

pat_df['category_new'] = newcategory
pat_df['filename_new'] = newfilename


# ---- handle duplicated filename_new entries ---- #

# newfilename_dupl = newfilename + '_0'  # 0 = unique, 1 (n) = first (n-th) occurrence of duplicated element
# pat_df['filename_new_dupl'] = newfilename_dupl

# analysis of duplicated old and new filenames
duplicated_filepathnames = list(pat_df.loc[pat_df['filepathname_old'].duplicated(), 'filepathname_old'])
print('OUTPUT: nr of duplicated filepath+filename (count duplicated values only once):', len(duplicated_filepathnames))


duplicated_filenames = list(pat_df.loc[pat_df['filename_old'].duplicated(), 'filename_old'])
print('OUTPUT: nr of duplicated filenames (count duplicated values only once):', len(duplicated_filenames))


filename_new_within_dupl_filename_old = list(pat_df.loc[[a in duplicated_filenames
                                                         for a in list(pat_df['filename_old'])],
                                                        'filename_new'])
filename_new_within_dupl_filename_old_all = list(pat_df.loc[[b in filename_new_within_dupl_filename_old
                                                             for b in list(pat_df['filename_new'])], 'filename_new'])
print('OUTPUT: nr of duplicated filenames_new within duplicated_filenameold (count duplicated values only once):',
      sum(pat_df.loc[[a in duplicated_filenames for a in list(pat_df['filename_old'])], 'filename_new'].duplicated()))


duplicated_filenames_new = list(pat_df.loc[pat_df['filename_new'].duplicated(), 'filename_new'])
duplicated_filenamenew = pat_df.loc[[a in duplicated_filenames_new for a in list(pat_df['filename_new'])],
                                    'filename_new']
print('OUTPUT: nr of duplicated filenames_new overall (count duplicated values only once):',
      len(duplicated_filenames_new))


# for new filenames: create column to distinguish between duplicates
# 0 = unique, 1 (n) = first (n-th) occurrence of duplicated element
processed_names = []
newnames = []
oldnames_new = pat_df['filename_new']
for i in oldnames_new:
    if i not in duplicated_filenames_new:
        i_new = i + '_0'
    else:
        if i in processed_names:
            i_new = i + '_' + str(Counter(processed_names)[i] + 1)
        else:
            i_new = i + '_1'
    newnames.append(i_new)
    processed_names.append(i)

pat_df['filename_new_dupl'] = newnames
duplicated_filenames_new_adj = pat_df.loc[pat_df['filename_new_dupl'].duplicated(), 'filename_new_dupl']
print('OUTPUT: nr of duplicated filenames_new after adjusting names'
      ' for duplication (count duplicated values only once):',
      len(duplicated_filenames_new_adj))

############################################
# save edited pat_df as pat_df_filenamenew #
############################################
print("START: saving df with new filenames...")
pat_df.to_pickle(output_folder + os.sep + "pat_df_filenamenew.pkl")
pat_df.to_csv(output_folder + os.sep + "pat_df_filenamenew.csv")
print("DONE: saving df with new filenames")

print(datetime.datetime.now())
