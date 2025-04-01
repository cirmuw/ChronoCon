#################################################################################################################
# BEFORE EVERY NEW LOCALIZATION TRAINING RUN, STORE OLD RESULTS SOMEWHERE AND REMOVE FROM landmark_localization #
#################################################################################################################

"""
module add Legacy
module add Python/3.6.2-goolf-1.4.10
lddpython
"""

from distutils.dir_util import copy_tree
import shutil
import os

# copy files and folders (lddpython in command line on server or in slurm script)
folder = "/home/cir/tdeimel/MedicalDataAugmentationTool/tensorflow_train/experiments/landmark_localization"
file_folder_names = ['debug_train', 'debug_val', 'train.csv', 'test.csv', 'train', 'test', 'weights', 'out']
# files_folders = [folder + os.sep + i for i in file_folder_names]
dest_folder = folder + os.sep + 'results_archive/split2_F_joints27_H_before_F'

if not os.path.isdir(dest_folder):
    os.mkdir(dest_folder)

for i in file_folder_names:
    src = folder + os.sep + i
    dest = dest_folder + os.sep + i
    if os.path.isfile(src):
        shutil.copy(src, dest)
    else:
        os.mkdir(dest)
        copy_tree(src, dest)

# bash to remove contents of training-specific folders and alangolous for files
"""
quit()
cd "/home/cir/tdeimel/MedicalDataAugmentationTool/tensorflow_train/experiments/landmark_localization"
rm -rf debug_train/*
rm -rf debug_val/*
rm -rf train/*
rm -rf test/*
rm -rf weights/*
rm -rf out/*
rm train.csv
rm test.csv
cd
"""

##########################################################################################
# LOAD BACK PREVIOUSLY OBTAINED LOCALIZATION RESULTS (E.G., IN ORDER TO USE RUN_APPLIC() #
##########################################################################################

##################
# NOT TESTED YET #
##################

"""
module add Python/3.6.2-goolf-1.4.10
lddpython
"""

from distutils.dir_util import copy_tree
import shutil
import os

# copy files and folders (lddpython in command line on server or in slurm script)
folder = "/home/cir/tdeimel/MedicalDataAugmentationTool/tensorflow_train/experiments/landmark_localization"
file_folder_names = ['debug_train', 'debug_val', 'train.csv', 'test.csv', 'train', 'test', 'weights', 'out']
# files_folders = [folder + os.sep + i for i in file_folder_names]
dest_folder = folder + os.sep + 'results_archive/35PointsNoRois_72train_CMCMCMCPPIPEND'

for i in file_folder_names:
    dest = folder + os.sep + i
    src = dest_folder + os.sep + i
    if os.path.isfile(src):
        shutil.copy(src, dest)
    else:
        os.mkdir(dest)
        copy_tree(src, dest)
