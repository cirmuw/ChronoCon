
set -x 
set -e 

###--------------------------------------------------------------------------------------------------------------------------###
###                                         Train Landmark detection                                                         ### 

# -) Train landmark detection 
# python ~/code/RA/ra_utils/ra_utils/training/landmarks/01_train_mlflow.py  --config config_landmarks/hands/train_landmarks_15.yaml
# python ~/code/RA/ra_utils/ra_utils/training/landmarks/01_train_mlflow.py  --config config_landmarks/feet/F_train_landmarks_08.yaml


# Rerun training with all data: 
# python ~/code/RA/ra_utils/ra_utils/training/landmarks/01_train_mlflow.py  --config config_landmarks/hands/train_landmarks_ALL_DATA.yaml
# python ~/code/RA/ra_utils/ra_utils/training/landmarks/01_train_mlflow.py  --config config_landmarks/feet/F_train_landmarks_08_ALL_DATA.yaml 

###                                                                                                                          ### 
###--------------------------------------------------------------------------------------------------------------------------###




###--------------------------------------------------------------------------------------------------------------------------###
###                                         Inference      detection                                                         ### 
###                                                                                                                          ###

########  0) Preprocessing (mirror images to one handedness)
# TODO
########


########  1) Landmark detection
# python ~/code/RA/ra_utils/ra_utils/inference/landmarks_inference.py  --config config_landmarks/inference/H_inference_DB.yaml
# python ~/code/RA/ra_utils/ra_utils/inference/landmarks_inference.py  --config config_landmarks/inference/H_inference.yaml

# python ~/code/RA/ra_utils/ra_utils/inference/landmarks_inference.py  --config config_landmarks/inference/F_inference_DB.yaml
# python ~/code/RA/ra_utils/ra_utils/inference/landmarks_inference.py  --config config_landmarks/inference/F_inference.yaml
########

####### OPTIONAL Rename columns to proper names: (DEPRICATED -- Added to landmarks_inference.py
#   python ~/code/RA/ra_utils/ra_utils/util_scripts/rename_landmarks.py -s \
#     -i /home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/landmark_data/output/H_coordinates__bb72a1aaf7f9476885d559d7baba9b8e_DB.csv  \
#     -o  /home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/landmark_data/output/H_coordinates__bb72a1aaf7f9476885d559d7baba9b8e_DB_renamed_transposed.csv --type H
#######


########  2) Patch extraction
# Cut out patches around each joint and save as numpy array
# Debugging...
#time python ~/code/RA/ra_utils/ra_utils/autoscora/autoscorRA_Pipeline/patch_extraction/patch_saving_dev_config.py  \
#             --config config_patches/H_patch_extraction_01.yml


# time python ~/code/RA/ra_utils/ra_utils/autoscora/autoscorRA_Pipeline/patch_extraction/patch_saving_dev_config.py  \
#             --config config_patches/H_patch_extraction_all.yml
########





###                                                                                                                          ### 
###--------------------------------------------------------------------------------------------------------------------------###




