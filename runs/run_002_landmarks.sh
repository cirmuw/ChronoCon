
set -x 
set -e 

###--------------------------------------------------------------------------------------------------------------------------###
###                                         FEET                                                                             ### 

# FEET 

# F.0.T
# -) Train landmark detection 50 cases
# python ~/code/RA/ra_utils/ra_utils/training/landmarks/01_train_mlflow.py  --config config_landmarks/feet/F_train_landmarks_08.yaml

# F.0.NOT_NEEDED
# Not needed .... Rerun training with all data: 
# python ~/code/RA/ra_utils/ra_utils/training/landmarks/01_train_mlflow.py  --config config_landmarks/feet/F_train_landmarks_08_ALL_DATA.yaml 

# F.0.I
# python ~/code/RA/ra_utils/ra_utils/inference/landmarks_inference.py  --config config_landmarks/inference/F_inference_50_cases.yaml

# F.1.T (retrain with 50+80 cases)
# python ~/code/RA/ra_utils/ra_utils/training/landmarks/01_train_mlflow.py  \
#   --config  /home/cwatzenboeck/code/RA/ra_utils/runs/config_landmarks/feet/F_train_landmarks_101.yaml

# F.1.I
# python ~/code/RA/ra_utils/ra_utils/inference/landmarks_inference.py  --config config_landmarks/inference/F_inference_130_cases.yaml



# F.2.T.prep. 
    # Combine labels -> see ... ~/data/AutoPIX_cirdata/projects__autoscora/tabular_data_cw/annotation_dir/combined_results
    # cd ~/data/AutoPIX_cirdata/projects__autoscora/tabular_data_cw/annotation_dir/combined_results
    # ra_utils__annotate_landmarks_combine_annotations --config config_H__TsTr_580.yml

    # Update splits file:
    # cd ~/data/AutoPIX_cirdata/projects__autoscora/landmark_data/splits/new_splits_nbodner
    # python  ~/code/RA/ra_utils/ra_utils/tools/landmark_annotation_update_splits_file.py  .... see commands.sh there

    # Exclude some according to ... 
    #   /home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/tabular_data_cw/annotation_dir_nbodner/annotation_dir/all/annotation_priorities/data_0-5_F_H_excluded_ids_only.yml


# F.2.T  580 cases
python ~/code/RA/ra_utils/ra_utils/training/landmarks/01_train_mlflow.py  \
   --config  /home/cwatzenboeck/code/RA/ra_utils/runs/config_landmarks/feet/F_train_landmarks_102.yaml




###                                                                                                                          ### 
###--------------------------------------------------------------------------------------------------------------------------###




###--------------------------------------------------------------------------------------------------------------------------###
###                                        HANDS                                                                             ### 
###                                                                                                                          ###


# H.0.T
# -) Train landmark detection 50 cases
# python ~/code/RA/ra_utils/ra_utils/training/landmarks/01_train_mlflow.py  --config config_landmarks/hands/train_landmarks_15.yaml

# H.0.NOT_NEEDED
# Not needed.... Rerun training with all data: 
# python ~/code/RA/ra_utils/ra_utils/training/landmarks/01_train_mlflow.py  --config config_landmarks/hands/train_landmarks_ALL_DATA.yaml

# H.0.I
# python ~/code/RA/ra_utils/ra_utils/inference/landmarks_inference.py  --config config_landmarks/inference/H_inference_50_cases.yaml


# H.1.T_prep
#
#   ### Combine landmarks 
#   #cd ~/data/AutoPIX_cirdata/projects__autoscora/tabular_data_cw/annotation_dir/combined_results
#   # ra_utils__annotate_landmarks_combine_annotations --config config_H__TsTr_0X.yml 
#   # -->   /home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/tabular_data_cw/annotation_dir/combined_results/out/H_files_TD_TrTs1Ts2_add03_new_labels.csv
#
#   ## Update splits file 
#   # cd    /home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/landmark_data/splits/new_splits_nbodner  
#   # see commands.sh there
#   #-> /home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/landmark_data/splits/new_splits_nbodner/splits_H_TD-03.yml




# H.1.T (retrain with 50+80 cases)
# python ~/code/RA/ra_utils/ra_utils/training/landmarks/01_train_mlflow.py  \
#    --config  /home/cwatzenboeck/code/RA/ra_utils/runs/config_landmarks/hands/train_landmarks_101.yaml


# H.1.I
# python ~/code/RA/ra_utils/ra_utils/inference/landmarks_inference.py  --config config_landmarks/inference/H_inference_130_cases.yaml
 

# H.2.T_prep 
# see F or above

# H.2.T  580 cases
# python ~/code/RA/ra_utils/ra_utils/training/landmarks/01_train_mlflow.py  \
#    --config  /home/cwatzenboeck/code/RA/ra_utils/runs/config_landmarks/hands/train_landmarks_102.yaml



