
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

## Feet!!
#time python ~/code/RA/ra_utils/ra_utils/autoscora/autoscorRA_Pipeline/patch_extraction/patch_saving_dev_config.py  \
#            --config config_patches/F_patch_extraction_all_LOCAL.yml


#time python ~/code/RA/ra_utils/ra_utils/autoscora/autoscorRA_Pipeline/patch_extraction/patch_saving_dev_config.py  \
#            --config config_patches/H_patch_extraction_all_LOCAL.yml
########

#######  3) Training for scoring
# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/examples/F_ERO_ResNetMTANAE_lx100_FL.yml



###                                                                                                                          ### 
###--------------------------------------------------------------------------------------------------------------------------###



#### OLD: 


#### LOCAL DEV
#time python ~/code/RA/ra_utils/ra_utils/autoscora/autoscorRA_Pipeline/patch_extraction/patch_saving_dev_config.py  \
#             --config config_patches/H_patch_extraction_all_LOCAL.yml
########

#time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/02_train_mlflow.py  \
#           --config config_scoring/scoring_H_XX_dev_resnet.yml


# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/02_train_mlflow.py  \
#            --config config_scoring/scoring_H_XX_dev_resnet_PIPEPALL.yml


#----------------------------------------------------------------------------------------

#time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/02_train_mlflow.py  \
#           --config config_scoring/ERO_H_PIP_Multimodal_ResNet18.yml

#time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/02_train_mlflow.py  \
#           --config config_scoring/ERO_H_PIP_Multimodal_ResNet34.yml

#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/02_train_mlflow.py  \
#              --config config_scoring/ERO_H_PIP_SM_ResNet18.yml


#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/02_train_mlflow.py  \
#              --config config_scoring/ERO_H_PIP_SM_ResNet34.yml


#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/02_train_mlflow.py  \
#              --config config_scoring/ERO_H_ALL_Multimodal_ResNet18.yml 

#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/02_train_mlflow.py  \
#              --config config_scoring/ERO_H_ALL_Multimodal_ResNet34.yml


#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/02_train_mlflow.py   --config config_scoring/Exp01_balance_01.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/02_train_mlflow.py   --config config_scoring/Exp01_balance_02.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/02_train_mlflow.py   --config config_scoring/Exp01_balance_03.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/02_train_mlflow.py   --config config_scoring/Exp01_balance_04.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/02_train_mlflow.py   --config config_scoring/Exp01_balance_05.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/02_train_mlflow.py   --config config_scoring/Exp01_balance_06.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/02_train_mlflow.py   --config config_scoring/Exp01_balance_07.yml


#time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/03_train_mlflow_AE_and_mulitheaded_classifier.py   --config config_scoring/Exp02_AE.yml
#time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/03_train_mlflow_AE_and_mulitheaded_classifier.py   --config config_scoring/Exp02_AE_1.yml
#time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/03_train_mlflow_AE_and_mulitheaded_classifier.py   --config config_scoring/Exp02_AE_2.yml



# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/03_train_mlflow_AE_and_mulitheaded_classifier.py   --config config_scoring/Exp03_UNetMTANAE.yml
#time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/03_train_mlflow_AE_and_mulitheaded_classifier.py   --config config_scoring/Exp03_ResNetAE.yml
#time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/03_train_mlflow_AE_and_mulitheaded_classifier.py   --config config_scoring/Exp03_ResNetAE_froozen.yml
# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/03_train_mlflow_AE_and_mulitheaded_classifier.py   --config config_scoring/Exp03_ResNetNoAE.yml
#time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/03_train_mlflow_AE_and_mulitheaded_classifier.py   --config config_scoring/Exp03_ResNetAE_v2.yml
#time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/03_train_mlflow_AE_and_mulitheaded_classifier.py   --config config_scoring/Exp03_ResNetAE_v2_l1000.yml

#time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/03_train_mlflow_AE_and_mulitheaded_classifier.py   --config config_scoring/Exp03_ResNetAE_v2_l1000_no_sigmoid.yml
#time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/03_train_mlflow_AE_and_mulitheaded_classifier.py   --config config_scoring/Exp03_ResNetAE_v2_ly0.yml
#time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/03_train_mlflow_AE_and_mulitheaded_classifier.py   --config config_scoring/Exp03_ResNetAE_v2p1_l1000.yml
#time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/03_train_mlflow_AE_and_mulitheaded_classifier.py   --config config_scoring/Exp03_ResNetAE_v2p1_l1000_no_sigmoid.yml


# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/04_train_mlflow_MTANAE.py   --config config_scoring/Exp04_ResNetMTANAE_EROH.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/04_train_mlflow_MTANAE.py   --config config_scoring/Exp04_ResNetMTANAE_EROH_big_group.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/04_train_mlflow_MTANAE.py   --config config_scoring/Exp04_ResNetMTANAE_EROH_singles.yml

#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/04_train_mlflow_MTANAE.py   --config config_scoring/Exp04_ResNetMTANAE_EROH_noAE.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/04_train_mlflow_MTANAE.py   --config config_scoring/Exp04_ResNetMTANAE_EROH_big_group_noAE.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/04_train_mlflow_MTANAE.py   --config config_scoring/Exp04_ResNetMTANAE_EROH_singles_noAE.yml

#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/04_train_mlflow_MTANAE.py   --config config_scoring/Exp04_ResNetMTANAE_EROH_dropout.yml

# Did not run yet: 7.5.24 ... 
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/04_train_mlflow_MTANAE.py   --config config_scoring/Exp04_ResNetMTANAE_EROH_big_group_dropout.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/04_train_mlflow_MTANAE.py   --config config_scoring/Exp04_ResNetMTANAE_EROH_singles_dropout.yml


#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/04_train_mlflow_MTANAE.py   --config config_scoring/Exp04_MTMAN_EROH_02.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/03_train_mlflow_AE_and_mulitheaded_classifier.py   --config config_scoring/Exp04_UNetMTANAE_EROH_02.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/03_train_mlflow_AE_and_mulitheaded_classifier.py   --config config_scoring/Exp04_ResNetAE_EROH_02.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/03_train_mlflow_AE_and_mulitheaded_classifier.py   --config config_scoring/Exp04_ResNetNoAE_EROH_02.yml


#time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/04_train_mlflow_MTANAE.py   --config config_scoring/Exp04_MTMAN_EROH.yml
#time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/03_train_mlflow_AE_and_mulitheaded_classifier.py   --config config_scoring/Exp04_UNetMTANAE_EROH.yml
#time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/03_train_mlflow_AE_and_mulitheaded_classifier.py   --config config_scoring/Exp04_ResNetAE_EROH.yml
#time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/03_train_mlflow_AE_and_mulitheaded_classifier.py   --config config_scoring/Exp04_ResNetNoAE_EROH.yml


#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/03_train_mlflow_AE_and_mulitheaded_classifier.py   --config config_scoring/Exp03_UNetMTANAE_MLP.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/03_train_mlflow_AE_and_mulitheaded_classifier.py   --config config_scoring/Exp03_ResNetAE_MLP.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/03_train_mlflow_AE_and_mulitheaded_classifier.py   --config config_scoring/Exp03_ResNetAE_froozen_MLP.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/03_train_mlflow_AE_and_mulitheaded_classifier.py   --config config_scoring/Exp03_ResNetNoAE_MLP.yml

#### Triplet: 


# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py   --config config_scoring/Exp05_ResNetAE_EROH_TRIPLET.yml
# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py   --config config_scoring/Exp05_ResNetAE_EROH_NoTRIPLET.yml

#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py   --config config_scoring/Exp05_ResNetMTANAE_EROH_TRIPLET.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py   --config config_scoring/Exp05_ResNetMTANAE_EROH_NoTRIPLET.yml

#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py   --config config_scoring/Exp05_ResNetMTANAE_EROH_groups_TRIPLET.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py   --config config_scoring/Exp05_ResNetMTANAE_EROH_groups_NoTRIPLET.yml

#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py   --config config_scoring/Exp05_UNetMTANAE_EROH_NoTRIPLET.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py   --config config_scoring/Exp05_UNetMTANAE_EROH_TRIPLET.yml

## PIP only:  #-----------------------------------------#
# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/03_train_mlflow_AE_and_mulitheaded_classifier.py  --config config_scoring/Exp06/EROHD_PIP_UNetMTAN.yml

# Done
# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp06/EROHD_PIP_UNetMTANAE_with_triplet_loss.yml
# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp06/EROHD_PIP_UNetMTANAE_with_triplet_loss_2.yml
# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp06/EROHD_PIP_ResNetAE_lx100.yml


# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp06/EROHD_PIP_ResNetMTANAE_lx100.yml
# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp06/EROHD_PIP_ResNetMTANAE_lx100_v2.yml
# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp06/EROHD_PIP_ResNetMTANAE_lx100_v3.yml
# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp06/EROHD_PIP_ResNetMTANAE_lx100_v4.yml

#time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp06/EROHD_PIP_UNetMTANAE_with_triplet_loss_att_paths.yml
#time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp06/EROHD_PIP_UNetMTANAE_with_triplet_loss_euclid.yml


# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp06/EROHD_PIP_UNetMTANAE_with_triplet_loss.yml

# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp06/EROHD_PIP_UNetMTANAE_with_triplet_loss_Lx10.yml
# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp06/EROHD_PIP_UNetMTANAE_dataloaders.yml


# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp06/EROHD_PIP_UNetMTANAE_with_triplet_loss.yml
# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp06/EROHD_PIP_UNetMTANAE_with_triplet_loss_no_ws.yml
# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp06/EROHD_PIP_ResNetNoAE_no_ws.yml
# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp06/EROHD_PIP_ResNetNoAE.yml
# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp06/EROHD_PIP_ResNetNoAE_lr2.yml
# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp06/EROHD_PIP_ResNetNoAE_mulithead.yml
# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp06/EROHD_PIP_ResNetNoAE_mulithead_no_ws.yml
# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp06/EROHD_PIP_ResNetNoAE_multiheads_lr2.yml


###                                                                                                                          ### 
###--------------------------------------------------------------------------------------------------------------------------###

#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp07/EROHD_PIP_ResNetMTANAE_lx100_v3.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp07/EROHD_PIP_ResNetMTANAE_lx100_v4_FL.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp07/EROHD_PIP_ResNetMTANAE_lx100_v4_PFL.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp07/EROHD_PIP_ResNetMTANAE_lx100_v4_PL.yml


# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp11/H_JSN_ResNetMTANAE_lx100_FL.yml
# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp10/F_ERO_JSN_ResNetMTANAE_lx100_FL.yml

#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp12/F_ERO_ResNetMTANAE_lx100_FL.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp12/F_ERO_ResNetMTANAE_lx100_FL_with_alpha.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp12/F_ERO_ResNetMTANAE_lx100_FL_with_alpha2.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp12/F_ERO_ResNetMTANAE_lx100_FL_single_path_many_heads.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp12/F_ERO_ResNetMTANAE_lx100_FL_single_path_single_head.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp12/F_ERO_UNetMTANAE_lx100_FL_single_path_single_head.yml




#time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp12/F_ERO_ResNetMTANAE_lx100_FL_single_path_many_heads.yml
#time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp12/F_ERO_ResNetNoAE.yml


#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp13/F_ERO_ResNetNoAE_01.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp13/F_ERO_ResNetNoAE_02.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp13/F_ERO_ResNetNoAE_03.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp13/F_ERO_ResNetNoAE_04.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp13/F_ERO_ResNetNoAE_05.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp13/F_ERO_ResNetNoAE_06.yml

#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp12/F_ERO_ResNetNoAE_MH.yml


#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp15/H_JSN_01.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp15/H_F_JSN_01.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp15/H_F_ERO_01.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp15/H_F_ERO_JSN_01.yml

#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp15/H_F_ERO_JSN_02.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp15/H_F_ERO_JSN_02a.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp15/H_F_ERO_JSN_02b.yml
#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp15/H_F_ERO_JSN_03.yml

# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config config_scoring/Exp17_dev_HPS/F_JSN_dev.yml  --debugging 
## time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/06_hps.py  --config config_scoring/Exp17_dev_HPS/HPS__F_JSN_dev.yml  
# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/06_hps.py  --config config_scoring/Exp17_dev_HPS/HPS__F_JSN_dev.yml  


# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/06_hps.py  --config config_scoring/Exp17_dev_HPS/HPS__F_JSN_dev2.yml
# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/06_hps.py  --config config_scoring/Exp17_dev_HPS/HPS__F_JSN_dev3.yml

# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config   config_scoring/Exp17_dev_HPS/dev.yml  --debugging
# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config   config_scoring/Exp17_dev_HPS/dev.yml
# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py  --config   config_scoring/Exp17_dev_HPS/dev_cls.yml





# time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/06_hps.py  --config config_scoring/Exp18_HPS/HPS__ALL.yml


#   time python ~/code/RA/ra_utils/ra_utils/training/scores_SHS/05_train_mlflow_triplet.py \
#      --config config_scoring/Exp17_dev_HPS/config_trial.yml 







