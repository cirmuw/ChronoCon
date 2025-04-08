import numpy as np
import torch
import time
import os
import shutil
import sys
import argparse
import csv
from torch.utils.data import DataLoader

import ra_utils
import ra_utils.autoscora.autoscorRA_Pipeline.scoring.src

from ra_utils.autoscora.autoscorRA_Pipeline.scoring.src.io_scoring_method import io_scoring, mandatory_train_val_test_ids
from ra_utils.autoscora.autoscorRA_Pipeline.scoring.src.network import Custom_VGG
from ra_utils.autoscora.autoscorRA_Pipeline.scoring.src.load import get_indices4, load2, balance_train_set, balanced_mean, get_patient_nr, CustomDataset
from ra_utils.autoscora.autoscorRA_Pipeline.scoring.src.train import train, validate, test
from ra_utils.autoscora.autoscorRA_Pipeline.scoring.src.wrist import get_sum_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


params = {}


params["all_vgg_types"] = ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19']
params["vgg_type"] = params["all_vgg_types"][6]

networks = [
[["IPIEP", "PIPIIEP", "PIPIIIEP", "PIPIVEP", "PIPVEP"], "ERO", "H"],  # 0
[["IPIED", "PIPIIED", "PIPIIIED", "PIPIVED", "PIPVED"], "ERO", "H"],  # 1
[["PIPII", "PIPIII", "PIPIV", "PIPV"], "JSN", "H"],                   # 2
[["MCPIEP", "MCPIIEP", "MCPIIIEP", "MCPIVEP", "MCPVEP"], "ERO", "H"], # 3
[["MCPIED", "MCPIIED", "MCPIIIED", "MCPIVED", "MCPVED"], "ERO", "H"], # 4
[["MCPI", "MCPII", "MCPIII", "MCPIV", "MCPV"], "JSN", "H"],           # 5
[["Base_MCIE"], "ERO", "H"],                                          # 6
[["CMCIII", "CMCIV", "CMCV"], "JSN", "H"],                            # 7
[["LunatE", "RadiusE", "ScaphE", "TrapE", "UlnaE"], "ERO", "H"],      # 8
[["LunatE"], "ERO", "H"],                                             # 9
[["RadiusE"], "ERO", "H"],                                            # 10
[["ScaphE"], "ERO", "H"],                                             # 11
[["TrapE"], "ERO", "H"],                                              # 12
[["UlnaE"], "ERO", "H"],                                              # 13
[["Rad_Carp", "Sca_Cap", "Tra_Sca"], "JSN", "H"],                     # 14
[["Rad_Carp"], "JSN", "H"],                                           # 15
[["Tra_Sca"], "JSN", "H"],                                            # 16
[["Sca_Cap"], "JSN", "H"],                                            # 17
### FEET ###
[["MTPIEP"], "ERO", "F"],                                             # 18
[["MTPIED"], "ERO", "F"],                                             # 19
[["MTPIIEP", "MTPIIIEP", "MTPIVEP", "MTPVEP"], "ERO", "F"],           # 20
[["MTPIIED", "MTPIIIED", "MTPIVED", "MTPVED"], "ERO", "F"],           # 21
[["IPEP"], "ERO", "F"],                                               # 22
[["IPED"], "ERO", "F"],                                               # 23
[["MTPI"], "JSN", "F"],                                               # 24
[["MTPII", "MTPIII", "MTPIV", "MTPV"], "JSN", "F"],                   # 25
[["IP"], "JSN", "F"]                                                  # 26
]

val = 5 #8
params["chosen_score"] = networks[val][0]
params["chosen_score_type"] = networks[val][1]
params["extremity"] = networks[val][2]
#params["chosen_score"] = ['RadiusE', 'ScaphE', 'TrapE', 'UlnaE', 'UlnaE', 'LunatE']
#params["chosen_score_type"] = "ERO"
#print(params["chosen_score"])
#print(params["chosen_score_type"])

if params["extremity"] == "H":
    # params["path_orig"] = "/project/autoscora/autoscoRA_images/H_images_of_interest_2_renamed_mirrored_inverted_patches/"
    # params["path_aug"] = "/project/autoscora/autoscoRA_images/H_images_of_interest_2_renamed_mirrored_inverted_augmented_patches/"

    # params["path_orig"] = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/autoscoRA_images/H_images_of_interest_2_renamed_mirrored_inverted_patches/"
    params["path_orig"] = "/home/cwatzenboeck/data/AutoPIX_local_data/autoscoRA_images/H_images_of_interest_2_renamed_mirrored_inverted_patches/"
    params["path_aug"] = "/home/cwatzenboeck/data/AutoPIX_local_data/autoscoRA_images/H_images_of_interest_2_renamed_mirrored_inverted_patches/"
                        #"/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/autoscoRA_images/H_images_of_interest_2_renamed_mirrored_inverted_augmented_patches/"
elif params["extremity"] == "F":
    # params["path_orig"] = "/project/autoscora/autoscoRA_images/F_images_of_interest_2_renamed_mirrored_inverted_patches/"
    # params["path_aug"] = "/project/autoscora/autoscoRA_images/F_images_of_interest_2_renamed_mirrored_inverted_augmented_patches/"

    params["path_orig"] = "NOT USED /home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/autoscoRA_images/F_images_of_interest_2_renamed_mirrored_inverted_patches/"
    params["path_aug"] =  "NOT USED" #"/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/autoscoRA_images/F_images_of_interest_2_renamed_mirrored_inverted_augmented_patches/"
params["path_list"] = []
params["score_list"] = []

params["binary"] = 0 # 0 ... all classes / 1 ... [0] vs. rest / 2 ... [0,1] vs. rest
params["n_classes"] = None  # Will be computed automatically
params["classes"] = None  # Will be computed automatically

#train_params = [100, 150, -1, -1, -1, True]  # for training
train_params = [2, 20, 5, 5, 5, True]  # for testing

params["n_epochs"] = train_params[0]
params["batch_size"] = train_params[1]
params["n_batches"] = train_params[2] # -1 for whole dataset
params["n_batches_val"] = train_params[3] # -1 for whole dataset
params["n_batches_test"] = train_params[4] # -1 for whole dataset
params["verbose"] = train_params[5]
params["lr"] = 5e-7
# IF regression == True, then regression is applied
# IF regression == False and ordinal == True, then ordinal is applied
# IF regression == False, ordinal == False, weighted_kappa == False, then regular classification is applied
# IF regression == False, ordinal == False, weighted_kappa == True, then weighted classification is applied
params["regression"] = False
params["ordinal"] = False
params["weighted_kappa"] = False
params["lambda"] = 1    # <---!!!!! CW: will be overwritten

params["train_test_split"] = [0.6, 0.2, 0.2]
params["balance"] = 1 # value between 0 and 1. 0 not balanced, 1 fully balanced
params["augment"] = False #True
params["train_indices"] = np.array([], dtype="int")
params["val_indices"] = np.array([], dtype="int")
params["test_indices"] = np.array([], dtype="int")



params["model_name"] =  "test"
params["path_to_model"] = "models/" + params["model_name"] + "/"
params["path_to_predictions"] = params["path_to_model"] + "predictions/"
#params["path_to_train_patients"] = "src/Localization_Pat.txt"
params["path_to_train_patients"] = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/autoscoRA_TDEIMEL_HOME/autoscoRA_Pipeline/scoring/src/Localization_Pat.txt"
if params["extremity"] == "H":
    #params["path_to_op_patients"] = "src/op_patients_date_HF_LR.txt"
    params["path_to_op_patients"] = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/autoscoRA_TDEIMEL_HOME/autoscoRA_Pipeline/scoring/src/op_patients_date_HF_LR.txt"
elif params["extremity"] == "F":
    #params["path_to_op_patients"] = "src/op_patients_date_HF_LR_FEET.txt"
    params["path_to_op_patients"] = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/autoscoRA_TDEIMEL_HOME/autoscoRA_Pipeline/scoring/src/op_patients_date_HF_LR_FEET.txt"
params["path_to_all_patients_npy"] = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/autoscoRA_TDEIMEL_HOME/autoscoRA_Pipeline/scoring/src/all_patients_num.npy"

# CW: corresponds to input_constants.py in autoscoRA_Pipeline
params["input_constants_yml_path"] = "/home/cwatzenboeck/code/RA/ra_utils/ra_utils/autoscora/autoscorRA_Pipeline/input/constants/input_constants_cw.yml"

b_clean_test = True
b_train = True
b_pred = True


aug_params = {'prob': 0.81762,
        'spacing': 12.4311,
        'magnitude': 2.7323,
        'rotate': 0.19521,
        'scale': 0.0319,
        'shear': 0.13073,
        'translate': 1.7906,
        'gaussia_prob': 0.1764,
        'gaussia_mean': 0.02283,
        'gaussia_std': 0.08562}

#prob:0.8176239835956653
#spacing:12.431112991902715
#magnitude:2.7323400486866776
#rotate:0.1952112128956887
#scale:0.031922514891724374
#shear:0.1307369175208472
#translate:1.790684274194188
#gaussia_prob:0.17645025845665457
#gaussia_mean:0.02283402915856836
#gaussia_std:0.08562432997747216

### Read Parameters ###

parser = argparse.ArgumentParser()
parser.add_argument("--name", "--model_name", help="Define the name of the model")
parser.add_argument("--bin", help="Define the categorisation of the classes", type=int)
parser.add_argument("--val", help="Define the joints that are used for training", type=int)
parser.add_argument("--augment", help="Define if augmentation is applied")
parser.add_argument("--regression", help="Apply regression (or classification)")
parser.add_argument("--kappa", help="Apply weighted kappa loss, only for classification")
parser.add_argument("--lam", help="If weighted kappa loss is applied, lambda parameter can be chosen here")
parser.add_argument("--ordinal", help="Set true to apply ordinal loss")

args = parser.parse_args()
if args.name:
    params["model_name"] = args.name
    #params["path_to_model"] = "models/"
    params["path_to_model"] = "/home/cwatzenboeck/data/tmp/dev_autoscora/models/scoring/" + params["model_name"] + "/"
    params["path_to_predictions"] = params["path_to_model"] + "predictions/"
if args.bin:
    params["binary"] = args.bin
if args.val:
    val = args.val
    params["chosen_score"] = networks[val][0]
    params["chosen_score_type"] = networks[val][1]
    params["extremity"] = networks[val][2]
if args.augment:
    if args.augment.lower() in ('yes', 'true', 't', 'y', '1'):
        params["augment"] = True
    elif args.augment.lower() in ('no', 'false', 'f', 'n', '0'):
        params["augment"] = False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected for argument')
if args.regression:
    if args.regression.lower() in ('yes', 'true', 't', 'y', '1'):
        params["regression"] = True
    elif args.regression.lower() in ('no', 'false', 'f', 'n', '0'):
        params["regression"] = False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected for argument')
if args.kappa:
    if args.kappa.lower() in ('yes', 'true', 't', 'y', '1'):
        params["weighted_kappa"] = True
    elif args.kappa.lower() in ('no', 'false', 'f', 'n', '0'):
        params["weighted_kappa"] = False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected for argument')
if args.lam:
    params["lambda"] = float(args.lam)
if args.ordinal:
    if args.ordinal.lower() in ('yes', 'true', 't', 'y', '1'):
        params["ordinal"] = True
    elif args.ordinal.lower() in ('no', 'false', 'f', 'n', '0'):
        params["ordinal"] = False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected for argument')



### Initialize ###

if b_clean_test == True:
    if params["model_name"] != "test":
        if os.path.isdir(params["path_to_model"]) == True:
            print(f"Model already exists in {params['path_to_model']}. Choose different model name.")
            sys.exit()
        else:
            os.mkdir(params["path_to_model"])
            os.mkdir(params["path_to_predictions"])
    else:
        if os.path.isdir(params["path_to_model"]) == True:
            shutil.rmtree(params["path_to_model"])
        os.mkdir(params["path_to_model"])
        os.mkdir(params["path_to_predictions"])


path_list = []
score_list = []
for i in range(len(params["chosen_score"])):
    print("chose_score: ", params["chosen_score"][i])
    print("chosen_score_type: ", params["chosen_score_type"])
    print("extremity: ", params["extremity"])
    path_list_tmp, score_list_tmp = io_scoring(chosen_score = params["chosen_score"][i],
                                                chosen_score_type = params["chosen_score_type"],
                                                extremity = params["extremity"])

    path_list = path_list + path_list_tmp
    score_list = score_list + score_list_tmp
#print(len(path_list))
#print(path_list[:10])

if params["chosen_score"] == ["LunatE", "RadiusE", "ScaphE", "TrapE", "UlnaE"]:
    path_list, score_list = get_sum_score(path_list, score_list)
if params["chosen_score"] == ["Rad_Carp", "Sca_Cap", "Tra_Sca"]:
    path_list, score_list = get_sum_score(path_list, score_list)



dict_indices = mandatory_train_val_test_ids(input_constants_yml_path = params["input_constants_yml_path"])
patient_nr, op_patient_nr =  get_patient_nr(path_to_train_patients=params["path_to_train_patients"],
                                                path_to_op_patients=params["path_to_op_patients"])

train_indices, val_indices, test_indices, score_list, path_list = get_indices4(train_test_split = params["train_test_split"],
                                                                                len_cur_path_list = len(params["path_list"]),
                                                                                path_list = path_list,
                                                                                score_list = np.array(score_list),
                                                                                train_pat = dict_indices["train"],
                                                                                val_pat = dict_indices["val"],
                                                                                test_pat = dict_indices["test"],
                                                                                op_patients = op_patient_nr,
                                                                                chosen_score = params["chosen_score"], 
                                                                                path_to_all_patients_npy=params["path_to_all_patients_npy"],)

params["path_list"] = params["path_list"] + path_list
params["score_list"] = np.concatenate([params["score_list"], score_list])

params["train_indices"] = np.concatenate([params["train_indices"], train_indices])
params["val_indices"] = np.concatenate([params["val_indices"], val_indices])
params["test_indices"] = np.concatenate([params["test_indices"], test_indices])
print("train_indices: ", len(params["train_indices"]))
print("val_indices: ", len(params["val_indices"]))
print("test_indices: ", len(params["test_indices"]))


unique_tmp, count_tmp = np.unique(params["score_list"][params["train_indices"]], return_counts=True)
if params["binary"] == 0:
    #params["n_classes"] = len(unique_tmp)
    #params["classes"] = unique_tmp
    params["n_classes"] = 6
    params["classes"] = np.array([0., 1., 2., 3., 4., 5.])
    if params["chosen_score"] == ["LunatE", "RadiusE", "ScaphE", "TrapE", "UlnaE"]:
        params["n_classes"] = 26
        params["classes"] = np.arange(26.0)
    if params["chosen_score"] == ["Rad_Carp", "Sca_Cap", "Tra_Sca"]:
        params["n_classes"] = 16
        params["classes"] = np.arange(16.0)
elif params["binary"] == 1:
    params["n_classes"] = 2
    params["classes"] = np.array([0., 1.])
elif params["binary"] == 2:
    params["n_classes"] = 2
    params["classes"] = np.array([0., 1.])
else:
    print("please define \"binary\" ")
params["train_indices"] = balance_train_set(train_indices = params["train_indices"],
                                            score_list = params["score_list"],
                                            balance = params["balance"],
                                            binary = params["binary"])
params["val_indices_bal"] = balance_train_set(train_indices = params["val_indices"],
                                                score_list = params["score_list"],
                                                balance = params["balance"],
                                                binary = params["binary"])


if params["n_batches"] != -1:
    params["train_indices"] = params["train_indices"][:params["n_batches"]*params["batch_size"]]
if params["n_batches_val"] != -1:
    params["val_indices_bal"] = params["val_indices_bal"][:params["n_batches_val"]*params["batch_size"]]
    params["val_indices"] = params["val_indices"][:params["n_batches_val"]*params["batch_size"]]
if params["n_batches_test"] != -1:
    params["test_indices"] = params["test_indices"][:params["n_batches_test"]*params["batch_size"]]


unique, count = np.unique(params["score_list"][params["val_indices"]], return_counts=True)
print("Validation Data:")
print(unique, count)
unique, count = np.unique(params["score_list"][params["val_indices_bal"]], return_counts=True)
print("Validation Data Balanced:")
print(unique, count)
unique, count = np.unique(params["score_list"][params["test_indices"]], return_counts=True)
print("Test Data:")
print(unique, count)
print("\n")


if params["n_batches"] == -1:
    params["n_batches"] = int(np.ceil(len(params["train_indices"])/params["batch_size"]))

if params["n_batches_val"] == -1:
    params["n_batches_val"] = int(np.ceil(len(params["val_indices"])/params["batch_size"]))

if params["n_batches_test"] == -1:
    params["n_batches_test"] = int(np.ceil(len(params["test_indices"])/params["batch_size"]))

file = open(params["path_to_model"] + "logfile.txt","a")
file.write("Notes:" + "\n")
file.write("Model Name: " + params["model_name"] + "\n")
file.write("VGG type: " + params["vgg_type"] + "\n")
file.write("\n")
file.write("Score: " + str(params["chosen_score"]) + "\n")
file.write("Score type: " + params["chosen_score_type"] + "\n")
file.write("Classes: " + str(params["n_classes"]) + "\n")
file.write("Binary type: " + str(params["binary"]) + "\n")
file.write("\n")
file.write("Total epochs trained: " + str(params["n_epochs"]) + "\n")
file.write("Batch size: " + str(params["batch_size"]) + "\n")
file.write("Batches per Epoch: " + str(params["n_batches"]) + "\n")
file.write("Batches per validation Epoch: " + str(params["n_batches_val"]) + "\n")
file.write("Learning rate: " + str(params["lr"]) + "\n")
file.write("\n")
file.write("Data Augmentation: " + str(params["augment"]) + "\n")
file.write("Regression: " + str(params["regression"]) + "\n")
file.write("Ordinal: " + str(params["ordinal"]) + "\n")
file.write("Weighted Kappa: " + str(params["weighted_kappa"]) + "\n")
file.write("Lambda: " + str(params["lambda"]) + "\n")
file.write("\n")
file.write("Data Balance:" + "\n")
file.write("Training Data before upsampling: " + str(unique_tmp) + str(count_tmp) + "\n")
unique, count = np.unique(params["score_list"][params["train_indices"]], return_counts=True)
file.write("Training Data: " + str(unique) + str(count) + "\n")
unique, count = np.unique(params["score_list"][params["val_indices"]], return_counts=True)
file.write("Validation Data: " + str(unique) + str(count) + "\n")
unique, count = np.unique(params["score_list"][params["test_indices"]], return_counts=True)
file.write("Test Data: " + str(unique) + str(count) + "\n")
file.write("\n")
file.close()


### Network ###

model = Custom_VGG(ipt_size=(128, 128),
                    pretrained=True,
                    num_classes=params["n_classes"],
                    vgg_type = params["vgg_type"],
                    regression = params["regression"],
                    ordinal = params["ordinal"]).to(device)



optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])


### Dataloader ###

training_data = CustomDataset(path = params["path_aug"],
                                    path_list = params["path_list"],
                                    score_list = params["score_list"],
                                    indices = params["train_indices"],
                                    binary = params["binary"],
                                    add_augm = False,  # used to be True
                                    augment = True,
                                    aug_params = aug_params)

train_dataloader = DataLoader(training_data, batch_size=params["batch_size"], shuffle=True, num_workers=8)


validation_data = CustomDataset(path = params["path_orig"],
                                    path_list = params["path_list"],
                                    score_list = params["score_list"],
                                    indices = params["val_indices_bal"],
                                    binary = params["binary"],
                                    add_augm = False,
                                    augment = False)

validation_dataloader = DataLoader(validation_data, batch_size=params["batch_size"], shuffle=True, num_workers=8)


### Training ###
if b_train:

    old_loss_batch = np.array([0]*params["n_classes"])
    old_loss_epoch = np.array([0]*params["n_classes"])
    log_epoch_loss = []

    old_loss_batch_val = np.array([0]*params["n_classes"])
    old_loss_epoch_val = np.array([0]*params["n_classes"])
    log_epoch_loss_val = []
    lowest_epoch_loss_val = np.array([])


    model_save_name = "model"


    for epoch in range(params["n_epochs"]):
        count_size = 0
        cur_loss_epoch = 0
        y_raw_epoch = np.array([])

        log_batch_loss = []
        log_batch_loss_val = []
        count_values_epoch = [np.array([0]*params["n_classes"])]
        count_values_epoch_val = [np.array([0]*params["n_classes"])]


        sta_epoch = time.time()
        #for i in range(params["n_batches"]):
        for i, batch in enumerate(train_dataloader):
            sta_batch = time.time()

            x_raw, y_raw, path_to_dataset = batch

            cur_loss_batch, count_values_batch = train(data_train_x = x_raw,
                                                        data_train_y = y_raw,
                                                        model = model,
                                                        optimizer = optimizer,
                                                        n_classes = params["n_classes"],
                                                        classes = params["classes"],
                                                        augment = params["augment"],
                                                        aug_params = aug_params,
                                                        regression = params["regression"],
                                                        weighted_kappa = params["weighted_kappa"],
                                                        lam = params["lambda"],
                                                        ordinal=params["ordinal"])

            #cur_loss_batch = cur_loss_batch.cpu().detach().numpy()
            cur_loss_batch = cur_loss_batch
            log_batch_loss.append(cur_loss_batch)
            count_values_epoch.append(count_values_batch)

            delt_loss_batch = balanced_mean(old_loss_batch, count_values_epoch[-2]) - balanced_mean(cur_loss_batch, count_values_epoch[-1])
            sto_batch = time.time() - sta_batch
            log_batch = ' ~ Epoch: {:03d}, Batch: ({:03d}/{:03d}) Loss: {:.8f}, Time: {:.4f}, Change in loss: {:.8f}'
            if params["verbose"]:
                print(log_batch.format(epoch+1, i+1, params["n_batches"], balanced_mean(cur_loss_batch, count_values_epoch[-1]), sto_batch, delt_loss_batch))
            old_loss_batch = cur_loss_batch


        #y_raw_epoch, log_batch_loss
        count_values_sum = np.array([0.]*params["n_classes"])
        cur_loss_epoch = np.array([0.]*params["n_classes"])
        for i in range(len(count_values_epoch)-1):
            cur_loss_epoch += log_batch_loss[i] * count_values_epoch[i+1]
            count_values_sum += count_values_epoch[i+1]
        cur_loss_epoch = np.divide(cur_loss_epoch, count_values_sum, out=np.zeros_like(cur_loss_epoch), where=count_values_sum!=0)

        log_epoch_loss.append((cur_loss_epoch, balanced_mean(cur_loss_epoch, count_values_sum)))
        delt_loss_epoch = balanced_mean(old_loss_epoch, count_values_sum) - balanced_mean(cur_loss_epoch, count_values_sum)
        sto_epoch = time.time() - sta_epoch
        log_epoch = 'Epoch: {:03d}, Loss: {:.8f}, Time: {:.4f}, Change in loss: {:.8f}\n'
        print(log_epoch.format(epoch+1, balanced_mean(cur_loss_epoch, count_values_sum) , sto_epoch ,delt_loss_epoch))
        old_loss_epoch = cur_loss_epoch



        ### Validate ###

        count_size = 0
        cur_loss_epoch_val = 0
        sta_epoch_val = time.time()
        #for i in range(params["n_batches_val"]):
        for i, batch in enumerate(validation_dataloader):
            sta_batch_val = time.time()

            x_raw, y_raw, path_to_dataset = batch

            cur_loss_batch_val, count_values_batch_val = validate(data_val_x = x_raw,
                                                        data_val_y = y_raw,
                                                        model = model,
                                                        n_classes = params["n_classes"],
                                                        classes = params["classes"],
                                                        regression = params["regression"],
                                                        weighted_kappa = params["weighted_kappa"],
                                                        lam = params["lambda"],
                                                        ordinal=params["ordinal"])



            log_batch_loss_val.append(cur_loss_batch_val)
            count_values_epoch_val.append(count_values_batch_val)
            delt_loss_batch_val = balanced_mean(old_loss_batch_val, count_values_epoch_val[-2]) - balanced_mean(cur_loss_batch_val, count_values_epoch_val[-1])
            sto_batch_val = time.time() - sta_batch_val
            log_batch_val = ' ~ Validate Epoch: {:03d}, Batch: ({:03d}/{:03d}) Loss: {:.8f}, Time: {:.4f}, Change in loss: {:.8f}'
            if params["verbose"]:
                print(log_batch_val.format(epoch+1, i+1, params["n_batches_val"], balanced_mean(cur_loss_batch_val, count_values_epoch_val[-1]), sto_batch_val, delt_loss_batch_val))

            old_loss_batch_val = cur_loss_batch_val



        count_values_sum_val = np.array([0.]*params["n_classes"])
        cur_loss_epoch_val = np.array([0.]*params["n_classes"])
        for i in range(len(count_values_epoch_val)-1):
            cur_loss_epoch_val += log_batch_loss_val[i] * count_values_epoch_val[i+1]
            count_values_sum_val += count_values_epoch_val[i+1]
        cur_loss_epoch_val = np.divide(cur_loss_epoch_val, count_values_sum_val, out=np.zeros_like(cur_loss_epoch_val), where=count_values_sum_val!=0)

        log_epoch_loss_val.append((cur_loss_epoch_val, balanced_mean(cur_loss_epoch_val, count_values_sum_val)))
        delt_loss_epoch_val = balanced_mean(old_loss_epoch_val, count_values_sum_val) - balanced_mean(cur_loss_epoch_val, count_values_sum_val)
        sto_epoch_val = time.time() - sta_epoch_val
        log_epoch_val = 'Validate Epoch: {:03d}, Loss: {:.8f}, Time: {:.4f}, Change in loss: {:.8f}\n'
        print(log_epoch_val.format(epoch+1, balanced_mean(cur_loss_epoch_val, count_values_sum_val), sto_epoch_val ,delt_loss_epoch_val))
        old_loss_epoch_val = cur_loss_epoch_val

        if lowest_epoch_loss_val.size == 0 or balanced_mean(cur_loss_epoch_val, count_values_sum_val) < lowest_epoch_loss_val:
            lowest_epoch_loss_val = balanced_mean(cur_loss_epoch_val, count_values_sum_val)
            torch.save(model.state_dict(), params["path_to_model"] + model_save_name + ".pt")
            file = open(params["path_to_model"] + "logfile.txt","a")
            file.write("Save Model from epoch: " + str(epoch+1) + "\n")
            file.close()



    with open(params["path_to_model"] + 'log_epoch_loss.csv', mode='w') as csv_file:
        fieldnames = ['epoch', 'epoch_loss', 'epoch_loss_all_classes', 'val_epoch_loss', 'val_epoch_loss_all_classes']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for ep in range(len(log_epoch_loss)):
            writer.writerow({'epoch': ep+1, 'epoch_loss': log_epoch_loss[ep][1], 'epoch_loss_all_classes': log_epoch_loss[ep][0], 'val_epoch_loss': log_epoch_loss_val[ep][1], 'val_epoch_loss_all_classes': log_epoch_loss_val[ep][0]})



### Testing ###


if b_pred:

    validation_data = CustomDataset(path = params["path_orig"],
                                        path_list = params["path_list"],
                                        score_list = params["score_list"],
                                        indices = params["val_indices"],
                                        binary = params["binary"],
                                        add_augm = False,
                                        augment = False)

    validation_dataloader = DataLoader(validation_data, batch_size=params["batch_size"], shuffle=False, num_workers=8)


    testing_data = CustomDataset(path = params["path_orig"],
                                        path_list = params["path_list"],
                                        score_list = params["score_list"],
                                        indices = params["test_indices"],
                                        binary = params["binary"],
                                        add_augm = False,
                                        augment = False)

    testing_dataloader = DataLoader(testing_data, batch_size=params["batch_size"], shuffle=False, num_workers=8)



    text1 = ["Validation Results: \n", "Test Results: \n"]
    text2 = ["+++ Validation +++ \n", "+++ Testing +++ \n"]
    batches = ["n_batches_val", "n_batches_test"]
    indi = ["val_indices", "test_indices"]
    loader = [validation_dataloader, testing_dataloader]

    for v in range(2):
        file = open(params["path_to_model"] + "logfile4.txt","a")
        file.write(text1[v])
        file.write("[File, Ground Truth, Prediction] \n")
        print(text2[v])
        model.load_state_dict(torch.load(params["path_to_model"] + "model.pt"))
        model.eval()

        #for i in range(params[batches[v]]):
        for i, batch in enumerate(loader[v]):
            percent = str(np.round((i+1)/params[batches[v]]*100, decimals = 2))
            if params["verbose"]:
                print("Progress: " + percent + "%  ", end='\r')

            x_raw, y_raw, path_to_dataset = batch

            y_pred = test(data_val_x = x_raw,
                            model = model)
            y_pred = y_pred.cpu()

            if params["regression"] == True:
                y_pred_lbl = np.around(y_pred.numpy(), decimals=0)
                y_pred_lbl[y_pred_lbl >= params["n_classes"]] = params["n_classes"]-1
                y_pred_lbl[y_pred_lbl < 0] = 0
                y_pred_lbl = y_pred_lbl.astype(int)
            elif params["ordinal"] == True:
                y_pred_lbl = y_pred.numpy()
                y_pred_lbl = (y_pred_lbl > 0.5).cumprod(axis=1).sum(axis=1)
                y_pred_lbl[y_pred_lbl>0] = y_pred_lbl[y_pred_lbl>0] - 1
            else:
                y_pred_lbl = torch.argmax(input=y_pred, dim=1)


            for j in range(len(y_raw)):
                y_pred_round = np.around(y_pred[j].numpy(), decimals=3)
                #print(y_pred_lbl[j])
                #print(y_pred_round)
                file.write("(" + path_to_dataset[j] + " " + str(y_raw[j].item()) + " " + str(y_pred_lbl[j].item()) + " " + str(y_pred_round) + ") ")

        file.write("\n")
        file.write("\n")
        print("\n")
