import numpy as np
import torch
import random
import os
from torch.utils.data import Dataset
from monai.transforms import Affine, Rand2DElastic, RandGaussianNoise

#from src.pixel_range2 import pixel_convert
from ra_utils.autoscora.autoscorRA_Pipeline.scoring.src.pixel_range2 import pixel_convert


def get_patient_nr(path_to_train_patients, path_to_op_patients):
    ### test patients ###
    patient_nr = []
    f = open(path_to_train_patients, "r")
    for line in f:
        i = line.index("_")
        if line[:i] not in patient_nr:
            patient_nr.append(line[:i])
    f.close()

    ### op patients ###
    op_patient_nr = []
    f = open(path_to_op_patients, "r")
    for line in f:
        #print(line[:-1])
        op_patient_nr.append(line[:-1])
    f.close()

    return patient_nr, op_patient_nr

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


def get_indices4(train_test_split, len_cur_path_list, path_list, score_list, train_pat, val_pat, test_pat, op_patients, chosen_score, 
                 path_to_all_patients_npy="src/all_patients_num.npy", 
                 path_to_model_dir=None,
                 return_patient_ids_instead=False):

    all_patients = np.load(path_to_all_patients_npy)

    train_patients = np.array([], dtype="int")
    val_patients = np.array([], dtype="int")
    test_patients = np.array([], dtype="int")
    remaining_patients = np.array([], dtype="int")

    train_indices = np.array([], dtype="int")
    val_indices = np.array([], dtype="int")
    test_indices = np.array([], dtype="int")
    remaining_indices = np.array([], dtype="int")

    n_train = int(len(all_patients) * train_test_split[0])
    n_val = int(len(all_patients) * train_test_split[1])
    n_test = len(all_patients) - n_train - n_val

    ### Assign patients accoriding to predefined sets (train_pat, val_pat) ###

    for ind, num in enumerate(all_patients):
        #undersc_ind = find(file, "_")
        #num = file[:undersc_ind[0]]
        #num_long = file[:undersc_ind[3]]
        #print(file)
        #print(file[:undersc_ind[0]])
        #if num_long not in op_patients:
        if num in train_pat:
            train_patients = np.append(train_patients, num)
        elif num in val_pat:
            val_patients = np.append(val_patients, num)
        elif num in test_pat:
            test_patients = np.append(test_patients, num)
        else:
            remaining_patients = np.append(remaining_patients, num)

    if n_train < len(train_patients):
        print("--> Number of patches in train set is higher than specified in the train/test split!!")
    if n_val < len(val_patients):
        print("--> Number of patches in validation set is higher than specified in the train/test split!!")

    ### Assign remaining patients to fill up train/val/test patient set ###

    diff_train = n_train - len(train_patients)
    diff_val = n_val - len(val_patients)
    train_patients = np.append(train_patients, remaining_patients[:diff_train])
    val_patients = np.append(val_patients, remaining_patients[diff_train: diff_train+diff_val])
    test_patients = np.append(test_patients, remaining_patients[diff_train+diff_val:])

    if return_patient_ids_instead: # Added by CW
        return dict(
            train_patients = list(train_patients), 
            val_patients = list(val_patients),
            test_patients = list(test_patients)
        )

    ### Assign indices to train/val/test index set according to train/val/test/set ###
    ### If patient is not an OP patient                                            ###
    ### And Filter out scoring values higher than 5 (or 15/25 for sum score)       ###

    threshold = 5
    if chosen_score == ["LunatE", "RadiusE", "ScaphE", "TrapE", "UlnaE"]:
        threshold = 25
    if chosen_score == ["Rad_Carp", "Sca_Cap", "Tra_Sca"]:
        threshold = 15

    for ind, file in enumerate(path_list):
        undersc_ind = find(file, "_")
        num = file[:undersc_ind[0]]
        num_long = file[:undersc_ind[3]]

        if num_long not in op_patients:
            if score_list[ind] <= threshold:
                if num in train_patients:
                     train_indices = np.append(train_indices, ind)
                if num in val_patients:
                    val_indices = np.append(val_indices, ind)
                if num in test_patients:
                    test_indices = np.append(test_indices, ind)

    if path_to_model_dir != None:
        np_path_list = np.array(path_list)
        np.save(f"{path_to_model_dir}/train_files_f.npy", np_path_list[train_indices])
        np.save(f"{path_to_model_dir}/val_files_f.npy", np_path_list[val_indices])
        np.save(f"{path_to_model_dir}/test_files_f.npy", np_path_list[test_indices])
        print(f"train/val/test patient split saved to {path_to_model_dir}!")

    return train_indices, val_indices, test_indices, score_list, path_list




def get_indices3(train_test_split, len_cur_path_list, path_list, score_list, patient_nr, op_patient_nr):
    #print("op_patient_nr: ", op_patient_nr[:10])
    all_patients = np.load("src/list_all_patients.npy")
    train_pat = np.array(patient_nr)
    val_pat = np.array([])
    test_pat = ([])
    print("patient_nr:")
    print(train_pat)
    print("all_patients:")
    print(all_patients[:10])

    n_train = int(len(all_patients) * train_test_split[0])
    n_val = int(len(all_patients) * train_test_split[1])
    n_test = len(all_patients) - n_train - n_val

    #print(all_patients[:10])
    for i in range(len(all_patients)):
        if all_patients[i] not in train_pat:
            if len(train_pat) < n_train:
                train_pat = np.append(train_pat, all_patients[i])
            elif len(val_pat) < n_val:
                val_pat = np.append(val_pat, all_patients[i])
            elif len(test_pat) < n_test:
                test_pat = np.append(test_pat, all_patients[i])
            else:
                print("upsi... to many patients: ", all_patients[i])

    #print(len(train_pat))
    #print("train_pat: ", train_pat)
    #print(len(val_pat))
    #print("val_pat: ", val_pat)
    #print(len(test_pat))
    #print("test_pat: ", test_pat)
    train_indices = np.array([], dtype="int")
    val_indices = np.array([], dtype="int")
    test_indices = np.array([], dtype="int")

    for ind in range(len(path_list)):
        undersc_ind = find(path_list[ind], "_")
        #print(path_list[ind][:undersc_ind[3]])
        if path_list[ind][:undersc_ind[3]] not in op_patient_nr:
            if path_list[ind][:undersc_ind[0]] in train_pat:
                train_indices = np.append(train_indices, ind)
            elif path_list[ind][:undersc_ind[0]] in val_pat:
                val_indices = np.append(val_indices, ind)
            elif path_list[ind][:undersc_ind[0]] in test_pat:
                test_indices = np.append(test_indices, ind)
            else:
                print("upsi... some patient not found:" + path_list[ind])
    #print(len(train_indices))
    #print(len(val_indices))
    #print(len(test_indices))

    return train_indices, val_indices, test_indices, score_list, path_list


class CustomDataset(Dataset):
    def __init__(self, path, path_list, score_list, indices, binary, add_augm, augment, aug_params=None):
        self.path = path
        self.path_list = path_list
        self.score_list = score_list
        self.indices = indices
        self.binary = binary
        self.add_augm = add_augm
        self.augment = augment

        if self.augment == True:
            self.deform = Rand2DElastic(
                prob=aug_params["prob"],
                spacing=(aug_params["spacing"], aug_params["spacing"]),
                magnitude_range=(-aug_params["magnitude"], aug_params["magnitude"]),
                rotate_range=(-aug_params["rotate"], aug_params["rotate"]),
                scale_range=(aug_params["scale"], aug_params["scale"]),
                shear_range=(aug_params["shear"], aug_params["shear"]),
                translate_range=(aug_params["translate"], aug_params["translate"]),
                padding_mode="zeros",
                #as_tensor_output=True
                )


    def __len__(self):
        return(len(self.indices))

    def __getitem__(self, index):
        if self.add_augm == True:
            num = random.randint(0, 53)
            file = self.path_list[self.indices[index]][:-4] + "_augm" + str(num) + ".npy"
        else:
            file = self.path_list[self.indices[index]]
        #path_to_dataset = [file]
        x_batch = np.load(self.path + file)
        x_batch = pixel_convert(x_batch[np.newaxis,...], clip_bottom=0)
        if self.augment == True:
            x_batch = self.deform(x_batch, mode="bilinear")
        x_batch = np.repeat(x_batch[:, ...], 3, axis=0)
        y_batch = int(self.score_list[self.indices[index]])

        return x_batch, y_batch, file

def preprocess(data, aug_params):

    deform = Rand2DElastic(
        prob=aug_params["prob"],
        spacing=(aug_params["spacing"], aug_params["spacing"]),
        magnitude_range=(-aug_params["magnitude"], aug_params["magnitude"]),
        rotate_range=(-aug_params["rotate"], aug_params["rotate"]),
        scale_range=(aug_params["scale"], aug_params["scale"]),
        shear_range=(aug_params["shear"], aug_params["shear"]),
        translate_range=(aug_params["translate"], aug_params["translate"]),
        padding_mode="zeros",
        as_tensor_output=True,
        device=device,
    )

    """
    deform = Rand2DElastic(
        prob=0.5,
        spacing=(30, 30),
        magnitude_range=(-2, 2),
        rotate_range=(-np.pi / 8, np.pi / 8),
        scale_range=(0.2, 0.2),
        shear_range=(.1,.1),
        translate_range=(10, 10),
        padding_mode="zeros",
        as_tensor_output=True,
        device=device,
    )
    """
    noise = RandGaussianNoise(
        prob = aug_params["gaussia_prob"], # 0.5
        mean = aug_params["gaussia_mean"], # 0.0
        std = aug_params["gaussia_std"] # 0.05
    )

    for i in range(len(data)):
        data[i] = deform(data[i], mode="bilinear")
    return data

def load2(path, path_list, score_list, indices, index, batch_size, binary, add_augm):
    if add_augm == True:
        num = random.randint(0, 53)
        file = path_list[indices[index * batch_size]][:-4] + "_augm" + str(num) + ".npy"
    else:
        file = path_list[indices[index * batch_size]]
    path_to_dataset = [file]
    x_batch = np.load(path + file)
    x_batch = pixel_convert(x_batch[np.newaxis,...], clip_bottom=0)
    y_batch = [int(score_list[indices[index * batch_size]])]

    for i in np.arange(1, batch_size):
        if (index * batch_size + i) < len(indices):
            if add_augm == True:
                num = random.randint(0, 53)
                file = path_list[indices[index * batch_size + i]][:-4] + "_augm" + str(num) + ".npy"
            else:
                file = path_list[indices[index * batch_size + i]]
            path_to_dataset.append(file)
            img = np.load(path + file)
            img = pixel_convert(img[np.newaxis,...], clip_bottom=0)
            x_batch = np.concatenate([x_batch, img], axis = 0)
            y_batch.append(int(score_list[indices[index * batch_size + i]]))

    x_batch = np.repeat(x_batch[:, np.newaxis,...], 3, axis=1)

    if binary == 1:
        y_batch = np.array(y_batch)
        y_batch[y_batch != 0] = 1
        #print("y_batch in load: ", y_batch)
    if binary == 2:
        y_batch = np.array(y_batch)
        y_batch[y_batch == 1] = 0
        y_batch[y_batch != 0] = 1

    return torch.Tensor(x_batch.astype(np.float32)), torch.Tensor(y_batch).long(), path_to_dataset


def load(path, path_list, score_list, indices, index, batch_size, binary):

    path_to_dataset = [path_list[indices[index * batch_size]]]
    print(path_list[indices[index * batch_size]])

    x_batch = np.load(path + path_list[indices[index * batch_size]])
    x_batch = pixel_convert(x_batch[np.newaxis,...], clip_bottom=0)
    y_batch = [int(score_list[indices[index * batch_size]])]

    for i in np.arange(1, batch_size):
        if (index * batch_size + i) < len(indices):
            path_to_dataset.append(path_list[indices[index * batch_size + i]])
            img = np.load(path + path_list[indices[index * batch_size + i]])
            img = pixel_convert(img[np.newaxis,...], clip_bottom=0)
            x_batch = np.concatenate([x_batch, img], axis = 0)
            y_batch.append(int(score_list[indices[index * batch_size + i]]))

    x_batch = np.repeat(x_batch[:, np.newaxis,...], 3, axis=1)

    if binary == 1:
        y_batch = np.array(y_batch)
        y_batch[y_batch != 0] = 1
        #print("y_batch in load: ", y_batch)
    if binary == 2:
        y_batch = np.array(y_batch)
        y_batch[y_batch == 1] = 0
        y_batch[y_batch != 0] = 1

    return torch.Tensor(x_batch.astype(np.float32)), torch.Tensor(y_batch).long(), path_to_dataset


def balance_train_set(train_indices, score_list, balance, binary):
    print("Balancing training set:")
    unique, counts = np.unique(score_list[train_indices], return_counts=True)
    print(unique, counts)

    max_ind = np.argmax(counts)

    for i in range(len(unique)):
        if i != max_ind:
            ind_entries = [train_indices[j] for j, ee in enumerate(score_list[train_indices]) if ee == unique[i]]
            if binary == 0 or binary == 2:
                multiplier = (counts[max_ind]/counts[i] * balance) - 1
            elif binary == 1:
                multiplier = (counts[max_ind]/counts[i]/(len(counts)-1) * balance) - 1
            train_indices = np.concatenate([train_indices, np.array(int(np.floor(multiplier)) * ind_entries, dtype="int")])

    new = np.array(score_list)[train_indices]
    unique, counts = np.unique(new, return_counts=True)
    print(unique, counts)
    print("\n")

    np.random.shuffle(train_indices)

    return train_indices


def balanced_mean(x,y):
    return np.sum(x * y / np.sum(y))
