import torch
# from torch.utils.data import Dataset
import numpy as np


import torchvision.transforms
from torch.utils.data import WeightedRandomSampler

from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    RandAffined,
    CenterSpatialCropd,
    EnsureTyped,
    LambdaD,
    ScaleIntensityRanged, 
    RandFlipd, 
    RandScaleIntensityd, 
    RandShiftIntensityd, 
    RandGaussianNoised,
    Rand2DElasticd
)
import sklearn.utils
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional


from ra_utils.autoscora.autoscorRA_Pipeline.scoring.src.run_utils import (
    restructure_paths_and_scores,
    restructure_paths_and_scores_v2
)
import yaml
from importlib import resources
from tqdm import tqdm


from ra_utils.data.data_utils import (
    extract_extras_from_filename
)

#--------------------------------------------------------------#
#------------------------- Score reading ----------------------#
#--------------------------------------------------------------#

def read_tabular_data_and_paths(image_path_folder: str = "/home/cwatzenboeck/data/AutoPIX_local_data/autoscoRA_images/H_images_of_interest_2_renamed_mirrored_inverted_patches/", 
                                scores_df_path: str = "/home/cwatzenboeck/data/AutoPIX_local_data/tabular/autoscoRA_data/autoscoRA_hands.csv"):
    df = pd.read_csv(scores_df_path)

    images = Path(image_path_folder).glob("*.npy")
    image_paths = [str(image_path) for image_path in images]
    image_paths_df = pd.DataFrame(image_paths, columns=["image_path"])
    image_paths_df["filename_manual"] = image_paths_df["image_path"].apply(
        lambda x: "_".join(Path(x).stem.split("_")[:-1]))  # get the filename without the ROI
    image_paths_df["file_stem"] = image_paths_df["image_path"].apply(lambda x: Path(x).stem)
    image_paths_df["file_name"] = image_paths_df["image_path"].apply(lambda x: Path(x).name)


    # merge the two DataFrames on the common column
    merged_df = pd.merge(df, image_paths_df, left_on="filename_manual", right_on="filename_manual", how="inner")
    return merged_df



def exclude_ROIS_according_surgery_status(df):
    # TODO Understand why ES for feet is for joints not in range [0,10] but [0,5]
    # But whatever just continue... 

    df = df.copy()

    m_wrist_ES = df["chosen_score"] == "LunatE+RadiusE+ScaphE+TrapE+UlnaE"
    m_wrist_JSN = df["chosen_score"] == "Rad_Carp+Sca_Cap+Tra_Sca"
    m_other_ROI = ~m_wrist_ES & ~m_wrist_JSN

    # not sure if this is correct logic, but this is what Paul implemented 
    # Thought: What if only some wrist joints have surgery? Do all get 6?
    # Also what is for feet? 
    m_exclude = ((m_wrist_ES  & (df["score"] > 25))  |
                    (m_wrist_JSN & (df["score"] > 15))  |
                    (m_other_ROI & (df["score"] > 5)) )
    df_exclude = df[m_exclude]
    df_include = df[~m_exclude] 

    return df_include, df_exclude


def make_paths_dataframe(image_path_folder: str = "/home/cwatzenboeck/data/AutoPIX_local_data/autoscoRA_images/H_images_of_interest_2_renamed_mirrored_inverted_patches/", 
                         pattern = "*.npy"):

    images = Path(image_path_folder).glob(pattern)
    image_paths = [str(image_path) for image_path in images]
    image_paths_df = pd.DataFrame(image_paths, columns=["image_path"])
    image_paths_df["filename_manual"] = image_paths_df["image_path"].apply(
        lambda x: "_".join(Path(x).stem.split("_")[:-1]))  # get the filename without the ROI
    image_paths_df["file_stem"] = image_paths_df["image_path"].apply(lambda x: Path(x).stem)
    image_paths_df["file_name"] = image_paths_df["image_path"].apply(lambda x: Path(x).name)
    return image_paths_df


def df_scores_to_dct_list(df: pd.DataFrame) -> List[dict]:
    df = df.copy()
    df_tmp = pd.DataFrame(list(df["file_name"].apply(lambda x: extract_extras_from_filename(x, ending=".npy", replace_ending=False, filename_str="file_name"))))
    df = pd.merge(df, df_tmp, left_on="file_name", right_on="file_name")

    data = []
    for idx, row in df.iterrows():
        # Each dict is a sample, referencing row["patch"] plus additional metadata
        data.append({
            "img": row["image_path"],  # .npy file path on disk
            "file_name": row["file_name"],
            "score": row["score"],
            "score_type": row["chosen_score"],
            "JSN_or_ERO": row["chosen_score_type"],
            "extremity": row["extremity"],
            "patient_id": row["patient_id"], 
            "date_str": row["date_str"], 
            "left_or_right": row["left_or_right"],
        })
    return data



#--------------------------------------------------------------#
#-------------------------- Splits    -------------------------#
#--------------------------------------------------------------#

def split_training_val_test__on_patient_level(df_include, ratios=(0.6, 0.2, 0.2), random_state=42):
    assert sum(ratios) == 1.0, "Ratios must sum to 1.0"
    
    # Step 1: Get unique patient IDs
    unique_patients = df_include["patient_id"].unique()
    
    # Step 2: Compute sizes
    n_patients = len(unique_patients)
    n_train = int(ratios[0] * n_patients)
    n_val = int(ratios[1] * n_patients)
    
    # Step 3: Shuffle and split
    shuffled_patients = sklearn.utils.shuffle(unique_patients, random_state=random_state)
    train_patients = shuffled_patients[:n_train]
    val_patients = shuffled_patients[n_train:n_train + n_val]
    test_patients = shuffled_patients[n_train + n_val:]

    # Step 4: Filter the dataframe
    df_train = df_include[df_include["patient_id"].isin(train_patients)].reset_index(drop=True)
    df_val = df_include[df_include["patient_id"].isin(val_patients)].reset_index(drop=True)
    df_test = df_include[df_include["patient_id"].isin(test_patients)].reset_index(drop=True)

    return df_train, df_val, df_test



#--------------------------------------------------------------#
#------------------------- Transforms -------------------------#
#--------------------------------------------------------------#

def repeat_channels(x):
    return np.repeat(x, axis=0, repeats=3)

def min_max_normalize(img):
    """
    Custom function that rescales "img" into [0,1] using
    per-image min and max. Works for NumPy arrays or Tensors.
    """
    # Convert to torch.Tensor if not already
    if not isinstance(img, torch.Tensor):
        img = torch.as_tensor(img, dtype=torch.float32)

    min_val = torch.min(img)
    max_val = torch.max(img)
    # Avoid division by zero if the image is uniform
    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = torch.zeros_like(img)

    return img


def make_trainings_transforms(transforms_config): 

    transforms = [
        LoadImaged(keys=["img"], reader="NumpyReader"),
        EnsureChannelFirstd(keys=["img"]),
        LambdaD(keys=["img"], func=min_max_normalize)
    ]




    if transforms_config.get("RandAffined__bool", True):
        RandAffined__prob = transforms_config.get("RandAffined__prob", 0.5)
        RandAffined__rotation_range_degree = transforms_config.get("RandAffined__rotation_range_degree", 10)
        RandAffined__translate_range_pixels = transforms_config.get("RandAffined__translate_range_pixels", 19)
        RandAffined__scale_range = transforms_config.get("RandAffined__scale_range", 0.2)

        t = RandAffined(
            keys=["img"],
            prob=RandAffined__prob,  
            rotate_range=RandAffined__rotation_range_degree * np.pi / 180, 
            translate_range=RandAffined__translate_range_pixels, 
            scale_range=RandAffined__scale_range,
            padding_mode="zeros"
        )
        transforms.append(t)


    Rand2DElasticd__bool = transforms_config.get("Rand2DElasticd__bool", False)
    if Rand2DElasticd__bool:
        Rand2DElasticd__prob = transforms_config.get("Rand2DElasticd__prob", 0.2)
        Rand2DElasticd__spacing = transforms_config.get("Rand2DElasticd__spacing", 12)
        Rand2DElasticd__magnitude = transforms_config.get("Rand2DElasticd__magnitude", 2.73)
        Rand2DElasticd__shear_range = transforms_config.get("Rand2DElasticd__shear_range", None)
        t = Rand2DElasticd(
            keys=["img"],
            spacing=(Rand2DElasticd__spacing, Rand2DElasticd__spacing),
            magnitude_range=(-Rand2DElasticd__magnitude, Rand2DElasticd__magnitude),
            prob=Rand2DElasticd__prob,
            shear_range= None if Rand2DElasticd__shear_range == None else (Rand2DElasticd__shear_range, Rand2DElasticd__shear_range),
            rotate_range=0,
            scale_range=0,
            translate_range=0,
            padding_mode="zeros",
        )
        transforms.append(t)



    # ROI was previously enlarged. Starting from this dim
    t = CenterSpatialCropd(keys=["img"], roi_size=transforms_config["OUTPUT_DIM"])
    transforms.append(t)
    
    RandFlipd__bool = transforms_config.get("RandFlipd__bool", True)
    if RandFlipd__bool:
        RandFlipd__prob = transforms_config.get("RandFlipd__prob", 0.5)
        t = RandFlipd(keys=["img"], prob=RandFlipd__prob, spatial_axis=1)
        transforms.append(t)


    RandGaussianNoised__bool = transforms_config.get("RandGaussianNoised__bool", True)
    if RandGaussianNoised__bool:
        RandGaussianNoised__prob = transforms_config.get("RandGaussianNoised__prob", 0.17)
        RandGaussianNoised__std = transforms_config.get("RandGaussianNoised__std", 0.085)
        t = RandGaussianNoised(
            keys=["img"], 
            prob=RandGaussianNoised__prob, 
            mean=0.0, 
            std=RandGaussianNoised__std)
        transforms.append(t)
    


    # Create RGB like data for ResNet
    RBG_channels__bool = transforms_config.get("RBG_channels__bool", True)
    if RBG_channels__bool: 
        t = LambdaD(keys=["img"], func=repeat_channels)
        transforms.append(t)

        # For ResNet this is the expected input? 
        t = LambdaD(keys=["img"],
                func=torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        )
        transforms.append(t)


    return transforms


def make_validation_transforms(transforms_config = {"OUTPUT_DIM": (128, 128)}): 
    transforms = [
        LoadImaged(keys=["img"], reader="NumpyReader"),
        EnsureChannelFirstd(keys=["img"]),
        LambdaD(keys=["img"], func=min_max_normalize)
    ]

    # ROI was previously enlarged. Starting from this dim
    t = CenterSpatialCropd(keys=["img"], roi_size=transforms_config["OUTPUT_DIM"])
    transforms.append(t)
    
    # Create RGB like data for ResNet
    RBG_channels__bool = transforms_config.get("RBG_channels__bool", True)
    if RBG_channels__bool:
        t = LambdaD(keys=["img"], func=repeat_channels)
        transforms.append(t)

        # For ResNet this is the expected input? 
        t = LambdaD(keys=["img"],
                func=torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        )
        transforms.append(t)

    return transforms




#--------------------------------------------------------------#
#------------------------- Combined-  -------------------------#
#--------------------------------------------------------------#



def load_img_SHS_patch_data_OLD(data_config: dict):
    df_paths = make_paths_dataframe(
        image_path_folder = data_config["image_path_folder"], # "/home/cwatzenboeck/data/AutoPIX_local_data/dev_cw/ds_100/H_patches_size1.5/", 
        pattern = data_config.get("pattern", "*/*.npy")
    )
    
    chosen_score = data_config["scores"]
    if data_config.get("sum_wrist_points", False):
        # SUM WRISTS OPTION INCLUDED
        with resources.files("ra_utils.resources.scores_metadata").joinpath("score_abbreviations_info_dct.yml").open("r") as f:
            score_abbreviations_info_dct = yaml.safe_load(f)
        extremity = [score_abbreviations_info_dct[score]["extremity"] for score in chosen_score]
        score_type = [score_abbreviations_info_dct[score]["score_type"] for score in chosen_score]
        assert len(set(extremity)) == 1, "All scores must have the same extremity"
        assert len(set(score_type)) == 1, "All scores must have the same score type"
        chosen_score_type = score_type[0]
        extremity = extremity[0]
        paths_scores_df = restructure_paths_and_scores(chosen_score = chosen_score,
                                                      chosen_score_type = chosen_score_type,
                                                      extremity = extremity, 
                                                      score_path_H = data_config.get("score_path_H", ""), 
                                                      score_path_F = data_config.get("score_path_F", ""), 
                                                      sum_wrist=True)
    else: 
        # General option for now (no sum wrists)
        paths_scores_df = restructure_paths_and_scores_v2(
            chosen_scores = chosen_score,
            score_path_H = data_config.get("score_path_H", ""), 
            score_path_F = data_config.get("score_path_F", "")
            )

    df_include, df_exclude = exclude_ROIS_according_surgery_status(paths_scores_df)
    df_include["patient_id"] = df_include["file_name"].apply(lambda x: x.split("_")[0])


    # TODO lateron: 
    # seperate into two functions and load split ids from file instead of splitting here

    use_splits_file = data_config.get("use_splits_file", False)
    if use_splits_file: 
        src = data_config["splits_file"]
        print("reading ", src, "and do Tr, Val, Test  split accordingly")
        splits_file = pd.read_csv(src)  # contains columns: "patient_id", "set_type" where "set_type" = "Tr", "Val", "Ts"
        train_patients = splits_file["patient_id"][splits_file["set_type"] == "Tr"]
        val_patients = splits_file["patient_id"][splits_file["set_type"] == "Val"]
        test_patients = splits_file["patient_id"][splits_file["set_type"] == "Ts"]
        df_train = df_include[df_include["patient_id"].isin(train_patients)].reset_index(drop=True)
        df_val = df_include[df_include["patient_id"].isin(val_patients)].reset_index(drop=True)
        df_test = df_include[df_include["patient_id"].isin(test_patients)].reset_index(drop=True)

    else:
        print("Performing split ")
        split_ratios =  data_config.get("split_ratio")
        df_train, df_val, df_test = split_training_val_test__on_patient_level(df_include, ratios=split_ratios)
        df_train = pd.merge(df_paths[["image_path", "file_name"]], df_train, on="file_name")
        df_val   = pd.merge(df_paths[["image_path", "file_name"]], df_val, on="file_name")
        df_test  = pd.merge(df_paths[["image_path", "file_name"]], df_test, on="file_name")

    print(f"{split_ratios = }")
    print(f"{len(df_exclude) = }")
    print(f"{len(df_include) = }")
    print(f"{len(df_paths) = }")
    print()

    print(f"{len(df_train) = }")
    print(f"{len(df_val) = }")
    print(f"{len(df_test) = }")



    # data_list__train = df_scores_to_dct_list(df_train)
    # data_list__val = df_scores_to_dct_list(df_val)
    # data_list__test = df_scores_to_dct_list(df_test)
    
    data = dict(
        df_include = df_include, 
        df_exclude = df_exclude,
        df_train = df_train,
        df_val = df_val,
        df_test = df_test,
        # data_list__train = data_list__train,
        # data_list__val = data_list__val,
        # data_list__test = data_list__test
    )
    return data

# def load_img_SHS_patch_data__several(data_config: dict):
#     df_paths = make_paths_dataframe(
#         image_path_folder = data_config["image_path_folder"], # "/home/cwatzenboeck/data/AutoPIX_local_data/dev_cw/ds_100/H_patches_size1.5/", 
#         pattern = data_config.get("pattern", "*/*.npy")
#     )
    
#     chosen_scores = data_config["score_groups"]
#     data_dct = {}
#     for name, chosen_score in tqdm(chosen_scores.items(), desc="Processing chosen scores"):
#         paths_scores_df = restructure_paths_and_scores_v2(
#             chosen_scores = chosen_score,
#             score_path_H = data_config.get("score_path_H", ""), 
#             score_path_F = data_config.get("score_path_F", "")
#             )

#         df_include, df_exclude = exclude_ROIS_according_surgery_status(paths_scores_df)
#         df_include["patient_id"] = df_include["file_name"].apply(lambda x: x.split("_")[0])


#         # TODO lateron: 
#         # seperate into two functions and load split ids from file instead of splitting here
#         use_splits_file = data_config.get("use_splits_file", False)
#         if use_splits_file: 
#             src = data_config["splits_file"]
#             print("reading ", src, "and do Tr, Val, Test  split accordingly")
#             splits_file = pd.read_csv(src)  # contains columns: "patient_id", "set_type" where "set_type" = "Tr", "Val", "Ts"
#             train_patients = splits_file["patient_id"][splits_file["set_type"] == "Tr"]
#             val_patients = splits_file["patient_id"][splits_file["set_type"] == "Val"]
#             test_patients = splits_file["patient_id"][splits_file["set_type"] == "Ts"]
#             df_train = df_include[df_include["patient_id"].isin(train_patients)].reset_index(drop=True)
#             df_val = df_include[df_include["patient_id"].isin(val_patients)].reset_index(drop=True)
#             df_test = df_include[df_include["patient_id"].isin(test_patients)].reset_index(drop=True)

#         else:
#             # print("Performing split ")
#             split_ratios =  data_config.get("split_ratio")
#             df_train, df_val, df_test = split_training_val_test__on_patient_level(df_include, ratios=split_ratios)
#             df_train = pd.merge(df_paths[["image_path", "file_name"]], df_train, on="file_name")
#             df_val   = pd.merge(df_paths[["image_path", "file_name"]], df_val, on="file_name")
#             df_test  = pd.merge(df_paths[["image_path", "file_name"]], df_test, on="file_name")

        
#         data = dict(
#             df_include = df_include, 
#             df_exclude = df_exclude,
#             df_train = df_train,
#             df_val = df_val,
#             df_test = df_test,
#         )
#         data_dct[name] = data

#     return data



# def dataset_and_loader(data_tables, config):
#     """
#     Create dataset and dataloaders for training, validation and testing.
#     """

#     data_list__train = df_scores_to_dct_list(data_tables["df_train"])
#     data_list__val = df_scores_to_dct_list(data_tables["df_val"])
#     data_list__test = df_scores_to_dct_list(data_tables["df_test"])


#     transform_train = make_trainings_transforms(config["transforms"])
#     transform_train = Compose(transform_train)


#     transform_val = make_validation_transforms(config["transforms"])
#     transform_val = Compose(transform_val)

#     dataset_train = Dataset(data=data_list__train, transform=transform_train)
#     dataset_validation = Dataset(data=data_list__val, transform=transform_val)
#     dataset_test = Dataset(data=data_list__test, transform=transform_val)


#     use_WeightedRandomSampler = config["training"].get("use_WeightedRandomSampler", False)
#     if use_WeightedRandomSampler: 
#         # Suppose y is a list of labels for your training dataset
#         labels = [item['score'] for item in data_list__train]

#         class_count = torch.bincount(torch.tensor(labels))
#         class_weights = 1.0 / class_count.float()
#         sample_weights = [class_weights[label] for label in labels]

#         # Create the sampler
#         sampler = WeightedRandomSampler(weights=sample_weights,
#                                         num_samples=len(sample_weights),
#                                         replacement=True)

#     # Dataloaders
#     train_loader = DataLoader(
#         dataset_train,
#         batch_size=config["training"]["batch_size"],
#         shuffle=False if use_WeightedRandomSampler else True,
#         sampler=sampler if use_WeightedRandomSampler else None,
#         num_workers=config["training"]["num_workers"],
#         drop_last=True
#     )
#     val_loader = DataLoader(
#         dataset_validation,
#         batch_size=config["training"]["batch_size"],
#         shuffle=False,
#         num_workers=config["training"]["num_workers"],
#     )
#     test_loader = DataLoader(
#         dataset_test,
#         batch_size=config["training"]["batch_size"],
#         shuffle=False,
#         num_workers=config["training"]["num_workers"],
#     )

#     data = {
#         "train_loader": train_loader,
#         "val_loader": val_loader,
#         "test_loader": test_loader,
#         "dataset_train": dataset_train,
#         "dataset_validation": dataset_validation,
#         "dataset_test": dataset_test
#     }
#     return data



def process_several_score_groups(data_config: dict):
    df_paths = make_paths_dataframe(
        image_path_folder=data_config["image_path_folder"],
        pattern=data_config.get("pattern", "*/*.npy")
    )

    data_dct = {}
    for name, chosen_score in tqdm(data_config["score_groups"].items(), desc="Processing chosen scores"):
        print(f"Loading data for: ", chosen_score)
        data_dct[name] = process_single_score_group(chosen_score, df_paths, data_config)
    
    return data_dct




def process_single_score_group(chosen_score, df_paths, data_config):
    paths_scores_df = restructure_paths_and_scores_v2(
        chosen_scores=chosen_score,
        score_path_H=data_config.get("score_path_H", ""),
        score_path_F=data_config.get("score_path_F", "")
    )

    df_include, df_exclude = exclude_ROIS_according_surgery_status(paths_scores_df)
    df_include["patient_id"] = df_include["file_name"].apply(lambda x: x.split("_")[0])

    if data_config.get("use_splits_file", False):
        df_train, df_val, df_test = split_using_file(df_include, data_config["splits_file"])
    else:
        df_train, df_val, df_test = split_using_ratios(df_include, df_paths, data_config.get("split_ratio"))

    return {
        "df_include": df_include,
        "df_exclude": df_exclude,
        "df_train": df_train,
        "df_val": df_val,
        "df_test": df_test,
    }

def load_img_SHS_patch_data(data_config: dict):
    df_paths = make_paths_dataframe(
        image_path_folder=data_config["image_path_folder"],
        pattern=data_config.get("pattern", "*/*.npy")
    )
    chosen_scores = data_config["scores"]
    print("Loading data for: ", chosen_scores)
    data = process_single_score_group(chosen_scores, df_paths, data_config)
    return data



def split_using_file(df_include, splits_file_path):
    print(f"Reading splits from {splits_file_path} and splitting accordingly")
    splits = pd.read_csv(splits_file_path)

    train_patients = splits.query('set_type == "Tr"')["patient_id"]
    val_patients = splits.query('set_type == "Val"')["patient_id"]
    test_patients = splits.query('set_type == "Ts"')["patient_id"]

    df_train = df_include[df_include["patient_id"].isin(train_patients)].reset_index(drop=True)
    df_val = df_include[df_include["patient_id"].isin(val_patients)].reset_index(drop=True)
    df_test = df_include[df_include["patient_id"].isin(test_patients)].reset_index(drop=True)

    return df_train, df_val, df_test


def split_using_ratios(df_include, df_paths, split_ratios):
    df_train, df_val, df_test = split_training_val_test__on_patient_level(df_include, ratios=split_ratios)

    df_train = pd.merge(df_paths[["image_path", "file_name"]], df_train, on="file_name")
    df_val   = pd.merge(df_paths[["image_path", "file_name"]], df_val, on="file_name")
    df_test  = pd.merge(df_paths[["image_path", "file_name"]], df_test, on="file_name")

    return df_train, df_val, df_test


def dataset_and_loader(data_tables, config):
    """
    Create datasets and dataloaders for training, validation, and testing.
    """
    datasets = prepare_datasets(data_tables, config)
    loaders = prepare_dataloaders(datasets, config)
    
    return {**datasets, **loaders}


def prepare_datasets(data_tables, config):
    transform_train = Compose(make_trainings_transforms(config["transforms"]))
    transform_val = Compose(make_validation_transforms(config["transforms"]))

    return {
        "dataset_train": Dataset(
            data=df_scores_to_dct_list(data_tables["df_train"]),
            transform=transform_train
        ),
        "dataset_validation": Dataset(
            data=df_scores_to_dct_list(data_tables["df_val"]),
            transform=transform_val
        ),
        "dataset_test": Dataset(
            data=df_scores_to_dct_list(data_tables["df_test"]),
            transform=transform_val
        ),
    }


def prepare_dataloaders(datasets, config):
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    use_sampler = config["training"].get("use_WeightedRandomSampler", False)

    sampler = None
    if use_sampler:
        labels = [item['score'] for item in datasets["dataset_train"].data]
        class_count = torch.bincount(torch.tensor(labels))
        class_weights = 1.0 / class_count.float()
        sample_weights = [class_weights[label] for label in labels]

        sampler = WeightedRandomSampler(weights=sample_weights,
                                        num_samples=len(sample_weights),
                                        replacement=True)

    return {
        "train_loader": DataLoader(
            datasets["dataset_train"],
            batch_size=batch_size,
            shuffle=not use_sampler,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=True
        ),
        "val_loader": DataLoader(
            datasets["dataset_validation"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test_loader": DataLoader(
            datasets["dataset_test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }


def dataset_and_loader_several(data_tables_several, config):
    return_dct = {}
    for key, item in data_tables_several.items():
        data = dataset_and_loader(item, config)
        return_dct[key] = data
    return return_dct

#--------------------------------------------------------------#
#------------------------- Other      -------------------------#
#--------------------------------------------------------------#




