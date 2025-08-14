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

from importlib import resources
import pandas as pd
import json
from  ra_utils.data.shap_sums import limit_treatment_number

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



def exclude_ROIS_according_surgery_status(df, surgery_patientids_list=[]):
    # surgery_patientids_list  ... extra list op patients with surgery which should be excluded I guess
    
    
    # TODO Understand why ES for feet is for joints not in range [0,10] but [0,5]
    # But whatever just continue... 

    df = df.copy()

    m_wrist_ES = df["chosen_score"] == "LunatE+RadiusE+ScaphE+TrapE+UlnaE"
    m_wrist_JSN = df["chosen_score"] == "Rad_Carp+Sca_Cap+Tra_Sca"
    m_other_ROI = ~m_wrist_ES & ~m_wrist_JSN
    m_surgery_according_to_list = df["image_instance_id"].isin(surgery_patientids_list)

    # not sure if this is correct logic, but this is what Paul implemented 
    # Thought: What if only some wrist joints have surgery? Do all get 6?
    # Also what is for feet? 
    m_exclude = ((m_wrist_ES  & (df["score"] > 25))  |
                    (m_wrist_JSN & (df["score"] > 15))  |
                    (m_other_ROI & (df["score"] > 5)) |
                    (m_surgery_according_to_list)
                    ) 
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


# def df_scores_to_dct_list(df: pd.DataFrame) -> List[dict]:
#     # moved to before split ... df_include
#     # df = df.copy()
#     # df_tmp = pd.DataFrame(list(df["file_name"].apply(lambda x: extract_extras_from_filename(x, ending=".npy", replace_ending=False, filename_str="file_name"))))
#     # df_tmp.drop(columns=["file_name"], inplace=True)
#     # df = pd.merge(df, df_tmp, left_index=True, right_index=True, how="left")

#     data = []
#     for idx, row in df.iterrows():
#         # Each dict is a sample, referencing row["patch"] plus additional metadata
#         data.append({
#             "img": row["image_path"],  # .npy file path on disk
#             "file_name": row["file_name"],
#             "score": row["score"],
#             "score_type": row["chosen_score"],
#             "JSN_or_ERO": row["chosen_score_type"],
#             "extremity": row["extremity"],
#             "patient_id": row["patient_id"], 
#             "date_str": row["date_str"], 
#             "left_or_right": row["left_or_right"],
#             "roi_name": row["roi_name"]
#         })
#     return data



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


def make_trainings_transforms(transforms_config, DIM=2): 

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
    out_dim = transforms_config["OUTPUT_DIM"]
    if isinstance(out_dim, int): #make 2d 
        out_dim = [out_dim]*DIM
    t = CenterSpatialCropd(keys=["img"], roi_size=out_dim)
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


def make_validation_transforms(transforms_config = {"OUTPUT_DIM": (128, 128)}, DIM=2): 
    transforms = [
        LoadImaged(keys=["img"], reader="NumpyReader"),
        EnsureChannelFirstd(keys=["img"]),
        LambdaD(keys=["img"], func=min_max_normalize)
    ]

    # ROI was previously enlarged. Starting from this dim
    out_dim = transforms_config["OUTPUT_DIM"]
    if isinstance(out_dim, int): #make 2d 
        out_dim = [out_dim]*DIM
    t = CenterSpatialCropd(keys=["img"], roi_size=out_dim)
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
    image_folders = data_config["image_path_folder"]
    if isinstance(image_folders, str):
        image_folders = [image_folders]
    df_paths_list = []
    for image_path_folder in image_folders:
        print(f"Loading paths from: {image_path_folder}")
        df_paths = make_paths_dataframe(
            image_path_folder=image_path_folder,
            pattern=data_config.get("pattern", "*/*.npy")
        )
        df_paths_list.append(df_paths)
    df_paths = pd.concat(df_paths_list, ignore_index=True)


    data_dct = {}
    for name, chosen_score in tqdm(data_config["score_groups"].items(), desc="Processing chosen scores"):
        print(f"Loading data for: ", chosen_score)
        data_dct[name] = process_single_score_group(chosen_score, df_paths, data_config)
    
    return data_dct

from typing import Literal
def add_patient_class_weights_and_labels(df_input, 
                                            rule: Literal[None, "mean_round_delta_bin", "median_score", "delta_range", "delta_buckets", "mean_ceil", "mean_round"] = None, 
                                            agg_cols = ["patient_id", "left_or_right", "chosen_score"], n_buckets=4):

    if rule in (None, "", "none", "None", False):
        # nothing to do
        return df_input
    else:
        

        # ---------- derive the label per patient/side/score_type ----------
        def _clean_stat(series, fn):
            """Apply fn to non-NaNs; keep NaN if the group is empty."""
            clean = series.dropna()
            return np.nan if len(clean) == 0 else fn(clean)

        if rule == "median_score":
            patient_lbl_df = (
                df_input
                .groupby(agg_cols)["score"]
                .agg(lambda x: _clean_stat(x, pd.Series.median))
                .reset_index(name="patient_cls_lbl")
            )

        elif rule == "delta_range":
            patient_lbl_df = (
                df_input
                .groupby(agg_cols)["score"]
                .agg(lambda x: _clean_stat(x, lambda s: s.max() - s.min()))
                .reset_index(name="patient_cls_lbl")
            )

        elif rule == "delta_buckets":
            tmp = (
                df_input
                .groupby(agg_cols)["score"]
                .agg(lambda x: _clean_stat(x, lambda s: float(np.std(s, ddof=0))))
                .reset_index()
                .rename(columns={"score": "std_val"})
            )

            # bucket the std into `n_buckets` roughly-equal-frequency bins
            tmp["patient_cls_lbl"] = pd.qcut(
                tmp["std_val"],
                q=n_buckets,
                labels=False,
                duplicates="drop"          # keeps it robust when many std==0
            )

            patient_lbl_df = tmp.drop(columns="std_val")

        elif rule == "delta_bin":
            tmp = (
                df_input
                .groupby(agg_cols)["score"]
                .agg(lambda x: _clean_stat(x, lambda s: int(np.std(s, ddof=0) > 1.0e-5)  ))
                .reset_index()
                .rename(columns={"score": "std_val"})
            )

            # bucket the std into `n_buckets` roughly-equal-frequency bins
            tmp["patient_cls_lbl"] = pd.qcut(
                tmp["std_val"],
                q=n_buckets,
                labels=False,
                duplicates="drop"          # keeps it robust when many std==0
            )

            patient_lbl_df = tmp.drop(columns="std_val")


        elif rule == "mean_round":
            patient_lbl_df = (
                df_input
                .groupby(agg_cols)["score"]
                .agg(lambda x: _clean_stat(x, lambda s: np.round(s.mean())))
                .reset_index(name="patient_cls_lbl")
            )

        elif rule == "mean_round_delta_bin":
            patient_lbl_df = (
                df_input
                .groupby(agg_cols)["score"]
                .agg(
                    lambda x: _clean_stat(
                        x,
                        lambda s: f"{int(np.round(s.mean()))}_{int(np.std(s, ddof=0) > 1.0e-5)}"
                    )
                )
                .reset_index(name="patient_cls_lbl")
            )
        # ---------

        elif rule == "mean_ceil":
            patient_lbl_df = (
                df_input
                .groupby(agg_cols)["score"]
                .agg(lambda x: _clean_stat(x, lambda s: np.ceil(s.mean())))
                .reset_index(name="patient_cls_lbl")
            )

        # ---------------------------------------------------------------
        else:
            raise ValueError(
                f"Unknown patient_level_class_balance_aggregation_rule: {rule}"
            )

        col = patient_lbl_df["patient_cls_lbl"]

        if pd.api.types.is_float_dtype(col):
            # only if still float AND every non-NaN value is an integer numerically
            if (col.dropna() % 1 == 0).all():
                patient_lbl_df["patient_cls_lbl"] = col.astype(pd.Int64Dtype())
        # else: column is already an (Int64/Int32/…) integer dtype → leave as-is


        # ---------- optional class weights (1 / class frequency) -------
        class_counts  = patient_lbl_df["patient_cls_lbl"].value_counts(dropna=True)
        class_weights = (1.0 / class_counts).to_dict()
        patient_lbl_df["patient_weight"] = (
            patient_lbl_df["patient_cls_lbl"].map(class_weights).fillna(0.0)
        )

        # ---------- attach to every image row -------------------------
        df_input = df_input.merge(patient_lbl_df, on=agg_cols, how="left")
        return df_input
    # ------------------------------------------------------------------


def process_single_score_group(chosen_score, df_paths, data_config):
    paths_scores_df = restructure_paths_and_scores_v2(
        chosen_scores=chosen_score,
        score_path_H=data_config.get("score_path_H", ""),
        score_path_F=data_config.get("score_path_F", "")
    )


    with resources.files("ra_utils.resources.scores_metadata").joinpath("roi_scores_matching.csv") as f:
        df_scores_meta = pd.read_csv(f)
        limits_dct = df_scores_meta[["score_name", "limit"]].set_index("score_name").to_dict()["limit"]

    how_to_deal_with_surgery = data_config.get("how_to_deal_with_surgery", "keep: map over limit to limit")
    verbose = data_config.get("verbose", True)
    if verbose:
        print(f"how_to_deal_with_surgery: {how_to_deal_with_surgery}")

    # options for how_to_deal_with_surgery:
    # - "exclude"
    # - "keep as is"
    # - "keep: map over limit to limit plus one"
    # - "keep: map over limit to limit"
    

    if how_to_deal_with_surgery == "exclude":
        paths_scores_df["image_instance_id"] = paths_scores_df["file_name"].apply(lambda x: "_".join(x.split("_")[:4]))
        surgery_patientids_list_path_json = data_config.get("surgery_patientids_list_path_json")
        surgery_patientids_list = []
        if surgery_patientids_list_path_json: 
            with open(surgery_patientids_list_path_json, "r") as f:
                surgery_patientids_list = json.load(f)
        df_include, df_exclude = exclude_ROIS_according_surgery_status(paths_scores_df, surgery_patientids_list=surgery_patientids_list)

    elif how_to_deal_with_surgery == "keep as is":
        df_include = paths_scores_df
        df_exclude = pd.DataFrame(columns=paths_scores_df.columns)

    elif how_to_deal_with_surgery == "keep: map over limit to limit plus one":
        df_exclude = pd.DataFrame(columns=paths_scores_df.columns)
        df_include = paths_scores_df
        def foo(row):
            t = row["chosen_score"]
            limit = limits_dct[t]
            s = row["score"]
            return limit_treatment_number(s, limit, limit_treatment="over_limit_to_limit_plus_1")
        df_include["score"] = df_include.apply(foo, axis=1)
    
    elif how_to_deal_with_surgery == "keep: map over limit to limit":
        df_exclude = pd.DataFrame(columns=paths_scores_df.columns)
        df_include = paths_scores_df
        def foo(row):
            t = row["chosen_score"]
            limit = limits_dct[t]
            s = row["score"]
            return limit_treatment_number(s, limit, limit_treatment="over_limit_to_limit")
        df_include["score"] = df_include.apply(foo, axis=1)
    else:
        raise ValueError(f"Unknown how_to_deal_with_surgery: {how_to_deal_with_surgery}")


    df_include["patient_id"] = df_include["file_name"].apply(lambda x: x.split("_")[0])

    # Add other infos at this point. Maybe important for sampling weights: 
    df = df_include.copy()
    # df_tmp = pd.DataFrame(list(df["file_name"].apply(lambda x: extract_extras_from_filename(x, ending=".npy", replace_ending=False, filename_str="file_name"))))
    # df_tmp.drop(columns=["file_name"], inplace=True)
    # df_include = pd.merge(df, df_tmp, left_index=True, right_index=True, how="left")
    df_tmp = (
        df_include["file_name"]
        .apply(extract_extras_from_filename)
        .apply(pd.Series)                    # explode dict → columns
    ).drop_duplicates()
    # keep file_name so the merge key is explicit
    df_include = df.merge(df_tmp, on="file_name", how="left")





    if data_config.get("use_splits_file", False):
        df_train, df_val, df_test, df_missed = split_using_file(df_include, data_config["splits_file"])
    else:
        df_train, df_val, df_test = split_training_val_test__on_patient_level(df_include, ratios=data_config["split_ratio"])
        df_missed = None

    df_train = pd.merge(df_paths[["image_path", "file_name"]], df_train, on="file_name")
    df_val   = pd.merge(df_paths[["image_path", "file_name"]], df_val, on="file_name")
    df_test  = pd.merge(df_paths[["image_path", "file_name"]], df_test, on="file_name")

    rule = data_config.get("patient_level_class_balance_aggregation_rule")
    n_buckets = data_config.get("delta_buckets_k", 4)    # only for delta_buckets
    df_train = add_patient_class_weights_and_labels(df_train, rule = rule, n_buckets=n_buckets)
    df_val   = add_patient_class_weights_and_labels(df_val,   rule = rule, n_buckets=n_buckets)
    df_test  = add_patient_class_weights_and_labels(df_test,  rule = rule, n_buckets=n_buckets)




    return {
        "df_include": df_include,
        "df_exclude": df_exclude,
        "df_train": df_train,
        "df_val": df_val,
        "df_test": df_test,
        "df_included_missed": df_missed
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

    train_patients = splits.query('set_type == "Tr"')["patient_id"].astype(int)
    val_patients = splits.query('set_type == "Val"')["patient_id"].astype(int)
    test_patients = splits.query('set_type == "Ts"')["patient_id"].astype(int)

    m_Tr = df_include["patient_id"].astype(int).isin(train_patients)
    m_Val = df_include["patient_id"].astype(int).isin(val_patients)
    m_Ts = df_include["patient_id"].astype(int).isin(test_patients)
    m_other = ~(m_Tr | m_Val | m_Ts)

    df_train = df_include[m_Tr].reset_index(drop=True)
    df_val   = df_include[m_Val].reset_index(drop=True)
    df_test  = df_include[m_Ts].reset_index(drop=True)
    
    df_missed = df_include[m_other].reset_index(drop=True)

    return df_train, df_val, df_test, df_missed


# def split_using_ratios(df_include, df_paths, split_ratios):
#     df_train, df_val, df_test = split_training_val_test__on_patient_level(df_include, ratios=split_ratios)

#     df_train = pd.merge(df_paths[["image_path", "file_name"]], df_train, on="file_name")
#     df_val   = pd.merge(df_paths[["image_path", "file_name"]], df_val, on="file_name")
#     df_test  = pd.merge(df_paths[["image_path", "file_name"]], df_test, on="file_name")

#     return df_train, df_val, df_test


#from ra_utils.data.datasampler_CR_patches import PatientBatchSampler

def df_scores_to_dct_list(df: pd.DataFrame) -> List[dict]:
    data = []
    for idx, row in df.iterrows():
        # Each dict is a sample, referencing row["patch"] plus additional metadata
        d = {
            "img": row["image_path"],  # .npy file path on disk
            "file_name": row["file_name"],
            "score": row["score"],
            "score_type": row["chosen_score"],
            "JSN_or_ERO": row["chosen_score_type"],
            "extremity": row["extremity"],
            "patient_id": row["patient_id"], 
            "date_str": row["date_str"], 
            "left_or_right": row["left_or_right"],
            "roi_name": row["roi_name"],
            "patient_scoretype_key":  f"{row['extremity']}_{row['left_or_right']}_{row['roi_name']}_{row['patient_id']}_{row['chosen_score']}" # everything except the time
        }
        if "patient_cls_lbl" in row.keys():
            d["patient_cls_lbl"] = row["patient_cls_lbl"]
        if "patient_weight" in row.keys():
            d["patient_weight"] = row["patient_weight"]
        data.append(d)

    return data



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
        "dataset_train_with_val_transforms": Dataset(
            data=df_scores_to_dct_list(data_tables["df_train"]),
            transform=transform_val
        ),
    }


# ---------------------------------------------------------------------
#  Samplers
# ---------------------------------------------------------------------
from collections import defaultdict
import random, math
from torch.utils.data import Sampler, DataLoader, WeightedRandomSampler
import torch


class PatientBatchSampler(Sampler):
    """
    Groups indices by patient and yields mini-batches that
    (almost) always contain a single patient.
    """

    def __init__(self, patient_ids, batch_size, drop_last=False):
        self.batch_size = batch_size
        self.drop_last  = drop_last

        self.patient2idx = defaultdict(list)
        for idx, pid in enumerate(patient_ids):
            self.patient2idx[pid].append(idx)

    # --------------------------------------------------
    def _epoch_batches(self):
        # shuffle within every patient
        for bucket in self.patient2idx.values():
            random.shuffle(bucket)

        patient_order = list(self.patient2idx.keys())
        random.shuffle(patient_order)

        for pid in patient_order:
            bucket = self.patient2idx[pid]
            for i in range(0, len(bucket), self.batch_size):
                chunk = bucket[i : i + self.batch_size]
                if len(chunk) == self.batch_size or not self.drop_last:
                    yield chunk

    # --------------------------------------------------
    def __iter__(self):
        yield from self._epoch_batches()

    def __len__(self):
        n = sum(len(v) for v in self.patient2idx.values())
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)


# class WeightedPatientBatchSampler(Sampler):
#     """
#     Same as PatientBatchSampler but draws patients with replacement
#     according to a weight per patient_id.
#     """

#     def __init__(self, dataset, patient_weights, batch_size, drop_last=False):
#         self.dataset      = dataset
#         self.batch_size   = batch_size
#         self.drop_last    = drop_last

#         # patient → list[idx]
#         self.patient2idx  = defaultdict(list)
#         for idx, item in enumerate(dataset.data):
#             self.patient2idx[item["patient_id"]].append(idx)

#         self.patient_ids  = list(self.patient2idx.keys())
#         self.weights      = [patient_weights.get(pid, 1.0) for pid in self.patient_ids]

#     # --------------------------------------------------
#     def __iter__(self):
#         # shuffle inside each bucket every epoch
#         buckets = {p: idxs.copy() for p, idxs in self.patient2idx.items()}
#         for idxs in buckets.values():
#             random.shuffle(idxs)

#         while any(buckets.values()):
#             pid = random.choices(self.patient_ids, weights=self.weights, k=1)[0]
#             if not buckets[pid]:
#                 continue
#             chunk = buckets[pid][: self.batch_size]
#             del buckets[pid][: len(chunk)]
#             if len(chunk) == self.batch_size or not self.drop_last:
#                 yield chunk

#     # --------------------------------------------------
#     def __len__(self):
#         n = sum(len(v) for v in self.patient2idx.values())
#         return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

# utils.py  (or wherever your samplers live)
from collections import defaultdict
import random, math
from torch.utils.data import Sampler
import numpy as np


class WeightedPatientBatchSampler(Sampler):
    """
    • Draws patients with replacement according to `patient_weights`.
    • Fills a batch with as many images of that patient as possible,
      but tops up with other patients so the batch has `batch_size`
      elements (except possibly the final batch if drop_last=False).
    """

    def __init__(self, dataset, patient_weights, batch_size, drop_last=False):
        self.dataset    = dataset
        self.batch_size = batch_size
        self.drop_last  = drop_last

        # patient_id → [idx, idx, …]
        self.patient2idx = defaultdict(list)
        for idx, item in enumerate(dataset.data):
            self.patient2idx[item["patient_id"]].append(idx)

        self.patient_ids = list(self.patient2idx.keys())
        self.weights     = np.array(
            [max(patient_weights.get(pid, 1.0), 0.0) for pid in self.patient_ids],
            dtype=float
        )
        # guarantee the weights sum > 0
        if self.weights.sum() == 0:
            self.weights[:] = 1.0

    # ------------------------------------------------------------------
    def __iter__(self):
        # fresh copy & shuffle inside each patient every epoch
        buckets = {p: idxs.copy() for p, idxs in self.patient2idx.items()}
        for idxs in buckets.values():
            random.shuffle(idxs)

        current_batch = []

        while any(buckets.values()):
            # (re)fill batch until we hit batch_size
            if len(current_batch) < self.batch_size:
                # sample a patient with replacement, weighted
                pid = random.choices(self.patient_ids, weights=self.weights, k=1)[0]
                if not buckets[pid]:
                    continue  # this patient is empty; pick again

                n_take = min(len(buckets[pid]),
                             self.batch_size - len(current_batch))
                current_batch.extend(buckets[pid][: n_take])
                del buckets[pid][: n_take]

            # emit if full
            if len(current_batch) == self.batch_size:
                yield current_batch
                current_batch = []

        # leftovers
        if current_batch and not self.drop_last:
            yield current_batch

    # ------------------------------------------------------------------
    def __len__(self):
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)



# ---------------------------------------------------------------------
#  Data-loader factory
# ---------------------------------------------------------------------
def prepare_dataloaders(datasets, config):
    tr_cfg       = config["training"]
    batch_size   = tr_cfg["batch_size"]
    num_workers  = tr_cfg["num_workers"]

    use_img_sampler        = tr_cfg.get("use_WeightedRandomSampler", False)
    use_patient_sampler    = tr_cfg.get("use_PatientBatchSampler", False)
    use_patient_w_sampler  = tr_cfg.get("use_PatientLevelWeightedRandomSampler", False)

    # only one of the three modes may be active
    assert sum([use_img_sampler,
                use_patient_sampler,
                use_patient_w_sampler]) <= 1, "Choose exactly one sampling mode."

    # ---------- classic image-level weighted sampler ----------
    if use_img_sampler:
        labels         = [item['score'] for item in datasets["dataset_train"].data]
        class_count    = torch.bincount(torch.tensor(labels))
        class_weights  = 1.0 / class_count.float()
        sample_weights = [class_weights[l] for l in labels]

        sampler = WeightedRandomSampler(sample_weights,
                                        num_samples=len(sample_weights),
                                        replacement=True)

        train_loader = DataLoader(datasets["dataset_train"],
                                  batch_size=batch_size,
                                  shuffle=False,
                                  sampler=sampler,
                                  num_workers=num_workers,
                                  drop_last=False)

        train_loader_with_val_transforms = DataLoader(datasets["dataset_train_with_val_transforms"],
                                  batch_size=batch_size,
                                  shuffle=False,
                                  sampler=sampler,
                                  num_workers=num_workers,
                                  drop_last=False)


    # ---------- patient-homogeneous batches, no weights ----------
    elif use_patient_sampler:
        patient_ids  = [d["patient_id"] for d in datasets["dataset_train"].data]
        batch_sampler = PatientBatchSampler(patient_ids,
                                            batch_size=batch_size,
                                            drop_last=False)

        train_loader = DataLoader(datasets["dataset_train"],
                                  batch_sampler=batch_sampler,
                                  num_workers=num_workers)
        
        train_loader_with_val_transforms = DataLoader(datasets["dataset_train_with_val_transforms"],
                                  batch_sampler=batch_sampler,
                                  num_workers=num_workers)

    # ---------- patient-homogeneous batches, WITH weights ----------
    elif use_patient_w_sampler:
        # Build patient → weight dict (default 1.0)
        patient_weights = {}
        for d in datasets["dataset_train"].data:
            pid = d["patient_id"]
            w   = d.get("patient_weight", 1.0)
            patient_weights.setdefault(pid, w)

        batch_sampler = WeightedPatientBatchSampler(datasets["dataset_train"],
                                                    patient_weights=patient_weights,
                                                    batch_size=batch_size,
                                                    drop_last=False)

        train_loader = DataLoader(datasets["dataset_train"],
                                  batch_sampler=batch_sampler,
                                  num_workers=num_workers)
        
        train_loader_with_val_transforms = DataLoader(datasets["dataset_train_with_val_transforms"],
                                  batch_sampler=batch_sampler,
                                  num_workers=num_workers)

    # ---------- plain shuffle (baseline) ----------
    else:
        train_loader = DataLoader(datasets["dataset_train"],
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=False)
        
        train_loader_with_val_transforms = DataLoader(datasets["dataset_train_with_val_transforms"],
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=False)

    # validation / test unchanged
    val_loader = DataLoader(datasets["dataset_validation"],
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)

    test_loader = DataLoader(datasets["dataset_test"],
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    return {
        "train_loader": train_loader,
        "val_loader":   val_loader,
        "test_loader":  test_loader,
        "train_loader_with_val_transforms": train_loader_with_val_transforms
    }

# def prepare_dataloaders(datasets, config):
#     batch_size = config["training"]["batch_size"]
#     num_workers = config["training"]["num_workers"]
#     use_sampler = config["training"].get("use_WeightedRandomSampler", False)
#     use_PatientLevelWeightedRandomSampler = config["training"].get("use_PatientLevelWeightedRandomSampler", False)
#     assert not (use_sampler & use_PatientLevelWeightedRandomSampler)


#     sampler = None
#     if use_sampler:
#         labels = [item['score'] for item in datasets["dataset_train"].data]
#         class_count = torch.bincount(torch.tensor(labels))
#         class_weights = 1.0 / class_count.float()
#         sample_weights = [class_weights[label] for label in labels]

#         sampler = WeightedRandomSampler(weights=sample_weights,
#                                         num_samples=len(sample_weights),
#                                         replacement=True)

#     return {
#         "train_loader": DataLoader(
#             datasets["dataset_train"],
#             batch_size=batch_size,
#             shuffle=not use_sampler,
#             sampler=sampler,
#             num_workers=num_workers,
#             drop_last=False
#         ),
#         "val_loader": DataLoader(
#             datasets["dataset_validation"],
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=num_workers,
#         ),
#         "test_loader": DataLoader(
#             datasets["dataset_test"],
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=num_workers,
#         ),
#     }


def dataset_and_loader_several(data_tables_several, config):
    return_dct = {}
    for key, item in data_tables_several.items():
        data = dataset_and_loader(item, config)
        return_dct[key] = data
    return return_dct

def check_duplicates_in_dataloader(data: Dict[str, Dict[str, DataLoader]], ds_key: str = "test_loader") -> None:
    """
    Check if there are duplicate entries in the dataloaders.
    """
    for k in data:
        dl = data[k][ds_key]
        seen = set()
        for b in dl:
            s = [ss for ss in zip(b["patient_id"], b["date_str"], b["left_or_right"], b["score_type"])]
            for ss in s:
                if ss in seen:
                    print("seen", ss)
                    raise ValueError("Duplicate found in dataloader  {k} ")
                else:
                    seen.add(ss)

#--------------------------------------------------------------#
#------------------------- Other      -------------------------#
#--------------------------------------------------------------#




