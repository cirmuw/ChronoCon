# Standard library imports
import json
from importlib import resources
from pathlib import Path
from typing import Dict, List, Literal, Any, Optional

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
from sklearn.model_selection import train_test_split
import sklearn.utils
from tqdm import tqdm

# MONAI imports
from monai.data import Dataset, DataLoader
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    CopyItemsd,
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LambdaD,
    LoadImaged,
    Rand2DElasticd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRanged, 
    RandHistogramShiftd,
    RandGaussianSmoothd,
    RandCoarseDropoutd,
    RandAdjustContrastd,
    AdjustContrastd
)
from torch.utils.data import WeightedRandomSampler

# Custom imports
from ra_utils.autoscora.autoscorRA_Pipeline.scoring.src.run_utils import (
    restructure_paths_and_scores,
    restructure_paths_and_scores_v2
)
from ra_utils.data.data_utils import extract_extras_from_filename
from ra_utils.data.shap_sums import limit_treatment_number

from typing import Literal



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
#------------------ Triplet Transforms ------------------------#
#--------------------------------------------------------------#

def make_training_transforms_triplet(cfg, DIM=2):
    # ----------------------
    # helpers
    # ----------------------
    def _gauss_smooth_kwargs():
        sig = cfg.get("RandGaussianSmoothd__sigma", (0.6, 1.2))
        if DIM == 2:
            return dict(sigma_x=sig, sigma_y=sig)
        else:
            return dict(sigma_x=sig, sigma_y=sig, sigma_z=sig)

    def _roi_size():
        return [cfg["OUTPUT_DIM"]]*DIM if isinstance(cfg["OUTPUT_DIM"], int) else cfg["OUTPUT_DIM"]

    def _clamp01(x):
        if torch.is_tensor(x):
            return x.clamp_(0.0, 1.0)
        return np.clip(x, 0.0, 1.0)

    def _add_view_aug(xforms, key):
        # ------- geometry -------
        if cfg.get("RandAffined__bool", False):
            xforms.append(
                RandAffined(
                    keys=[key],
                    prob=cfg.get("RandAffined__prob", 0.5),
                    rotate_range=cfg.get("RandAffined__rotation_range_degree", 10) * np.pi / 180,
                    translate_range=cfg.get("RandAffined__translate_range_pixels", 19),
                    scale_range=cfg.get("RandAffined__scale_range", 0.2),
                    shear_range=cfg.get("RandAffined__shear_range", None),  # e.g. (-0.05, 0.05)
                    padding_mode="zeros",
                )
            )

        if cfg.get("Rand2DElasticd__bool", False):
            sp = cfg.get("Rand2DElasticd__spacing", 12)
            mag = cfg.get("Rand2DElasticd__magnitude", 2.73)
            xforms.append(
                Rand2DElasticd(
                    keys=[key],
                    spacing=(sp, sp),
                    magnitude_range=(-mag, mag),
                    prob=cfg.get("Rand2DElasticd__prob", 0.2),
                    shear_range=None,
                    rotate_range=0, scale_range=0, translate_range=0,
                    padding_mode="zeros",
                )
            )

        if cfg.get("RandFlipd__bool", False):
            xforms.append(RandFlipd(keys=[key], prob=cfg.get("RandFlipd__prob", 0.5), spatial_axis=1))

        # ------- crop AFTER geometry (non-commutative with affine) -------
        xforms.append(CenterSpatialCropd(keys=[key], roi_size=_roi_size()))

        # ------- photometric -------
        if cfg.get("RandScaleIntensityd__bool", False):
            xforms.append(
                RandScaleIntensityd(
                    keys=[key],
                    factors=cfg.get("RandScaleIntensityd__factors", (-0.1, 0.1)),
                    prob=cfg.get("RandScaleIntensityd__prob", 0.3),
                )
            )
        if cfg.get("RandShiftIntensityd__bool", False):
            xforms.append(
                RandShiftIntensityd(
                    keys=[key],
                    offsets=cfg.get("RandShiftIntensityd__offsets", (-0.1, 0.1)),
                    prob=cfg.get("RandShiftIntensityd__prob", 0.3),
                )
            )
        # photometric
        if cfg.get("RandAdjustContrastd__bool", False):
            xforms.append(
                RandAdjustContrastd(
                    keys=[key],
                    prob=cfg.get("RandAdjustContrastd__prob", 0.3),
                    gamma=cfg.get("RandAdjustContrastd__gamma", (0.7, 1.5)),
                    invert_image=cfg.get("RandAdjustContrastd__invert_image", False),
                )
            )
        if cfg.get("RandHistogramShiftd__bool", False):
            xforms.append(
                RandHistogramShiftd(
                    keys=[key],
                    num_control_points=cfg.get("RandHistogramShiftd__num_control_points", 10),
                    prob=cfg.get("RandHistogramShiftd__prob", 0.2),
                )
            )

        # ------- blur / noise / occlusion -------
        if cfg.get("RandGaussianSmoothd__bool", False):
            xforms.append(
                RandGaussianSmoothd(
                    keys=[key],
                    prob=cfg.get("RandGaussianSmoothd__prob", 0.3),
                    **_gauss_smooth_kwargs(),
                )
            )
        if cfg.get("RandGaussianNoised__bool", False):
            xforms.append(
                RandGaussianNoised(
                    keys=[key],
                    prob=cfg.get("RandGaussianNoised__prob", 0.17),
                    mean=0.0,
                    std=cfg.get("RandGaussianNoised__std", 0.085),
                )
            )
        if cfg.get("RandCoarseDropoutd__bool", False):
            xforms.append(
                RandCoarseDropoutd(
                    keys=[key],
                    prob=cfg.get("RandCoarseDropoutd__prob", 0.15),
                    holes=cfg.get("RandCoarseDropoutd__holes", 3),
                    spatial_size=cfg.get("RandCoarseDropoutd__spatial_size", (8, 8) if DIM == 2 else (8, 8, 4)),
                    fill_value=cfg.get("RandCoarseDropoutd__fill_value", 0.0),
                )
            )

        if cfg.get("Clamp01__bool", True):
            xforms.append(LambdaD(keys=[key], func=_clamp01))

        # ------- per-view normalization at the end -------
        if cfg.get("RBG_channels__bool", False) and cfg.get("Normalize__bool", True):
            xforms.append(
                LambdaD(
                    keys=[key],
                    func=torchvision.transforms.Normalize(
                        mean=cfg.get("Normalize__mean", [0.485, 0.456, 0.406]),
                        std=cfg.get("Normalize__std", [0.229, 0.224, 0.225]),
                    ),
                )
            )



        return xforms



    # ----------------------
    # base (deterministic-ish)
    # ----------------------
    xforms = [
        LoadImaged(keys=["img"], reader="NumpyReader"),
        EnsureChannelFirstd(keys=["img"]),
        LambdaD(keys=["img"], func=min_max_normalize),
    ]

    # Make 3-ch (before branching & photometric ops)
    if cfg.get("RBG_channels__bool", False):
        xforms.append(LambdaD(keys=["img"], func=repeat_channels))

    # -------- branch into two independent views --------
    xforms += [CopyItemsd(keys=["img"], times=2, names=["img_original", "img_pos"])]

    # view 1 & 2
    xforms = _add_view_aug(xforms, "img")
    xforms = _add_view_aug(xforms, "img_pos")

    xforms.append(EnsureTyped(keys=["img", "img_pos", "img_original"], dtype=torch.float32))

    if cfg.get("drop_img_original", True):
        xforms.append(DeleteItemsd(keys=["img_original"]))

    return xforms



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
        def num_classes_foo(row):
            t = row["chosen_score"]
            limit = limits_dct[t]
            return limit + 1 # 0 is class
        df_include["number of score classes"] = df_include.apply(num_classes_foo, axis=1)        


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
        def num_classes_foo(row):
            t = row["chosen_score"]
            limit = limits_dct[t]
            return limit + 2 # 0 is class anb over limit to limit + 1
        df_include["number of score classes"] = df_include.apply(num_classes_foo, axis=1)        

    
    elif how_to_deal_with_surgery == "keep: map over limit to limit":
        df_exclude = pd.DataFrame(columns=paths_scores_df.columns)
        df_include = paths_scores_df
        def foo(row):
            t = row["chosen_score"]
            limit = limits_dct[t]
            s = row["score"]
            return limit_treatment_number(s, limit, limit_treatment="over_limit_to_limit")
        df_include["score"] = df_include.apply(foo, axis=1)
        def num_classes_foo(row):
            t = row["chosen_score"]
            limit = limits_dct[t]
            return limit + 1 # 0 is class
        df_include["number of score classes"] = df_include.apply(num_classes_foo, axis=1)       

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



    df_include = add__image_series_length(df_include)
    df_include = add__score_maxdiff(df_include)
    df_include = add__progession(df_include)
    df_include = add__score_mean(df_include)
    df_include = add__score_median(df_include)
    df_include["score mean ceil"] = np.ceil(df_include["score mean"]).astype(int)
    df_include["score mean round"] = np.round(df_include["score mean"]).astype(int)


    time_scale = data_config.get("tSLR_time_scale", 0.0) # 0.1 is good default value, but want to keep backward compatible
    forward_scale = data_config.get("tSLR_forward_scale", 1.0) 
    backward_scale = data_config.get("tSLR_backward_scale", 1.0)
    score_correction = score_modification_tSLR(df_include, 
                                        time_scale = time_scale, 
                                        forward_scale = forward_scale, 
                                        backward_scale = backward_scale
                                        )
    df_include["score tSLR"] = df_include["score"] + score_correction

    if data_config.get("use_splits_file", False):
        df_train, df_val, df_test, df_missed = split_using_file(df_include, data_config["splits_file"])
    else:
        df_train, df_val, df_test = split_training_val_test__on_patient_level(df_include, ratios=data_config["split_ratio"])
        df_missed = None

    df_train = pd.merge(df_paths[["image_path", "file_name"]], df_train, on="file_name")
    df_val   = pd.merge(df_paths[["image_path", "file_name"]], df_val, on="file_name")
    df_test  = pd.merge(df_paths[["image_path", "file_name"]], df_test, on="file_name")


    # if data_config["sampler_name"] == "PatientLevelWeightedRandomSampler": 
    #     rule = data_config.get("patient_level_class_balance_aggregation_rule")
    #     n_buckets = data_config.get("delta_buckets_k", 4)    # only for delta_buckets
    #     df_train = add_patient_class_weights_and_labels(df_train, rule = rule, n_buckets=n_buckets)
    #     df_val   = add_patient_class_weights_and_labels(df_val,   rule = rule, n_buckets=n_buckets)
    #     df_test  = add_patient_class_weights_and_labels(df_test,  rule = rule, n_buckets=n_buckets)

    # Score correction t-SLR (time-soft-label regularization) only on training set!
    enable_tSLR = data_config.get("enable_tSLR", False)
    if enable_tSLR: 
        print("t-SLR for training set is activate!")
        df_train["score"] = df_train["score tSLR"]


    return {
        "df_include": df_include,
        "df_exclude": df_exclude,
        "df_train": df_train,
        "df_val": df_val,
        "df_test": df_test,
        "df_included_missed": df_missed
    }


# Basically only for Patientlevel loader. 
# Not advisable to use this! Maybe for plotting... Performance was always worse with this. 
def add_weights_for_oversampling(data_tables: dict, data_config: dict):
    data_keys = list(data_tables.keys())

    rule = data_config.get("patient_level_class_balance_aggregation_rule")
    n_buckets = data_config.get("delta_buckets_k", 4)    # only for delta_buckets
    for key in data_keys:
        data_part = data_tables[key]
        data_part["df_train"] = add_patient_class_weights_and_labels(data_part["df_train"], rule = rule, n_buckets=n_buckets)
        data_part["df_val"]   = add_patient_class_weights_and_labels(data_part["df_val"], rule = rule, n_buckets=n_buckets)
        data_part["df_test"]  = add_patient_class_weights_and_labels(data_part["df_test"], rule = rule, n_buckets=n_buckets)
    return data_tables

#data_tables = add_weights_for_oversampling(data_tables, data_config=config["data"])




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

def df_scores_to_dct_list(df: pd.DataFrame, 
                          optional_keys = ["patient_cls_lbl", "patient_weight", 'preds', 'preds_float', 'model_confidence', 
                                           "score mean", "score median", "score mean ceil", "score mean round"]
                                           ) -> List[dict]:
    data = []
    for idx, row in df.iterrows():
        # Each dict is a sample, referencing row["patch"] plus additional metadata
        d = {
            "img": row["image_path"],  # .npy file path on disk
            "image_path": row["image_path"],  # .npy file path on disk            
            "file_name": row["file_name"],
            "score": row["score"],
            "score_type": row["chosen_score"],
            "JSN_or_ERO": row["chosen_score_type"],
            "extremity": row["extremity"],
            "patient_id": row["patient_id"], 
            "date_str": row["date_str"], 
            "left_or_right": row["left_or_right"],
            "roi_name": row["roi_name"],
            "patient_scoretype_key":  f"{row['extremity']}_{row['left_or_right']}_{row['roi_name']}_{row['patient_id']}_{row['chosen_score']}",     # everything except the time
            "patient_scoretype_key_and_date":  f"{row['extremity']}_{row['left_or_right']}_{row['roi_name']}_{row['patient_id']}_{row['chosen_score']}_{row['date_str']}" 
        }

        for optional_key in optional_keys:
            if optional_key in row.keys():
                d[optional_key] = row[optional_key]
        data.append(d)

    return data



def dataset_and_loader(data_tables, config):
    """
    Create datasets and dataloaders for training, validation, and testing.
    """
    datasets = prepare_datasets(data_tables, config)
    loaders = prepare_dataloaders(datasets, config)
    
    return {**datasets, **loaders}


def transforms_from_config(config: dict):
    use_triplet_transforms = config["transforms"].get("ACTIVATE_TRIPLET_TRAINING_TRANSFORMS", False)
    if use_triplet_transforms:
        transform_train = Compose(make_training_transforms_triplet(config["transforms"]))
    else:
        transform_train = Compose(make_trainings_transforms(config["transforms"]))

    transform_val = Compose(make_validation_transforms(config["transforms"]))
    return {"transform_train": transform_train, 
            "transform_val": transform_val}


def prepare_datasets(data_tables, config):
    tran = transforms_from_config(config)
    transform_train = tran["transform_train"]
    transform_val = tran["transform_val"]

    # TODO: Read from config
    optional_keys=["patient_cls_lbl", "patient_weight", 'preds', 'preds_float', 'model_confidence',
                   'score_change_per_year_to_next', 'score_change_per_year_to_prev',
                   'years_to_next_visit', 'years_to_prev_visit',
                   'score_difference_prev_visit', 'score_difference_next_visit', 
                   "number of score classes",  'score tSLR', 'score maxdiff'
                   ]

    return {
        "dataset_train": Dataset(
            data=df_scores_to_dct_list(data_tables["df_train"], optional_keys=optional_keys),
            transform=transform_train
        ),
        "dataset_validation": Dataset(
            data=df_scores_to_dct_list(data_tables["df_val"], optional_keys=optional_keys),
            transform=transform_val
        ),
        "dataset_test": Dataset(
            data=df_scores_to_dct_list(data_tables["df_test"], optional_keys=optional_keys),
            transform=transform_val
        ),
        "dataset_train_with_val_transforms": Dataset(
            data=df_scores_to_dct_list(data_tables["df_train"], optional_keys=optional_keys),
            transform=transform_val
        ),
    }




def add__image_series_length(df):
    group_sizes = df.groupby(["patient_id", "left_or_right", "chosen_score"]).size()
    group_sizes_df = group_sizes.reset_index(name='image_series_length')
    df = df.merge(group_sizes_df, on=["patient_id", "left_or_right", "chosen_score"])
    return df 

def add__score_maxdiff(df):
    group_progression = df.groupby(["patient_id", "left_or_right", "chosen_score"])["score"].max() - df.groupby(["patient_id", "left_or_right", "chosen_score"])["score"].min()
    group_progression_df = group_progression.reset_index(name='score maxdiff')
    df = df.merge(group_progression_df, on=["patient_id", "left_or_right", "chosen_score"])
    return df

def add__score_mean(df):
    group_mean = (
        df.groupby(["patient_id", "left_or_right", "chosen_score"])["score"]
        .mean()
        .reset_index(name="score mean")
    )
    df = df.merge(group_mean, on=["patient_id", "left_or_right", "chosen_score"])
    return df


def add__score_median(df):
    group_median = (
        df.groupby(["patient_id", "left_or_right", "chosen_score"])["score"]
        .median()
        .reset_index(name="score median")
    )
    df = df.merge(group_median, on=["patient_id", "left_or_right", "chosen_score"])
    return df



def score_modification_tSLR(sub: pd.DataFrame, time_scale = 0.1, forward_scale = 1.0, backward_scale = 1.0):

    # will_progress = (sub["score_difference_next_visit"] > 0)
    # did_progress = (sub["score_difference_prev_visit"] > 0)
    is_first = (sub["years_to_prev_visit"].isna())
    is_last = (sub["years_to_next_visit"].isna())

    correction_f = (sub["years_to_next_visit"]*time_scale) * sub["score_change_per_year_to_next"]
    correction_f *= (~is_first).astype(float)

    correction_b = -(sub["years_to_prev_visit"]*time_scale) * sub["score_change_per_year_to_prev"]
    correction_b *= (~is_last).astype(float)

    correction = correction_b * backward_scale + correction_f * forward_scale
    correction.iloc[is_last] = 0.0
    correction.iloc[is_first] = 0.0

    return correction





def add__progession(df):
    """
    Adds per-series (patient_id, left_or_right, chosen_score):
      - years_to_next_visit
      - score_difference_next_visit
      - score_change_per_year_to_next      (NaN if time<=0 or missing)

      - years_to_prev_visit
      - score_difference_prev_visit
      - score_change_per_year_to_prev      (NaN if time<=0 or missing)
    """
    group_cols = ["patient_id", "left_or_right", "chosen_score"]
    orig_index = df.index

    # Work on a copy sorted within groups by date (parse if needed)
    tmp = df.copy()
    if "date_dt" not in tmp.columns:
        tmp["date_dt"] = pd.to_datetime(tmp["date_str"], errors="coerce")

    tmp = tmp.sort_values(group_cols + ["date_dt"])

    # ---------- NEXT visit (lead)
    
    #tmp["max_score_difference"] = tmp.groupby(group_cols)["score"].max() - tmp.groupby(group_cols)["score"].min()
    tmp["_score_next"] = tmp.groupby(group_cols)["score"].shift(-1)
    tmp["_date_next"]  = tmp.groupby(group_cols)["date_dt"].shift(-1)

    tmp["score_difference_next_visit"] = tmp["_score_next"] - tmp["score"]
    dt_next = (tmp["_date_next"] - tmp["date_dt"])
    tmp["years_to_next_visit"] = dt_next.dt.days / 365.25

    tmp["score_change_per_year_to_next"] = (
        tmp["score_difference_next_visit"] / tmp["years_to_next_visit"]
    )
    tmp.loc[tmp["years_to_next_visit"] <= 0, "score_change_per_year_to_next"] = np.nan

    # ---------- PREVIOUS visit (lag)
    tmp["_score_prev"] = tmp.groupby(group_cols)["score"].shift(1)
    tmp["_date_prev"]  = tmp.groupby(group_cols)["date_dt"].shift(1)

    tmp["score_difference_prev_visit"] = tmp["score"] - tmp["_score_prev"]
    dt_prev = (tmp["date_dt"] - tmp["_date_prev"])
    tmp["years_to_prev_visit"] = dt_prev.dt.days / 365.25

    tmp["score_change_per_year_to_prev"] = (
        tmp["score_difference_prev_visit"] / tmp["years_to_prev_visit"]
    )
    tmp.loc[tmp["years_to_prev_visit"] <= 0, "score_change_per_year_to_prev"] = np.nan

    # Cleanup helpers and restore original row order
    tmp = tmp.drop(columns=["_score_next", "_date_next", "_score_prev", "_date_prev"]).reindex(orig_index)

    return tmp


def filter_dataset_to_instance(df, transform, chosen_score="PIPII", left_or_right = "L", patient_id="557"):
    m_1 = (df["chosen_score"] == chosen_score)
    m_2 = (df["left_or_right"] == left_or_right)
    m_3 = (df["patient_id"] == patient_id)
    m = m_1 & m_2 & m_3

    df_tmp = df[m].sort_values("date_str")#[["date_str", "score"]]

    ds = Dataset(
        data=df_scores_to_dct_list(df_tmp.reset_index(drop=True)),
        transform=transform
    )
    return ds



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



#### GroupedWeightedRandomSampler
import math
import random
from collections import defaultdict, Counter
from typing import List, Dict, Sequence, Tuple, Optional, Iterable, Callable, Union

import numpy as np
import torch
from torch.utils.data import BatchSampler


def _ceil_mean(arr: Sequence[float]) -> int:
    """ceil(mean(arr)) as an int; empty -> 0"""
    return 0 if not arr else int(math.ceil(float(np.mean(arr))))


def _build_group_key(row: Dict, group_identifiers: Sequence[str]) -> Tuple:
    """Make a hashable group key from row fields."""
    return tuple(row[g] for g in group_identifiers)


class GroupedWeightedRandomBatchSampler(BatchSampler):
    """
    Mostly-homogeneous batches with label-balanced group selection and graceful top-up.

    - Picks a PRIMARY group each batch (weighted by inverse-frequency of an aggregated label).
    - Tries to fill >= max(ceil(min_primary_fraction*B), min_primary_items) from the primary group.
    - If primary can't fill the batch, tops up from other groups (weighted), capped by
      `max_groups_per_batch` distinct groups per batch.
    - When `replacement=True`, can re-use samples (optionally capped via `repeats_per_group_cap`)
      to reach the desired primary fraction/size.

    Assumes dataset has attribute `.data` which is a list[dict] of metadata rows containing:
      - the `group_identifiers` fields (e.g. "patient_id", "score_type", "left_or_right")
      - the `target_col` (e.g. "score") to aggregate for label balancing

    Args
    ----
    dataset: any object with `.data: list[dict]` aligned to indexing
    batch_size: int > 0
    group_identifiers: fields defining a group
    target_col: column used to build labels for balancing (default "score")
    group_score_agg: "mean_ceil" or a callable(list[float]) -> int
    replacement: allow sampling within a group with replacement when its pool empties
    drop_last: drop final incomplete batch
    seed: RNG seed
    min_primary_fraction: target fraction from primary group (0..1)
    max_groups_per_batch: max distinct groups allowed in a single batch (>=1)
    min_primary_items: hard minimum #items from primary (overrides fraction if larger); None to disable
    repeats_per_group_cap: when replacement=True, limit per-sample index repeats in a single batch;
                           None means no explicit cap (other than batch_size)
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        group_identifiers: Sequence[str],
        target_col: str = "score",
        group_score_agg: Union[str, Callable[[List[float]], int]] = "mean_ceil",
        replacement: bool = True,
        drop_last: bool = False,
        seed: int = 0,
        min_primary_fraction: float = 0.7,
        max_groups_per_batch: int = 3,
        min_primary_items: Optional[int] = None,
        repeats_per_group_cap: Optional[int] = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not (0.0 <= min_primary_fraction <= 1.0):
            raise ValueError("min_primary_fraction must be in [0, 1]")
        if max_groups_per_batch < 1:
            raise ValueError("max_groups_per_batch must be >= 1")

        self.dataset = dataset
        self.data = getattr(dataset, "data", None)
        if self.data is None:
            raise ValueError("dataset must have attribute `.data` (list of dicts)")

        self.batch_size = int(batch_size)
        self.group_identifiers = list(group_identifiers)
        self.target_col = target_col
        self.replacement = bool(replacement)
        self.drop_last = bool(drop_last)
        self.min_primary_fraction = float(min_primary_fraction)
        self.max_groups_per_batch = int(max_groups_per_batch)
        self.min_primary_items = None if min_primary_items is None else int(min_primary_items)
        self.repeats_per_group_cap = None if repeats_per_group_cap is None else int(repeats_per_group_cap)
        self.rng = random.Random(int(seed))

        # 1) Build groups: key -> list of dataset indices
        groups: Dict[Tuple, List[int]] = defaultdict(list)
        for idx, row in enumerate(self.data):
            key = _build_group_key(row, self.group_identifiers)
            groups[key].append(idx)
        self.groups: Dict[Tuple, List[int]] = dict(groups)

        if len(self.groups) == 0:
            raise ValueError("No groups found — check `group_identifiers` and dataset.data")

        # 2) Determine group labels for balancing (via aggregated target)
        if callable(group_score_agg):
            agg_fn = group_score_agg
        else:
            if group_score_agg != "mean_ceil":
                raise ValueError(f"Unknown group_score_agg='{group_score_agg}' (use 'mean_ceil' or a callable)")
            agg_fn = _ceil_mean

        self.group_label: Dict[Tuple, int] = {}
        for key, idxs in self.groups.items():
            vals = [float(self.data[i][self.target_col]) for i in idxs]
            self.group_label[key] = int(agg_fn(vals))

        # 3) Inverse-frequency weighting over group labels -> per-group weight
        label_counts = Counter(self.group_label.values())
        # Guard against divide-by-zero (shouldn't happen, but be safe)
        class_weight = {lab: (1.0 / c if c > 0 else 0.0) for lab, c in label_counts.items()}
        self.group_weight = {k: float(class_weight[self.group_label[k]]) for k in self.groups}

        # 4) Pools for within-group draws (for without-replacement behavior)
        self._pools: Dict[Tuple, List[int]] = {}
        for key, idxs in self.groups.items():
            pool = idxs.copy()
            self.rng.shuffle(pool)
            self._pools[key] = pool

        # 5) Cache keys/probs for weighted picks
        self._group_keys: List[Tuple] = list(self.groups.keys())
        self._group_probs: List[float] = self._normalize([self.group_weight[k] for k in self._group_keys])

        # 6) Length estimate (approx): number of full batches; if not drop_last, add one if remainder
        total = len(self.data)
        self._len = total // self.batch_size
        if not self.drop_last and (total % self.batch_size):
            self._len += 1

    @staticmethod
    def _normalize(weights: List[float]) -> List[float]:
        s = float(sum(weights))
        return [1.0 / len(weights)] * len(weights) if s <= 0 else [w / s for w in weights]

    def _pick_group_weighted(self, exclude: Optional[set] = None) -> Optional[Tuple]:
        keys, probs = [], []
        for k, p in zip(self._group_keys, self._group_probs):
            if exclude and k in exclude:
                continue
            keys.append(k)
            probs.append(p)
        if not keys:
            return None
        # simple roulette wheel
        r = self.rng.random()
        acc = 0.0
        for k, p in zip(keys, self._normalize(probs)):
            acc += p
            if r <= acc:
                return k
        return keys[-1]

    def _draw_from_group(
        self,
        key: Tuple,
        need: int,
        per_batch_repeat_counter: Optional[Dict[int, int]] = None,
    ) -> List[int]:
        """
        Draw up to `need` indices from group `key`.
        - Without replacement: pop from shuffled pool until empty.
        - With replacement: when pool empties, refill (reshuffle). If `repeats_per_group_cap`
          is set, avoid drawing an index more than that many times within the *current* batch.
        """
        if need <= 0:
            return []

        drawn: List[int] = []
        pool = self._pools[key]

        def can_use(idx: int) -> bool:
            if not self.replacement:
                return True
            if self.repeats_per_group_cap is None or per_batch_repeat_counter is None:
                return True
            return per_batch_repeat_counter.get(idx, 0) < self.repeats_per_group_cap

        # To avoid potential infinite loops with replacement+caps, bound attempts
        max_attempts = max(50, 5 * need)

        attempts = 0
        while need > 0 and attempts < max_attempts:
            attempts += 1

            if pool:
                candidate = pool.pop()
                if can_use(candidate):
                    drawn.append(candidate)
                    need -= 1
                    if per_batch_repeat_counter is not None:
                        per_batch_repeat_counter[candidate] = per_batch_repeat_counter.get(candidate, 0) + 1
                # if can't use (cap hit), skip; continue to try others
            else:
                if self.replacement:
                    # refill the pool by reshuffling the full group
                    pool[:] = self.groups[key]
                    self.rng.shuffle(pool)
                else:
                    break  # can't draw more from this group without replacement

        return drawn

    def __iter__(self) -> Iterable[List[int]]:
        produced = 0
        target_total = self._len * self.batch_size if self.drop_last else len(self.data)

        while produced < target_total:
            # Per-batch repeat counter (only used when replacement=True and cap provided)
            per_batch_repeat_counter: Optional[Dict[int, int]] = {} if (self.replacement and self.repeats_per_group_cap is not None) else None

            # 1) Pick primary group
            primary = self._pick_group_weighted()
            if primary is None:
                break

            batch: List[int] = []
            groups_used = {primary}
            need = self.batch_size

            # desired minimum from primary (fraction and/or hard floor)
            want_primary = int(math.ceil(self.min_primary_fraction * self.batch_size))
            if self.min_primary_items is not None:
                want_primary = max(want_primary, int(self.min_primary_items))

            # 2) Draw from primary first
            primary_draw = self._draw_from_group(primary, min(need, want_primary), per_batch_repeat_counter)
            batch.extend(primary_draw)
            need -= len(primary_draw)

            # If we still haven't reached want_primary and replacement is allowed, try to top up more from primary
            if self.replacement and len(batch) < want_primary and need > 0:
                extra = self._draw_from_group(primary, min(need, want_primary - len(batch)), per_batch_repeat_counter)
                batch.extend(extra)
                need -= len(extra)

            # 3) Top-up from other groups (respect max_groups_per_batch)
            attempts = 0
            while need > 0 and len(groups_used) < self.max_groups_per_batch and attempts < 10 * self.max_groups_per_batch:
                attempts += 1
                g = self._pick_group_weighted(exclude=groups_used)
                if g is None:
                    break
                draw = self._draw_from_group(g, need, per_batch_repeat_counter)
                if draw:
                    batch.extend(draw)
                    need -= len(draw)
                    groups_used.add(g)
                else:
                    # couldn't draw from this group (e.g., empty w/o replacement) -> skip it
                    groups_used.add(g)

            # 4) If still short and replacement is True, allow draws from any group to fill
            attempts = 0
            while need > 0 and self.replacement and attempts < 20:
                attempts += 1
                g = self._pick_group_weighted()
                if g is None:
                    break
                draw = self._draw_from_group(g, need, per_batch_repeat_counter)
                if draw:
                    batch.extend(draw)
                    need -= len(draw)

            # 5) Yield or drop
            if len(batch) == self.batch_size:
                yield batch
                produced += self.batch_size
            else:
                if not self.drop_last and batch:
                    yield batch
                    produced += len(batch)
                # else: drop incomplete and continue

    def __len__(self) -> int:
        return self._len


### Sampler in two stages. 
# Stage 1: uniform or mostly uniform regarding score types 
#  -> e.g. PIPII_JSN, PIPII_ERO, PIPIII_JSN, ... ) should appear on average equally often. 
# Stage 2: Oversample on score values (within the same score type)  
#  -> frequency of JSN=4 should be equally often JSN=0



import math
import random
from collections import defaultdict, Counter
from typing import Dict, List, Sequence, Tuple, Optional, Iterable, Union, Callable

import numpy as np
from torch.utils.data import BatchSampler

def _build_group_key(row: Dict, cols: Sequence[str]) -> Tuple:
    return tuple(row[c] for c in cols)

def _ceil_mean(vals: List[float]) -> int:
    if not vals:
        return 0
    return int(math.ceil(float(np.mean(vals))))

def _roulette_idx(rng: random.Random, weights: List[float]) -> int:
    s = float(sum(weights))
    if s <= 0 or not np.isfinite(s):
        return rng.randrange(len(weights))
    r = rng.random() * s
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return len(weights) - 1

class GroupLevelWeightedBatchSampler(BatchSampler):
    """
    Batch sampler that:
      • Forms groups by `group_identifiers` (e.g. ["patient_id","score_type","left_or_right"]).
      • Computes a **group label** from `target_col` via `group_score_agg` (default: ceil(mean(score))).
      • **Oversamples groups** according to label-level weights (e.g., inverse frequency^alpha),
        so groups with higher labels appear more often.
      • Emits batches by concatenating **whole groups** (no splitting, no within-group repeats).
        If adding the next group would exceed batch_size, it starts a new batch.

    Notes
    -----
    - If any single group has size > batch_size → raise ValueError (cannot keep group intact).
    - Group selection can be with or without replacement across an "epoch" (`group_replacement`).
      Within one batch, the same group is used at most once.

    Args
    ----
    dataset: object with `.data: list[dict]`
    batch_size: int
    group_identifiers: fields defining a group
    target_col: numeric score column used to build per-group label
    group_score_agg: "mean_ceil" or callable(list[float]) -> int
    weight_mode: "inverse_freq" or "custom"
    alpha: tempering for inverse frequency weights in [0,1] (0=no reweight, 1=full inverse)
    epsilon: smoothing for counts (>=0)
    prior_per_label: optional dict {label:int -> prior:float} (used when weight_mode="custom" or to scale inverse_freq)
    group_replacement: sample groups with replacement across batches (True) or without (False)
    drop_last: drop final short batch
    seed: RNG seed
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        group_identifiers: Sequence[str],
        *,
        target_col: str = "score",
        group_score_agg: Union[str, Callable[[List[float]], int]] = "mean_ceil",
        weight_mode: str = "inverse_freq",
        alpha: float = 1.0,
        epsilon: float = 1.0,
        prior_per_label: Optional[Dict[int, float]] = None,
        group_replacement: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0,1]")

        self.dataset = dataset
        self.data: List[Dict] = getattr(dataset, "data", None)
        if self.data is None:
            raise ValueError("dataset must have attribute `.data` (list of dicts)")

        self.batch_size = int(batch_size)
        self.group_identifiers = list(group_identifiers)
        self.target_col = target_col
        self.weight_mode = weight_mode
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.prior_per_label = prior_per_label or {}
        self.group_replacement = bool(group_replacement)
        self.drop_last = bool(drop_last)
        self.rng = random.Random(int(seed))

        # ---- 1) Build groups: key -> list of indices
        groups: Dict[Tuple, List[int]] = defaultdict(list)
        for idx, row in enumerate(self.data):
            key = _build_group_key(row, self.group_identifiers)
            groups[key].append(idx)
        self.groups: Dict[Tuple, List[int]] = dict(groups)
        if not self.groups:
            raise ValueError("No groups found; check `group_identifiers`.")

        # ---- 2) Compute per-group label from target_col
        if callable(group_score_agg):
            agg_fn = group_score_agg
        else:
            if group_score_agg != "mean_ceil":
                raise ValueError("Unsupported group_score_agg; use 'mean_ceil' or provide a callable.")
            agg_fn = _ceil_mean

        self.group_label: Dict[Tuple, int] = {}
        for key, idxs in self.groups.items():
            vals = [float(self.data[i][self.target_col]) for i in idxs]
            self.group_label[key] = int(agg_fn(vals))

        # ---- 3) Build label counts and group weights (oversampling on group level)
        label_counts = Counter(self.group_label.values())  # frequency over groups
        self.labels_sorted = sorted(label_counts.keys())

        weights: List[float] = []
        self._group_keys: List[Tuple] = list(self.groups.keys())

        for key in self._group_keys:
            lab = self.group_label[key]
            if self.weight_mode == "inverse_freq":
                # inverse count with tempering and optional prior scaling
                base = (label_counts[lab] + self.epsilon)
                inv = base ** (-self.alpha)
                w = inv * float(self.prior_per_label.get(lab, 1.0))
            elif self.weight_mode == "custom":
                w = float(self.prior_per_label.get(lab, 1.0))
            else:
                raise ValueError(f"Unknown weight_mode '{self.weight_mode}'")
            weights.append(w)

        # normalize to probabilities for group selection
        total_w = float(sum(weights))
        if total_w <= 0 or not np.isfinite(total_w):
            self._group_probs = [1.0 / len(weights)] * len(weights)
        else:
            self._group_probs = [w / total_w for w in weights]

        # ---- 4) Sanity check: no group larger than batch_size (cannot keep intact)
        self.group_sizes: Dict[Tuple, int] = {k: len(v) for k, v in self.groups.items()}
        too_big = [k for k, sz in self.group_sizes.items() if sz > self.batch_size]
        if too_big:
            bad = too_big[0]
            raise ValueError(f"Group {bad} size {self.group_sizes[bad]} > batch_size {self.batch_size}. "
                             f"Cannot keep group intact. Reduce grouping granularity or increase batch_size.")

        # ---- 5) Length estimate (approx): total_samples / batch_size (+1 if remainder and not drop_last)
        total_samples = len(self.data)
        self._len = total_samples // self.batch_size
        if not self.drop_last and total_samples % self.batch_size:
            self._len += 1

    def __len__(self) -> int:
        return self._len

    def _pick_group_idx_weighted(self, mask_available: Optional[List[bool]] = None) -> Optional[int]:
        """Pick a group index by weighted roulette; optionally mask out used groups."""
        probs = self._group_probs
        if mask_available is not None:
            avail_indices = [i for i, ok in enumerate(mask_available) if ok]
            if not avail_indices:
                return None
            sub_weights = [probs[i] for i in avail_indices]
            j = _roulette_idx(self.rng, sub_weights)
            return avail_indices[j]
        else:
            return _roulette_idx(self.rng, probs)

    def __iter__(self) -> Iterable[List[int]]:
        produced = 0
        target_total = self._len * self.batch_size if self.drop_last else len(self.data)

        # For group_replacement=False, we keep an epoch-level "available" mask
        epoch_available = [True] * len(self._group_keys)

        while produced < target_total:
            batch: List[int] = []
            batch_groups_used = set()

            # keep picking whole groups until we fill (or decide to end)
            while True:
                # stop if we cannot add any group without exceeding
                remaining = self.batch_size - len(batch)
                if remaining <= 0:
                    break

                # build a list of candidate groups that fit into remaining and are allowed this batch
                # (avoid repeating the same group within a batch)
                candidates_mask = []
                for i, key in enumerate(self._group_keys):
                    if key in batch_groups_used:
                        candidates_mask.append(False)
                        continue
                    if not self.group_replacement and not epoch_available[i]:
                        candidates_mask.append(False)
                        continue
                    if self.group_sizes[key] <= remaining:
                        candidates_mask.append(True)
                    else:
                        candidates_mask.append(False)

                # if no candidate fits:
                if not any(candidates_mask):
                    # if batch is empty (e.g., remaining < smallest group), we must start a new batch
                    # to avoid splitting a group. So end this batch loop.
                    break

                # pick a candidate by weights (restricted to mask)
                gi = self._pick_group_idx_weighted(mask_available=candidates_mask)
                if gi is None:
                    break
                gkey = self._group_keys[gi]
                # append whole group
                batch.extend(self.groups[gkey])
                batch_groups_used.add(gkey)
                if not self.group_replacement:
                    epoch_available[gi] = False

                # continue loop to try adding more groups (if space remains)

            if len(batch) == self.batch_size:
                yield batch
                produced += self.batch_size
            else:
                if not self.drop_last and len(batch) > 0:
                    yield batch
                    produced += len(batch)
                # else: drop too-small batch and continue

            # If we've exhausted all groups in epoch_available (no replacement), reset for another pass
            if not self.group_replacement and not any(epoch_available):
                epoch_available = [True] * len(self._group_keys)





class GroupLevelTypePriorBatchSampler(BatchSampler):
    """
    Hierarchical group-level batch sampler for contrastive trajectories.

    Guarantees:
      • Keeps each group (defined by `group_identifiers`, e.g. ["patient_id","score_type","left_or_right"])
        intact within a batch (no splitting, no within-batch repeats).
      • Enforces a score_type prior π(t): "uniform" | "match_val" | "blend".
      • Within each chosen type, oversamples groups by inverse-frequency (tempered) of the
        group label computed from `target_col` via `group_score_agg` (default ceil(mean(score))).
      • Emits batches by concatenating whole groups until `batch_size` is reached; if the next group
        does not fit, starts a new batch.

    Args
    ----
    dataset: object with `.data: list[dict]` aligned with __getitem__
    batch_size: int > 0
    group_identifiers: fields defining a group, e.g. ["patient_id","score_type","left_or_right"]
    target_col: numeric score column for group-label aggregation
    group_score_agg: "mean_ceil" or callable(list[float]) -> int
    # type prior
    prior_mode: "uniform" | "match_val" | "blend"
    val_type_freq: optional dict {score_type: proportion}, used for match/blend
    blend_mix: in [0,1], π = blend_mix * uniform + (1-blend_mix) * match_val
    # within-type balancing (over group labels)
    alpha: in [0,1], tempering for inverse-frequency (0=no balance, 1=full inverse)
    epsilon: >=0, smoothing for counts
    # batching behavior
    size_correction: bool, if True downweights large groups within type (weight /= group_size)
    group_replacement: if True, groups may be reused across batches within an epoch
    drop_last: drop final short batch
    seed: RNG seed
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        group_identifiers: Sequence[str],
        *,
        target_col: str = "score",
        group_score_agg: Union[str, Callable[[List[float]], int]] = "mean_ceil",
        prior_mode: str = "uniform",
        val_type_freq: Optional[Dict[str, float]] = None,
        blend_mix: float = 0.5,
        alpha: float = 1.0,
        epsilon: float = 1.0,
        size_correction: bool = False,
        group_replacement: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0,1]")
        if not (0.0 <= blend_mix <= 1.0):
            raise ValueError("blend_mix must be in [0,1]")

        self.dataset = dataset
        self.data: List[Dict] = getattr(dataset, "data", None)
        if self.data is None:
            raise ValueError("dataset must have attribute `.data` (list of dicts)")

        self.batch_size = int(batch_size)
        self.group_identifiers = list(group_identifiers)
        self.target_col = target_col
        self.prior_mode = str(prior_mode)
        self.val_type_freq = val_type_freq or {}
        self.blend_mix = float(blend_mix)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.size_correction = bool(size_correction)
        self.group_replacement = bool(group_replacement)
        self.drop_last = bool(drop_last)
        self.rng = random.Random(int(seed))

        # --- 1) Build groups: key -> list[idx]
        groups: Dict[Tuple, List[int]] = defaultdict(list)
        for idx, row in enumerate(self.data):
            key = _build_group_key(row, self.group_identifiers)
            groups[key].append(idx)
        self.groups: Dict[Tuple, List[int]] = dict(groups)
        if not self.groups:
            raise ValueError("No groups found; check `group_identifiers`.")

        # --- 2) Group sizes and guard (no splitting allowed)
        self.group_sizes: Dict[Tuple, int] = {k: len(v) for k, v in self.groups.items()}
        too_big = [k for k, sz in self.group_sizes.items() if sz > self.batch_size]
        if too_big:
            k = too_big[0]
            raise ValueError(
                f"Group {k} has size {self.group_sizes[k]} > batch_size {self.batch_size}. "
                f"Increase batch_size or reduce grouping granularity."
            )

        # --- 3) Compute per-group label from target_col
        if callable(group_score_agg):
            agg_fn = group_score_agg
        elif group_score_agg == "mean_ceil":
            agg_fn = _ceil_mean
        else:
            raise ValueError("Unsupported group_score_agg; use 'mean_ceil' or provide a callable.")

        self.group_label: Dict[Tuple, int] = {}
        for key, idxs in self.groups.items():
            vals = [float(self.data[i][self.target_col]) for i in idxs]
            self.group_label[key] = int(agg_fn(vals))

        # --- 4) Indexing helpers
        self._group_keys: List[Tuple] = list(self.groups.keys())
        # map type for each group (assumes score_type is in the row; recommended that it is in the key)
        self.group_type: Dict[Tuple, str] = {}
        for key, idxs in self.groups.items():
            t = self.data[idxs[0]]["score_type"]
            self.group_type[key] = t

        # --- 5) Type-level structures
        self.types = sorted(set(self.group_type.values()))
        # map: type -> list of group indices (into self._group_keys)
        self.type_to_group_idxs: Dict[str, List[int]] = defaultdict(list)
        for gi, gkey in enumerate(self._group_keys):
            t = self.group_type[gkey]
            self.type_to_group_idxs[t].append(gi)

        # --- 6) Build type prior π(t)
        if self.prior_mode == "uniform":
            self.type_prior = {t: 1.0 / len(self.types) for t in self.types}
        elif self.prior_mode == "match_val":
            vf = self.val_type_freq
            s = sum(vf.get(t, 0.0) for t in self.types)
            if s <= 0:
                self.type_prior = {t: 1.0 / len(self.types) for t in self.types}
            else:
                self.type_prior = {t: vf.get(t, 0.0) / s for t in self.types}
        elif self.prior_mode == "blend":
            u = {t: 1.0 / len(self.types) for t in self.types}
            vf = self.val_type_freq
            s = sum(vf.get(t, 0.0) for t in self.types)
            mv = {t: (vf.get(t, 0.0) / s if s > 0 else u[t]) for t in self.types}
            self.type_prior = {t: self.blend_mix * u[t] + (1 - self.blend_mix) * mv[t] for t in self.types}
            # renormalize
            s2 = sum(self.type_prior.values())
            self.type_prior = {t: self.type_prior[t] / s2 for t in self.types}
        else:
            raise ValueError(f"Unknown prior_mode '{self.prior_mode}'")

        # --- 7) Within-type group weights via inverse freq of group labels (tempered)
        # Count per-type label frequencies across groups
        type_label_counts: Dict[str, Counter] = defaultdict(Counter)
        for gkey, lab in self.group_label.items():
            t = self.group_type[gkey]
            type_label_counts[t][lab] += 1

        # Precompute normalized weights: for each type t, dict {group_index -> weight}
        self.within_type_group_weight: Dict[str, Dict[int, float]] = {}
        for t, gis in self.type_to_group_idxs.items():
            counts = type_label_counts[t]
            w_pairs = []
            for gi in gis:
                gkey = self._group_keys[gi]
                lab = self.group_label[gkey]
                base = (counts[lab] + self.epsilon)
                w = (base ** (-self.alpha))
                if self.size_correction:
                    w /= max(1, self.group_sizes[gkey])
                w_pairs.append((gi, float(w)))
            s = float(sum(w for _, w in w_pairs))
            if s <= 0:
                w_pairs = [(gi, 1.0 / max(1, len(w_pairs))) for gi, _ in w_pairs]
            else:
                w_pairs = [(gi, w / s) for gi, w in w_pairs]
            self.within_type_group_weight[t] = dict(w_pairs)

        # --- 8) Length estimate (approximate)
        total_samples = len(self.data)
        self._len = total_samples // self.batch_size
        if not self.drop_last and total_samples % self.batch_size:
            self._len += 1

    # ---- utilities ----
    def _roulette_from_pairs(self, rng: random.Random, pairs: List[Tuple[int, float]]) -> int:
        """pairs: list[(index, weight)] -> chosen index"""
        s = float(sum(w for _, w in pairs))
        if s <= 0 or not np.isfinite(s):
            return pairs[rng.randrange(len(pairs))][0]
        r = rng.random() * s
        acc = 0.0
        for idx, w in pairs:
            acc += w
            if r <= acc:
                return idx
        return pairs[-1][0]

    def _pick_type_with_soft_quota(self, batch_type_counts: Dict[str, float]) -> str:
        """
        Pick a score_type by prior, softly nudging toward underrepresented types within the current batch.
        Target per-batch type fraction is uniform: 1/|T|.
        """
        tgt = 1.0 / max(1, len(self.types))
        pairs: List[Tuple[str, float]] = []
        for t in self.types:
            prior = self.type_prior[t]
            # penalize types that exceeded target count in this batch
            mult = 1.0 / (1.0 + max(0.0, batch_type_counts.get(t, 0.0) - tgt))
            pairs.append((t, prior * mult))
        # convert to indexable pairs
        idx_pairs = list(enumerate([w for _, w in pairs]))
        chosen_idx = _roulette_idx(self.rng, [w for _, w in idx_pairs])
        return self.types[chosen_idx]

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterable[List[int]]:
        produced = 0
        target_total = self._len * self.batch_size if self.drop_last else len(self.data)

        # epoch-level mask when group_replacement=False
        epoch_available = [True] * len(self._group_keys)

        while produced < target_total:
            batch: List[int] = []
            batch_groups_used = set()
            batch_type_counts: Dict[str, float] = {}

            while True:
                remaining = self.batch_size - len(batch)
                if remaining <= 0:
                    break

                # 1) choose a type by prior (soft quotas inside the batch)
                t = self._pick_type_with_soft_quota(batch_type_counts)

                # 2) build candidates among groups of that type that fit & are allowed
                cand_pairs: List[Tuple[int, float]] = []
                for gi in self.type_to_group_idxs[t]:
                    gkey = self._group_keys[gi]
                    if gkey in batch_groups_used:
                        continue
                    if not self.group_replacement and not epoch_available[gi]:
                        continue
                    if self.group_sizes[gkey] <= remaining:
                        w = self.within_type_group_weight[t][gi]
                        cand_pairs.append((gi, w))

                # If none fit for this type, try other types (fallback)
                picked_gi = None
                picked_type = t
                if not cand_pairs:
                    for t2 in self.types:
                        if t2 == t:
                            continue
                        cand2: List[Tuple[int, float]] = []
                        for gi in self.type_to_group_idxs[t2]:
                            gkey = self._group_keys[gi]
                            if gkey in batch_groups_used:
                                continue
                            if not self.group_replacement and not epoch_available[gi]:
                                continue
                            if self.group_sizes[gkey] <= remaining:
                                w = self.within_type_group_weight[t2][gi]
                                cand2.append((gi, w))
                        if cand2:
                            picked_gi = self._roulette_from_pairs(self.rng, cand2)
                            picked_type = t2
                            break
                else:
                    picked_gi = self._roulette_from_pairs(self.rng, cand_pairs)

                if picked_gi is None:
                    # nothing fits → finish the batch
                    break

                gkey = self._group_keys[picked_gi]
                batch.extend(self.groups[gkey])
                batch_groups_used.add(gkey)
                batch_type_counts[picked_type] = batch_type_counts.get(picked_type, 0.0) + 1.0
                if not self.group_replacement:
                    epoch_available[picked_gi] = False

            if len(batch) == self.batch_size:
                yield batch
                produced += self.batch_size
            else:
                if not self.drop_last and batch:
                    yield batch
                    produced += len(batch)
                # else: drop incomplete batch and continue

            # reset availability if we exhausted all groups without replacement
            if not self.group_replacement and not any(epoch_available):
                epoch_available = [True] * len(self._group_keys)


# ---------------------------------------------------------------------
#  Data-loader factory
# ---------------------------------------------------------------------



def prepare_dataloaders(datasets, config):
    tr_cfg       = config["training"]
    batch_size   = tr_cfg["batch_size"]
    num_workers  = tr_cfg["num_workers"]
    pin_memory   = tr_cfg.get("pin_memory", False)
    multiprocessing_context = tr_cfg.get("multiprocessing_context", None)

    sampler_name = tr_cfg.get("sampler_name", None)
    sampler_settings = tr_cfg.get("sampler_settings", {})  # <--- NEW unified place for knobs
    print(" Sampler for training: ", sampler_name)

    # ---------- classic image-level weighted sampler ----------
    if sampler_name == "WeightedRandomSampler":
        labels = [item['score'] for item in datasets["dataset_train"].data]
        labels = np.int_(np.round(labels))
        class_count    = torch.bincount(torch.tensor(labels))
        class_weights  = 1.0 / class_count.float().clamp_min(1)
        sample_weights = [class_weights[l].item() for l in labels]

        sampler = WeightedRandomSampler(sample_weights,
                                        num_samples=len(sample_weights),
                                        replacement=True)

        train_loader = DataLoader(datasets["dataset_train"],
                                  batch_size=batch_size,
                                  shuffle=False,
                                  sampler=sampler,
                                  num_workers=num_workers,
                                  drop_last=False,
                                  pin_memory=pin_memory,
                                  multiprocessing_context=multiprocessing_context)


    elif sampler_name == "GroupLevelWeightedBatchSampler":
        s = sampler_settings
        group_identifiers  = s.get("group_identifiers", ["patient_id","score_type","left_or_right"])
        target_col         = s.get("target_col", "score")
        group_score_agg    = s.get("group_score_agg", "mean_ceil")   # or callable
        weight_mode        = s.get("weight_mode", "inverse_freq")    # "inverse_freq" | "custom"
        alpha              = float(s.get("alpha", 1.0))              # tempering on inverse-freq
        epsilon            = float(s.get("epsilon", 1.0))
        prior_per_label    = s.get("prior_per_label", None)          # dict or None
        group_replacement  = bool(s.get("group_replacement", True))  # replacement across batches
        drop_last          = bool(s.get("drop_last", False))
        seed               = int(s.get("seed", 0))

        batch_sampler = GroupLevelWeightedBatchSampler(
            dataset=datasets["dataset_train"],
            batch_size=batch_size,
            group_identifiers=group_identifiers,
            target_col=target_col,
            group_score_agg=group_score_agg,
            weight_mode=weight_mode,
            alpha=alpha,
            epsilon=epsilon,
            prior_per_label=prior_per_label,
            group_replacement=group_replacement,
            drop_last=drop_last,
            seed=seed,
        )

        train_loader = DataLoader(
            datasets["dataset_train"],
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            multiprocessing_context=multiprocessing_context,
        )

    elif sampler_name == "GroupLevelTypePriorBatchSampler":
        s = sampler_settings
        group_identifiers = s.get("group_identifiers", ["patient_id", "score_type", "left_or_right"])
        target_col        = s.get("target_col", "score")
        group_score_agg   = s.get("group_score_agg", "mean_ceil")  # or a callable

        # type prior
        prior_mode   = s.get("prior_mode", "uniform")              # "uniform" | "match_val" | "blend"
        val_type_freq = s.get("val_type_freq", None)               # dict {score_type: proportion} or None
        blend_mix    = float(s.get("blend_mix", 0.5))              # only used if prior_mode="blend"

        # within-type balancing over group labels
        alpha        = float(s.get("alpha", 1.0))
        epsilon      = float(s.get("epsilon", 1.0))
        size_correction = bool(s.get("size_correction", False))    # optional: downweight large groups

        # batching behavior
        group_replacement = bool(s.get("group_replacement", True))
        drop_last         = bool(s.get("drop_last", False))
        seed              = int(s.get("seed", 0))

        batch_sampler = GroupLevelTypePriorBatchSampler(
            dataset=datasets["dataset_train"],
            batch_size=batch_size,
            group_identifiers=group_identifiers,
            target_col=target_col,
            group_score_agg=group_score_agg,
            prior_mode=prior_mode,
            val_type_freq=val_type_freq,
            blend_mix=blend_mix,
            alpha=alpha,
            epsilon=epsilon,
            size_correction=size_correction,
            group_replacement=group_replacement,
            drop_last=drop_last,
            seed=seed,
        )

        train_loader = DataLoader(
            datasets["dataset_train"],
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            multiprocessing_context=multiprocessing_context,
        )
            

    elif sampler_name == "GroupedWeightedRandomSampler":
        # read from sampler_settings
        group_identifiers     = sampler_settings.get("group_identifiers", ["patient_id", "score_type", "left_or_right"])
        target_col            = sampler_settings.get("target_col", "score")
        group_score_agg       = sampler_settings.get("group_score_agg", "mean_ceil")  # or a callable
        replacement           = sampler_settings.get("replacement", True)
        drop_last             = sampler_settings.get("drop_last", False)
        seed                  = int(sampler_settings.get("seed", 0))
        min_primary_fraction  = float(sampler_settings.get("min_primary_fraction", 0.7))
        max_groups_per_batch  = int(sampler_settings.get("max_groups_per_batch", 3))
        min_primary_items     = sampler_settings.get("min_primary_items", None)
        repeats_per_group_cap = sampler_settings.get("repeats_per_group_cap", None)

        batch_sampler = GroupedWeightedRandomBatchSampler(
            dataset=datasets["dataset_train"],
            batch_size=batch_size,
            group_identifiers=group_identifiers,
            target_col=target_col,
            group_score_agg=group_score_agg,
            replacement=replacement,
            drop_last=drop_last,
            seed=seed,
            min_primary_fraction=min_primary_fraction,
            max_groups_per_batch=max_groups_per_batch,
            min_primary_items=min_primary_items,
            repeats_per_group_cap=repeats_per_group_cap,
        )

        train_loader = DataLoader(
            datasets["dataset_train"],
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            multiprocessing_context=multiprocessing_context,
        )



    elif sampler_name == "PatientBatchSampler":
        patient_ids  = [d["patient_id"] for d in datasets["dataset_train"].data]
        batch_sampler = PatientBatchSampler(patient_ids,
                                            batch_size=batch_size,
                                            drop_last=False)

        train_loader = DataLoader(datasets["dataset_train"],
                                  batch_sampler=batch_sampler,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  multiprocessing_context=multiprocessing_context)

    elif sampler_name == "PatientLevelWeightedRandomSampler":
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
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  multiprocessing_context=multiprocessing_context)

    elif sampler_name is None:
        train_loader = DataLoader(datasets["dataset_train"],
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=False,
                                  pin_memory=pin_memory,
                                  multiprocessing_context=multiprocessing_context)

    else:
        raise ValueError(f"Sampler name is not allowed {sampler_name = }. "
                         f"Use one of the implemented ones (e.g. 'WeightedRandomSampler', "
                         f"'TwoStageGroupedWeightedRandomBatchSampler', "
                         f"'PatientLevelWeightedRandomSampler', 'PatientBatchSampler', None)")

    # validation / test unchanged
    val_loader = DataLoader(datasets["dataset_validation"],
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=pin_memory)

    test_loader = DataLoader(datasets["dataset_test"],
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    train_loader_with_val_transforms = DataLoader(
        datasets["dataset_train_with_val_transforms"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory
    )

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




