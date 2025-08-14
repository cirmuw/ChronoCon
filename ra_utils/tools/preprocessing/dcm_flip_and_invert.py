"""
Goal: 
Maybe invert color for MONOCHROME2 images -> all have black background, white foreground
Maybe flip L to L

save dicoms to save_dst_dir
"""


from pathlib import Path
import pandas as pd


import monai
from monai.data import Dataset, CacheDataset, DataLoader

import monai
from monai.data import PILReader
from monai.transforms import LoadImage, LoadImaged, Resized, Compose, SaveImage, Spacingd, SpatialCropd, ResizeWithPadOrCropd


import numpy as np
import os
import yaml

from tqdm import tqdm


import SimpleITK as sitk
import pydicom
import sys
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from tqdm import tqdm
#-----


from ra_utils.utils.xray_class import XRay
from datetime import datetime


import ra_utils.utils.config_parser


def filter_df(config, df):

    laterality = config.get("laterality")
    if laterality: 
        print(f"{set(df['derived__laterality']) = }")
        print("Using only ", laterality)
        m_lat = df["derived__laterality"].isin(laterality)
    else:
        m_lat = pd.Series(True, index=df.index)

    view = config.get("view")
    if view: 
        print(f"{set(df['derived__view']) = }")
        print("Using only ", view)
        m_view = df["derived__view"].isin(view)
    else: 
        m_view = pd.Series(True, index=df.index)


    PhotometricInterpretation = config.get("PhotometricInterpretation")
    if PhotometricInterpretation: 
        print(f"{set(df['PhotometricInterpretation']) = }")
        print("Using only ", PhotometricInterpretation)
        m_PI = df["PhotometricInterpretation"].isin(PhotometricInterpretation)
    else: 
        m_PI = pd.Series(True, index=df.index)


    body_part = config.get("body_part")
    if body_part: 
        print(f"{set(df['derived__body_part']) = }")
        print("Using only ", body_part)
        m_body_part = df["derived__body_part"].isin(body_part)
    else:
        m_body_part = pd.Series(True, index=df.index)


    # Filter the dataframe based on the conditions    
    m = m_lat & m_view & m_PI & m_body_part
    print(f"Before filtering:  {len(df)=}")
    df = df[m].reset_index(drop=True)


    print(f"  After filtering:  {len(df)=}")
    print(f"  {laterality = }")
    print(f"  {view = }")
    print(f"  {PhotometricInterpretation = }")
    print(f"  {body_part = }")
    return df


def main():
    config, config_name = ra_utils.utils.config_parser.load_config(
    default_config="/home/cwatzenboeck/code/RA/ra_utils/runs/config_preprocessing/config_flip_invert_H_dev.yml", 
    debugging_in_jupyter_nb=False, silencium=False, return_config_name=True, 
    )

    src = config["file_paths_csv"]
    df = pd.read_csv(src, low_memory=False)
    assert "file_path" in df.columns


    cols = ["derived__laterality", "derived__view", "derived__position", "derived__orientation"]
    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna("NA")
            
    
    root_from = config.get("reroot_file_path_dir__from")
    if root_from: 
        root_to = config["reroot_file_path_dir__to"]
        print(f"Rerooting file paths from {root_from} to {root_to}")
        df["file_path"] = df["file_path"].str.replace(root_from, root_to)
        

    
    
    df = filter_df(config, df)         
    
    OUT_DIR = config["out_dir"]
    OUT_DIR_DCM = Path(OUT_DIR) / "dcm"
    OUT_DIR_DCM.mkdir(parents=True, exist_ok=True)
    df["file_name_out"] = df["img_id"] + "__" + df["file_path"].apply(lambda x: Path(x).name)
    df["file_path_out"] = df["file_name_out"].apply(lambda x: Path(OUT_DIR_DCM)  / str(x + ".dcm"))


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # out_csv = Path(OUT_DIR) / f"dcm_flip_and_invert_{timestamp}.csv"
    out_csv = Path(OUT_DIR) / f"dcm_flip_and_invert.csv"
    if out_csv:
        out_csv = Path(out_csv)
        df.to_csv(out_csv, index=False)
        print(f"Saved output CSV to {out_csv}")

    config_yaml_path = Path(OUT_DIR) / f"config_flip_and_invert_{timestamp}.yaml"
    with open(config_yaml_path, 'w') as f:
        yaml.dump(config, f)
    print(f"Saved config to {config_yaml_path}")


    for i, file_i in tqdm(enumerate(df["file_path"])): 
        xray = XRay(file_i)
        if config.get("flip_RtoL", False): 
            if xray.img.Laterality == "R":
                xray.img_flip(flipped_axis=1)
        
        if config.get("flip_LtoR", False): 
            if xray.img.Laterality == "L":
                xray.img_flip(flipped_axis=1)
        
        if config.get("invert_M1toM2", False):
            if xray.img.PhotometricInterpretation == "MONOCHROME1":
                xray.img_monochrome(mono=2, maximum='maxbit', change_photometric_interpretation=True)

        if config.get("invert_M2toM1", False):
            if xray.img.PhotometricInterpretation == "MONOCHROME2":
                xray.img_monochrome(mono=1, maximum='maxbit', change_photometric_interpretation=True)
        
        file_path_out = df.loc[i, "file_path_out"]
        file_path_out.parent.mkdir(parents=True, exist_ok=True)
        xray.img.save_as(file_path_out, enforce_file_format=False)
        print(f"Saved {i+1}/{len(df)}: {file_path_out}")

        
        if config.get("debugging", False): 
            print("Debugging mode: Showing first image only")
            #xray.plot_img()
            print(xray.img.PhotometricInterpretation)
            print(xray.img.Laterality)
            print(xray.img.ViewPosition)    
            break 
        




if __name__ == "__main__":
    main()
    
    