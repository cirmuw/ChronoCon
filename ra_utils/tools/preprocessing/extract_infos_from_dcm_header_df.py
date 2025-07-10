"""
Goal: 
Add derived columns to dataframe from DICOM header.
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
import re
from typing import Tuple

from ra_utils.data.crawler_helpers import scantree, get_and_maybe_save_crawler_data
import argparse

from ra_utils.data.dicom_helpers import (clean_patient_name, 
                                         body_part, 
                                         laterality, 
                                         view, 
                                         add_img_id_column,
                                         safe_str)


def _T_F_to_bool(value: str) -> bool:
    """Convert 'T'/'F' to boolean True/False."""
    if value.upper() == 'T':
        return True
    elif value.upper() == 'F':
        return False
    else:
        raise ValueError(f"Invalid value for boolean conversion: {value}. Use 'T' or 'F'.")


def main():
    
    path_to_crawler_metadata_combined__DEFAULT = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/tabular_data_cw/crawler_out_msc1/crawler_out_msc1_combined.csv"
    path_to_crawler_metadata_dirs_DEFAULT = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/tabular_data_cw/crawler_out_msc1/out/"
    out_path_DEFAULT = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/tabular_data_cw/crawler_out_msc1/out/"
    
    parser = argparse.ArgumentParser(
        description=(f"Add derived columns to dataframe from DICOM header. "
                     f"This script processes a CSV file generated from crawler metadata to derive additional columns based on DICOM header attributes. "
                     f"It supports loading metadata from a directory or a combined CSV file, and optionally saving the processed data back to CSV.")
    )
    parser.add_argument("--path_to_crawler_metadata_combined", type=str,
                        default=f"{path_to_crawler_metadata_combined__DEFAULT}",
                        help=f"Path to the combined crawler metadata CSV file (default: {path_to_crawler_metadata_combined__DEFAULT})")
    parser.add_argument("--path_to_crawler_metadata_dirs", type=str,
                        default=f"{path_to_crawler_metadata_dirs_DEFAULT}",
                        help=f"Path to the crawler metadata directories (default: {path_to_crawler_metadata_dirs_DEFAULT})")
    parser.add_argument("--reload_from_dir", type=str, default=f"F",
                        help="Reload data from directory (default: F)")
    parser.add_argument("--reload_from_combined_csv", type=str, default=f"T",
                        help="Reload data from combined CSV (default: T)")
    parser.add_argument("--save_to_combined_csv", type=str, default=f"F",
                        help="Save data to combined CSV (default: F)")
    parser.add_argument("--out_path", type=str,
                        default=f"{out_path_DEFAULT}",
                        help=f"Output path (default: {out_path_DEFAULT})")

    args = parser.parse_args()

    path_to_crawler_metadata_combined = args.path_to_crawler_metadata_combined
    path_to_crawler_metadata_dirs = args.path_to_crawler_metadata_dirs
    reload_from_dir = _T_F_to_bool(args.reload_from_dir)
    reload_from_combined_csv = _T_F_to_bool(args.reload_from_combined_csv )
    save_to_combined_csv = _T_F_to_bool(args.save_to_combined_csv)
    out_path = args.out_path
    
    

    cfg = {"path_to_crawler_metadata_dirs": path_to_crawler_metadata_dirs, 
           "path_to_crawler_metadata_combined": path_to_crawler_metadata_combined}
    df_crawler = get_and_maybe_save_crawler_data(cfg,
                                reload_from_dir=reload_from_dir,
                                reload_from_combined_csv=reload_from_combined_csv,
                                save_to_combined_csv=save_to_combined_csv)



    df = df_crawler
    df["derived__patientID"] = df.apply(lambda row: clean_patient_name(row.get("PatientName")), axis=1)
    df["derived__date"] = df.apply(lambda row: safe_str(row.get("StudyDate")) or safe_str(row.get("ContentDate")), axis=1)
    df["derived__body_part"] = df.apply(body_part, axis=1)
    df["derived__laterality"] = df.apply(laterality, axis=1)
    df["derived__view"] = df.apply(view, axis=1)
    df["img_id_partial"] = df["derived__patientID"] + "_" + df["derived__date"] #+ "_" + df["derived__body_part"]

    df = add_img_id_column(df)
    df.to_csv(out_path + "crawler_metadata_with_derived_columns.csv", index=False)
    print(f"Saved output CSV with derived columns to {out_path + 'crawler_metadata_with_derived_columns.csv'}")

if __name__ == "__main__":
    main()
    
    