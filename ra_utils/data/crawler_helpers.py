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

### Code to be refactored
def scantree(p):
    '''Path generator that recursivly yields DIREntry objects of file paths in a given directory.'''
    for entry in os.scandir(p):
        if entry.is_dir(follow_symlinks=False):
            yield from scantree(entry)
        else:
            yield entry.path
            
def get_and_maybe_save_crawler_data(config: dict, 
                             reload_from_dir = False,
                             reload_from_combined_csv = True,
                             save_to_combined_csv = False):
    assert not (reload_from_dir and reload_from_combined_csv), \
        "You cannot reload from both directory and combined CSV at the same time. Choose one."
    df = None
    
    if reload_from_dir:
        # Load all CSV files from the specified directory
        path2outputcsv = config["path_to_crawler_metadata_dirs"]
        csvs = [f for f in scantree(path2outputcsv) if '.csv' in f]
        print(f"Found {len(csvs)} CSV files in {path2outputcsv}")

        # Read and concatenate all CSV files
        all_dfs = []
        for csv_file in tqdm(csvs):
            df_temp = pd.read_csv(csv_file)
            all_dfs.append(df_temp)

        # Combine all dataframes
        df = pd.concat(all_dfs, ignore_index=True)
        
    if reload_from_combined_csv:
        # Load the combined CSV file
        path2combinedcsv = config["path_to_crawler_metadata_combined"]
        df = pd.read_csv(path2combinedcsv, low_memory=False)
        print(f"Loaded combined CSV with {len(df)} entries from {path2combinedcsv}")    
        
    if save_to_combined_csv:
        dst = config["path_to_crawler_metadata_combined"]
        df.to_csv(dst, index=False)
        print(f"Combined CSV saved to {dst}")
        
    return df
