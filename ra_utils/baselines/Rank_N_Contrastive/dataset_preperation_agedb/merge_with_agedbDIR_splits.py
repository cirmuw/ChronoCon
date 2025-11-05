# Imports: 
import pandas as pd
import numpy as np
from collections import Counter

import argparse


def merge_with_agedbDIR_splits(old_splits_file_to_use, metadata_file):



    df_DIR = pd.read_csv(old_splits_file_to_use)
    df_metadata = pd.read_csv(metadata_file)


    df_DIR["image_id"] = df_DIR["path"].apply(lambda x: int(x.split("/")[-1].split("_")[0]))

    assert df_DIR["image_id"].value_counts().max() == 1
    assert df_metadata["image_id"].value_counts().max() == 1

    df_DIR.drop(columns=["path"], inplace=True)
    df_merged = pd.merge(df_DIR, df_metadata, on="image_id", how="inner")

    assert (df_merged["age_x"]  == df_merged["age_y"]).all()

    # Since both "age_x" and "age_y" columns are equal, let's drop "age_y" and rename "age_x" to "age" for clarity.
    df_merged = df_merged.drop(columns=["age_y"])
    df_merged = df_merged.rename(columns={"age_x": "age"})


    return df_merged



def main():
    parser = argparse.ArgumentParser(description="Merge old AgeDB splits CSV with metadata CSV")
    parser.add_argument('--old_splits_file_to_use', type=str, required=True, help='Path to old splits file')
    parser.add_argument('--metadata_file', type=str, required=True, help='Path to AgeDB metadata file')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    args = parser.parse_args()

    old_splits_file_to_use = args.old_splits_file_to_use
    metadata_file = args.metadata_file
    output = args.output
    
    
    
    df_merged = merge_with_agedbDIR_splits(old_splits_file_to_use, metadata_file)
    df_merged.to_csv(output, index=False)
    print(f"Merged dataframe saved to {output}")


if __name__ == "__main__":
    main()
    
