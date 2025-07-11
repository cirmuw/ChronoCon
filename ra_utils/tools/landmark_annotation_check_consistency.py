

"""
A simple script to convert model predictions of landmarks to correct format for correcting the annotations
"""


import pandas as pd
from pathlib import Path
from  ra_utils.tools.landmark_annotation import make_df_double_scoring
from  ra_utils.utils.config_parser import load_config
import argparse



def main():
    parser = argparse.ArgumentParser(description="Landmark Annotation Check Consistency")
    parser.add_argument("--dir", required=True, help="Path to the Folder containing the csv files (e.g. out_H, out_F)")
    args = parser.parse_args()
    csv_dir = args.dir


    folders2combine = [csv_dir]
    for folder in folders2combine:
        if not Path(folder).exists():
            raise FileNotFoundError(f"Folder {folder} does not exist.")


    files = []
    for dir in folders2combine: 
        files.extend(list(Path(dir).glob("*__landmarks.csv")))

    df = []
    for f in files:
        df.append(pd.read_csv(f))
    df = pd.concat(df, ignore_index=True)
    df["img"] = df["file_name"].apply(lambda x: Path(x).stem)
    df = df[["img", "x", "y", "label"]]  
    print(f"Loaded {len(files) = }")



    vc = (df["img"]).value_counts()
    vc_median = vc.median()
    vc_diff = vc[vc != vc_median]
    if len(vc_diff) > 0: 
        print(f"Median count: {vc_median}")
        print(f"Entries with counts different from the median (={vc_median}):")
        print(vc_diff)
        print("\n\n")

    dupes = df[df.duplicated(subset=["img", "label"], keep=False)]
    if len(dupes) > 0:
        print("Duplicates: ")
        print(dupes.sort_values(["img", "label"]).head())
        raise ValueError("Duplicate labels are not allowed!")


    df_wide = (
        df.pivot(index="img", columns="label", values=["x", "y"])
        .sort_index(axis=1, level=1)  # Optional: sort labels alphabetically
    )

    # Flatten the multi-index columns
    df_wide.columns = [f"{label}-{axis.upper()}" for axis, label in df_wide.columns]
    df_wide = df_wide.reset_index()

    
    print("Excellent! No duplicates found and number of labels is consistent")
    



if __name__ == "__main__":
    main()

    


