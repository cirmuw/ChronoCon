

"""
A simple script to convert model predictions of landmarks to correct format for correcting the annotations
"""


import pandas as pd
from pathlib import Path
from  ra_utils.tools.landmark_annotation import make_df_double_scoring
from  ra_utils.utils.config_parser import load_config



def main():
    config, config_name = load_config(
        default_config="/home/clemens/data/AutoPIX_cirdata/projects__autoscora/tabular_data_cw/annotation_dir/day0/config_F.yml", 
        debugging_in_jupyter_nb=False, silencium=True, return_config_name=True, 
    )


    save_dst_dir = config["save_dst_dir"]
    dst_path = Path(save_dst_dir)
    if dst_path.exists() and any(dst_path.iterdir()):
        answer = input(f"Warning: directory {dst_path} is not empty. Continue? (y/n): ")
        if answer.lower() != "y":
            print("Operation cancelled by user.")
            return
    else:
        dst_path.mkdir(parents=True, exist_ok=True)

    # Read the file paths we want to process only
    file_paths_csv = pd.read_csv(config["file_paths_csv"])[["file_path", "file_name"]].rename(columns={"file_path": "dicom_path"})


    # Merge with prepopulation file (landmarks predictions of model)
    df = pd.read_csv(config["prepopulation_file"])
    df["file_name"] = df["img"] + ".dcm"
    dfm = pd.merge(df, file_paths_csv, on="file_name", how="right")
    
    s1 = set(config["LANDMARK_NAMES"])
    for i in range(len(dfm)):
        df_part = dfm.iloc[i:i+1]
        df_part_reformated = make_df_double_scoring(df_part)
        s2 = set(df_part_reformated["label"])
        assert s1 - s2 == set(), f"Not all Landmark_names were present {s1-s2} missing!"    
        m = df_part_reformated["label"].isin(config["LANDMARK_NAMES"])
        df_part_reformated = df_part_reformated[m].reset_index(drop=True)
        dst_name = df_part["file_name"].values[0]
        dst_name = f"{save_dst_dir}/{dst_name}__landmarks.csv"
        if Path(dst_name).exists():
            answer_overwrite = input(f"File {dst_name} already exists. Overwrite? (y/n): ")
            if answer_overwrite.lower() != "y":
                print(f"Skipping file {dst_name}.")
                continue

        df_part_reformated.to_csv(dst_name, index=False)
        print(f"Saved to {dst_name}")


if __name__ == "__main__":
    main()

    


