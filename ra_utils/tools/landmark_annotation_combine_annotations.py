

"""
A simple script to convert model predictions of landmarks to correct format for correcting the annotations
"""


import pandas as pd
from pathlib import Path
from  ra_utils.tools.landmark_annotation import make_df_double_scoring
from  ra_utils.utils.config_parser import load_config



def main():
    config, config_name = load_config(
        default_config="/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/tabular_data_cw/annotation_dir/combined_results/config_F_Ts.yml", 
        debugging_in_jupyter_nb=False, silencium=False, return_config_name=True, 
    )


    save_dst_csv = config["save_dst_csv"]
    folders2combine = config["folders2combine"]
    other_lm_files = config.get("other_lm_files", [])
    LANDMARK_NAMES = config["LANDMARK_NAMES"]

    destination_path = Path(save_dst_csv)
    assert not destination_path.exists(), f"The file {destination_path} already exists."



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

    dupes = df[df.duplicated(subset=["img", "label"], keep=False)]
    if len(dupes) > 0:
        print("Duplicates: ")
        print(dupes.sort_values(["img", "label"]).head())


    df_wide = (
        df.pivot(index="img", columns="label", values=["x", "y"])
        .sort_index(axis=1, level=1)  # Optional: sort labels alphabetically
    )

    # Flatten the multi-index columns
    df_wide.columns = [f"{label}-{axis.upper()}" for axis, label in df_wide.columns]

    df_wide = df_wide.reset_index()


    columns = ["img"] + sum([[f"{l}-X", f"{l}-Y"] for l in LANDMARK_NAMES], [])
    
    if len(other_lm_files) > 0: 
        df_other_lm_files = pd.concat([pd.read_csv(f) for f in other_lm_files], ignore_index=True)
        df_combined = pd.concat([df_wide[columns], df_other_lm_files[columns]], ignore_index=True)
    else: 
        df_combined = df_wide[columns]
    df_combined.to_csv(save_dst_csv, index=False)
    print("Saved to ", save_dst_csv)



if __name__ == "__main__":
    main()

    


