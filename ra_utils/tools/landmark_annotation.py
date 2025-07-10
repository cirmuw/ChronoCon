import napari
import numpy as np
import pandas as pd
import pydicom
from pathlib import Path
from ra_utils.annotation.point_annotator import point_annotator, point_annotator_multi
import ra_utils.utils.config_parser

import pandas as pd
from pathlib import Path

def make_df_double_scoring(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a table that has one row per image and columns like
    'LM1-X', 'LM1-Y', … into the long format required by
    `point_annotator_multi`:

        y   x   label   file_name   dicom_path
        ─ ─ ─   ─────── ──────────  ──────────
        … …

    The function is agnostic to the landmark names – everything before the
    trailing '-X'/'-Y' is treated as a label.

    The input must contain either a ``dicom_path`` column, a ``file_name``
    column, or both.  Any rows with missing x *or* y are dropped.
    """
    if "dicom_path" not in df_wide.columns and "file_name" not in df_wide.columns:
        raise ValueError("Need at least one of 'dicom_path' or 'file_name'")

    # identify the coordinate columns
    x_cols = [c for c in df_wide.columns if c.endswith("-X")]
    y_cols = [c for c in df_wide.columns if c.endswith("-Y")]

    # melt into long form: separate tables for X and Y …
    df_x = df_wide.melt(
        id_vars=[c for c in ["dicom_path", "file_name"] if c in df_wide.columns],
        value_vars=x_cols,
        var_name="label",
        value_name="x",
    )
    df_x["label"] = df_x["label"].str[:-2]   # drop "-X"

    df_y = df_wide.melt(
        id_vars=[c for c in ["dicom_path", "file_name"] if c in df_wide.columns],
        value_vars=y_cols,
        var_name="label",
        value_name="y",
    )
    df_y["label"] = df_y["label"].str[:-2]   # drop "-Y"

    # … then combine X and Y
    df_long = (
        pd.merge(df_x, df_y, on=[*(set(df_x.columns) & set(df_y.columns))])
        .dropna(subset=["x", "y"])
        .loc[:, ["y", "x", "label", *[c for c in ["file_name", "dicom_path"] if c in df_wide.columns]]]
    )

    # ensure canonical column order
    return df_long.sort_values(["file_name" if "file_name" in df_long else "dicom_path", "label"]).reset_index(drop=True)


# === CONFIGURATION ===
def main():
    # 👇 Replace this with your actual DICOM path
    
    config, config_name = ra_utils.utils.config_parser.load_config(
        default_config="/home/clemens/data/AutoPIX_cirdata/projects__autoscora/tabular_data_cw/annotation_dir/day0/config_F.yml", 
        debugging_in_jupyter_nb=False, silencium=False, return_config_name=True, 
    )
    print("\n\n")
    
    df_file_names_src = config["file_paths_csv"]
    df_file_names = pd.read_csv(df_file_names_src)

    
    file_path_column = config.get("file_path_column", "file_path")
    if config.get("reroot_file_path_column", True) not in [None, False, ""]:
        reroot_file_path_dir__from = config["reroot_file_path_dir__from"]
        reroot_file_path_dir__to = config["reroot_file_path_dir__to"]
        print(f"Rerooting file paths (column = {file_path_column}) from {reroot_file_path_dir__from} to {reroot_file_path_dir__to}")
        df_file_names[file_path_column] = df_file_names[file_path_column].apply(lambda x: x.replace(reroot_file_path_dir__from, reroot_file_path_dir__to))

    
    print(f"Using file names from {df_file_names_src} (column = {file_path_column}).")
    print(f"Number of file names: {len(df_file_names)}")
    idx_start = config.get("idx_start", None)
    idx_end = config.get("idx_end", None)

    if idx_start is not None or idx_end is not None:
        print(f"Using file names from index {idx_start if idx_start is not None else 0} "
            f"to {idx_end if idx_end is not None else 'end'} (exclusive).")
        df_file_names = df_file_names.iloc[idx_start:idx_end]
    else:
        print("Using all file names (no index slicing applied).")


    idx_list = config.get("idx_list", None)
    if idx_list is not None:
        print(f"Using file names from index list {idx_list}.")
        df_file_names = df_file_names.iloc[idx_list]
        

    save_dst_dir = config["save_dst_dir"]
    
    
    DICOM_PATHS = df_file_names[file_path_column].tolist()
    
    LANDMARK_NAMES = config["LANDMARK_NAMES"]    

    
    # maybe display double scoring results: 
    if config.get("AddGT", False):
        try:
            df_double_scoring = df_double_scoring = make_df_double_scoring(df_file_names)
        except Exception as e:
            print(f"Error creating double scoring DataFrame: {e}")
            df_double_scoring = None
    else: 
        df_double_scoring = None
    
    point_annotator_multi(DICOM_PATHS, labels=LANDMARK_NAMES, 
                          save_dst_dir=save_dst_dir, 
                          df_double_scoring = df_double_scoring
                          )


if __name__ == "__main__":
    main()

    


