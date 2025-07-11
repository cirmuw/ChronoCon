"""
A simple script to convert model predictions of landmarks to correct format for correcting the annotations
"""

import pandas as pd
from pathlib import Path
from ra_utils.tools.landmark_annotation import make_df_double_scoring
from ra_utils.utils.config_parser import load_config
import argparse
import yaml


def main():
    parser = argparse.ArgumentParser(description="Update splits file by adding new training cases")
    parser.add_argument("--splits_origin", required=True, help="Path to the original splits file (read only)")
    parser.add_argument("--splits_new", required=True, help="Path to the new splits file")
    parser.add_argument("--landmarks_data_csv", required=True, help="Path to the landmark data CSV file (read only)")
    args = parser.parse_args()

    with open(args.splits_origin, 'r') as f:
        src_data = yaml.safe_load(f)
    scr_data_notes = src_data.get("notes", "")

    src_data_new = src_data.copy()
    if "cv" in src_data_new:
        del src_data_new["cv"]

    df = pd.read_csv(args.landmarks_data_csv)
    ids_all = set(df["img"])
    ids_tr_new = (ids_all - set(src_data.get("test1", []))) - set(src_data.get("test2", []))

    src_data_new["training"] = list(ids_tr_new)

    print("Original dict lengths:")
    for key in src_data:
        print(f"{key}: {len(src_data[key])}")

    print("\nNew dict lengths:")
    for key in src_data_new:
        print(f"{key}: {len(src_data_new[key])}")
    src_data_new["notes"] = f"{scr_data_notes}\n\nUpdated with new training cases from {args.landmarks_data_csv}\n \nOriginal splits file: {args.splits_origin}"

    with open(args.splits_new, 'w') as f:
        yaml.dump(src_data_new, f)
    print(f"Saved to {args.splits_new}")


if __name__ == "__main__":
    main()

    


