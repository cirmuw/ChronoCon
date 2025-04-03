#!/usr/bin/env python3

import argparse
import pandas as pd
import os

def main():
    parser = argparse.ArgumentParser(description="Rename columns based on a mapping.")
    parser.add_argument("-i", "--input", required=True, help="Path to input CSV file.")
    parser.add_argument("-o", "--output", required=True, help="Path to output CSV file.")
    parser.add_argument("-type", "--type", choices=["H", "F"], required=True, help="Type of mapping to use: H or F.")
    parser.add_argument("-m", "--mapping", default=None, help="Optional path to custom column mapping CSV.")
    parser.add_argument("-s", "--switch_x_y", action="store_true", help="Switch '_x' with '_y' in column names.")
    
    args = parser.parse_args()

    # Default mapping paths
    default_mapping_H = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/landmark_data/100_all_H_joints36/columns_mapping.csv"
    default_mapping_F = "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/landmark_data/100_all_F_joints27/columns_mapping.csv"

    # Determine which mapping file to use
    mapping_path = args.mapping
    if mapping_path is None:
        mapping_path = default_mapping_H if args.type == "H" else default_mapping_F

    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    # Define the switching function
    def maybe_switch_suffix(col: str) -> str:
        if args.switch_x_y:
            if col.endswith("_x"):
                return col[:-2] + "_y"
            elif col.endswith("_y"):
                return col[:-2] + "_x"
        return col

    # Load mapping
    mapping_df = pd.read_csv(mapping_path, header=None)
    renaming_dict = {maybe_switch_suffix(k): v for k, v in mapping_df.transpose().values}

    # Load input CSV
    df = pd.read_csv(args.input)

    # Rename columns
    df.rename(columns=renaming_dict, inplace=True)

    # Save output
    df.to_csv(args.output, index=False)
    print(f"Renamed columns and saved to {args.output}")

if __name__ == "__main__":
    main()
