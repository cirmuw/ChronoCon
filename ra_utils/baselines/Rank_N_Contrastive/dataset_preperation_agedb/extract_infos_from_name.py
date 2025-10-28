#!/usr/bin/env python3
"""
Extract information (ID, name, age, sex) from AgeDB filenames.

Filename format expected: <image_id>_<name>_<age>_<sex>.jpg
Example: 1222_ClaudeLéviStrauss_30_m.jpg
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path 


def extract_info_from_file_name(file_name: str, i = 0):
    try:
        fname_no_ext = Path(file_name).stem  # remove .jpg extension 
        parts = fname_no_ext.split("_")
        return parts[i]
    except Exception as e:
        return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract information (ID, name, age, sex) from AgeDB filenames',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python extract_infos_from_name.py \\
    -i /path/to/input.csv \\
    -o /path/to/output.csv
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Input CSV file with file paths (no header expected, first column should be paths)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output CSV file to save extracted information'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite output file if it already exists'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Check if output file already exists
    if os.path.exists(args.output) and not args.force:
        print(f"❌ Error: Output file already exists: {args.output}")
        print(f"   Use --force to overwrite or choose a different output path.")
        sys.exit(1)
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file does not exist: {args.input}")
        sys.exit(1)
    
    print(f"{'='*80}")
    print("EXTRACT INFORMATION FROM FILENAMES")
    print('='*80)
    
    # Load data
    print(f"\n📂 Loading data from: {args.input}")
    try:
        df = pd.read_csv(args.input, header=None, names=["abs_path"], encoding="utf-8")
        print(f"✅ Loaded {len(df)} rows")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        sys.exit(1)
    
    # Extract filename
    print(f"\n🔍 Extracting information from filenames...")
    df["file_name"] = df["abs_path"].apply(lambda x: Path(x).name)
    # Add "path" as the last folder and file name (relative from input root)
    df["path"] = df["abs_path"].apply(lambda x: os.path.join(Path(x).parent.name, Path(x).name))
    
    # Extract individual components
    print(f"   - Extracting image IDs...")
    df["image_id"] = df["file_name"].apply(lambda x: extract_info_from_file_name(x, i=0))
    df["image_id"] = df["image_id"].apply(lambda x: int(x))
    
    print(f"   - Extracting names...")
    df["name"] = df["file_name"].apply(lambda x: extract_info_from_file_name(x, i=1))
    
    print(f"   - Extracting ages...")
    df["age"] = df["file_name"].apply(lambda x: extract_info_from_file_name(x, i=2))
    df["age"] = df["age"].apply(lambda x: int(x))
    
    
    print(f"   - Extracting sex...")
    df["sex"] = df["file_name"].apply(lambda x: extract_info_from_file_name(x, i=3))
    
    # Show summary
    print(f"\n📊 Extraction summary:")
    print(f"   Total rows: {len(df)}")
    print(f"   Unique image IDs: {df['image_id'].nunique()}")
    print(f"   Unique names: {df['name'].nunique()}")
    print(f"   Age range: {df['age'].dropna().unique()[:10]}{'...' if df['age'].nunique() > 10 else ''}")
    print(f"   Sex distribution: {df['sex'].value_counts().to_dict()}")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"\n⚠️  Warning: Missing values detected:")
        for col, count in missing[missing > 0].items():
            print(f"   {col}: {count} missing")
    
    # Save data
    print(f"\n💾 Saving data to: {args.output}")
    try:
        df.to_csv(args.output, index=False, encoding='utf-8')
        print(f"✅ Successfully saved {len(df)} rows with {len(df.columns)} columns")
        print(f"   Columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print("✅ COMPLETE")
    print('='*80)


if __name__ == "__main__":
    main()
