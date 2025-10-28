#!/usr/bin/env python
"""
Check and analyze AgeDB dataset splits.

This script performs exploratory data analysis on AgeDB splits, including:
- Name/image frequency analysis
- Split exclusivity checks (ensuring persons don't appear in multiple splits)
- Split distribution statistics
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Check and analyze AgeDB dataset splits"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to the input CSV file with split information"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Directory to save output plots (default: same directory as input file)"
    )
    parser.add_argument(
        "--person-col",
        type=str,
        default="name",
        help="Name of the column containing person identifiers (default: 'name')"
    )
    parser.add_argument(
        "--split-col",
        type=str,
        default="split",
        help="Name of the column containing split information (default: 'split')"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots"
    )
    
    return parser.parse_args()


def load_data(input_path):
    """Load the CSV file and perform basic validation."""
    print(f"Loading data from: {input_path}")
    
    if not Path(input_path).exists():
        print(f"❌ Error: File not found: {input_path}")
        sys.exit(1)
    
    try:
        df = pd.read_csv(input_path)
        print(f"✅ Successfully loaded {len(df)} rows")
        return df
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        sys.exit(1)


def show_basic_info(df):
    """Display basic information about the dataset."""
    print("\n" + "="*80)
    print("BASIC DATASET INFORMATION")
    print("="*80)
    
    print(f"\nDataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Show first few rows if key columns exist
    display_cols = []
    for col in ["image_id", "name", "age", "sex", "split", "path"]:
        if col in df.columns:
            display_cols.append(col)
    
    if display_cols:
        print(f"\nFirst 5 rows ({', '.join(display_cols)}):")
        print(df[display_cols].head())


def analyze_name_frequencies(df, person_col, output_dir, no_plots):
    """Analyze and plot name/person frequencies."""
    print("\n" + "="*80)
    print("NAME FREQUENCY ANALYSIS")
    print("="*80)
    
    if person_col not in df.columns:
        print(f"❌ Error: Column '{person_col}' not found in dataset")
        return
    
    name_freq_counts = df[person_col].value_counts()
    
    print(f"\nTotal unique persons: {len(name_freq_counts)}")
    print(f"Images per person statistics:")
    print(f"  Min:    {name_freq_counts.min()}")
    print(f"  Max:    {name_freq_counts.max()}")
    print(f"  Mean:   {name_freq_counts.mean():.2f}")
    print(f"  Median: {name_freq_counts.median():.2f}")
    
    # Show top 10 persons with most images
    print(f"\nTop 10 persons with most images:")
    for idx, (name, count) in enumerate(name_freq_counts.head(10).items(), 1):
        print(f"  {idx:2d}. {name}: {count} images")
    
    if not no_plots:
        # Create histogram
        plt.figure(figsize=(10, 6))
        plt.hist(name_freq_counts, bins=range(1, name_freq_counts.max()+2), edgecolor='k', alpha=0.7)
        plt.xlabel("Number of images per person", fontsize=12)
        plt.ylabel("Number of persons", fontsize=12)
        plt.title("Distribution: Number of images per individual", fontsize=14, fontweight='bold')
        plt.xticks(range(1, name_freq_counts.max()+1, max(1, name_freq_counts.max()//20)))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_dir) / "name_frequency_histogram.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Saved plot to: {output_path}")
        plt.close()


def check_split_exclusivity(df, person_col, split_col):
    """Check if persons appear exclusively in one split."""
    print("\n" + "="*80)
    print("SPLIT EXCLUSIVITY ANALYSIS")
    print("="*80)
    
    if person_col not in df.columns:
        print(f"❌ Error: Column '{person_col}' not found in dataset")
        return False
    
    if split_col not in df.columns:
        print(f"❌ Error: Column '{split_col}' not found in dataset")
        return False
    
    print(f"Analyzing: {person_col} vs {split_col}")
    
    # Show available splits
    unique_splits = df[split_col].unique()
    print(f"\nAvailable splits: {sorted(unique_splits)}")
    
    # Group by person and check their splits
    person_splits = df.groupby(person_col)[split_col].apply(set).reset_index()
    person_splits.columns = [person_col, 'splits']
    
    # Count how many splits each person appears in
    person_splits['num_splits'] = person_splits['splits'].apply(len)
    
    print(f"\nTotal unique persons: {len(person_splits)}")
    print(f"\nPersons per number of splits:")
    split_dist = person_splits['num_splits'].value_counts().sort_index()
    for num_splits, count in split_dist.items():
        print(f"  {num_splits} split(s): {count} persons")
    
    # Check for persons appearing in multiple splits
    multi_split_persons = person_splits[person_splits['num_splits'] > 1]
    
    all_exclusive = len(multi_split_persons) == 0
    
    if not all_exclusive:
        print(f"\n⚠️  WARNING: Found {len(multi_split_persons)} persons appearing in multiple splits!")
        print("\nFirst 10 violating persons:")
        for _, row in multi_split_persons.head(10).iterrows():
            splits_str = ', '.join(sorted(row['splits']))
            print(f"  {row[person_col]}: [{splits_str}]")
        if len(multi_split_persons) > 10:
            print(f"  ... and {len(multi_split_persons) - 10} more")
    else:
        print("\n✅ PASSED: All persons appear exclusively in one split!")
    
    # Show split distribution
    print(f"\nSplit distribution (number of images):")
    split_counts = df[split_col].value_counts().sort_index()
    total = len(df)
    for split_name, count in split_counts.items():
        percentage = (count / total) * 100
        print(f"  {split_name:10s}: {count:6d} images ({percentage:5.2f}%)")
    print(f"  {'Total':10s}: {total:6d} images")
    
    # Show split distribution by persons
    print(f"\nSplit distribution (number of unique persons):")
    person_split_dist = person_splits[person_splits['num_splits'] == 1].copy()
    person_split_dist['split'] = person_split_dist['splits'].apply(lambda x: list(x)[0])
    person_counts = person_split_dist['split'].value_counts().sort_index()
    total_persons = len(person_split_dist)
    for split_name, count in person_counts.items():
        percentage = (count / total_persons) * 100
        print(f"  {split_name:10s}: {count:6d} persons ({percentage:5.2f}%)")
    print(f"  {'Total':10s}: {total_persons:6d} persons")
    
    # Show sample persons and their splits
    print(f"\nSample of 10 random persons and their splits:")
    sample_persons = person_splits.sample(min(10, len(person_splits)))
    for _, row in sample_persons.iterrows():
        splits_str = ', '.join(sorted(row['splits']))
        print(f"  {row[person_col]:20s}: [{splits_str}]")
    
    return all_exclusive


def main():
    """Main execution function."""
    args = parse_args()
    
    # Load data
    df = load_data(args.input)
    
    # Set output directory
    if args.output_dir is None:
        output_dir = Path(args.input).parent
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analyses
    show_basic_info(df)
    analyze_name_frequencies(df, args.person_col, output_dir, args.no_plots)
    split_exclusive = check_split_exclusivity(df, args.person_col, args.split_col)
    
    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if split_exclusive:
        print("✅ All checks passed!")
        return 0
    else:
        print("⚠️  Some checks failed - please review the warnings above")
        return 1


if __name__ == "__main__":
    sys.exit(main())

