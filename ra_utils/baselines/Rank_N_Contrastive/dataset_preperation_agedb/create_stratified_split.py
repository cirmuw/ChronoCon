"""
Create Stratified Person-Level Split for AgeDB Dataset

This script creates a train/val/test split where:
- Each person appears exclusively in one split (no data leakage)
- Stratified by number of images per person
- Stratified by age
- Maintains approximately 70:15:15 split ratio
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

import argparse

import os
# ============================================================================
# CONFIGURATION
# ============================================================================

#src = "/home/cwatzenboeck/data/public/agedb/tabular/agedb_splits_RnC_paper.csv"


# src = "/home/cwatzenboeck/data/public/agedb/tabular/agedb_splits_RnC_paper_with_extras.csv"
# output_path = "/home/cwatzenboeck/data/public/agedb/tabular/agedb_splits_stratified_new.csv"




def safe_stratified_split(persons_df, train_size, val_size, test_size, random_state):
    """
    Perform stratified split with fallback for small groups.
    """
    # Count group sizes
    group_counts = persons_df['strat_key'].value_counts()
    
    # Separate persons into stratifiable and non-stratifiable
    stratifiable_mask = persons_df['strat_key'].isin(group_counts[group_counts >= 3].index)
    stratifiable = persons_df[stratifiable_mask].copy()
    non_stratifiable = persons_df[~stratifiable_mask].copy()
    
    print(f"Stratifiable persons: {len(stratifiable)} ({len(stratifiable)/len(persons_df)*100:.1f}%)")
    print(f"Non-stratifiable persons: {len(non_stratifiable)} ({len(non_stratifiable)/len(persons_df)*100:.1f}%)")
    
    # Split stratifiable persons
    if len(stratifiable) > 0:
        # First split: train vs (val+test)
        train_persons, temp_persons = train_test_split(
            stratifiable,
            train_size=train_size,
            stratify=stratifiable['strat_key'],
            random_state=random_state
        )
        
        # Second split: val vs test
        val_ratio = val_size / (val_size + test_size)
        val_persons, test_persons = train_test_split(
            temp_persons,
            train_size=val_ratio,
            stratify=temp_persons['strat_key'],
            random_state=random_state
        )
    else:
        train_persons = pd.DataFrame()
        val_persons = pd.DataFrame()
        test_persons = pd.DataFrame()
    
    # Split non-stratifiable persons randomly
    if len(non_stratifiable) > 0:
        # First split: train vs (val+test)
        train_non_strat, temp_non_strat = train_test_split(
            non_stratifiable,
            train_size=train_size,
            random_state=random_state
        )
        
        # Second split: val vs test
        val_ratio = val_size / (val_size + test_size)
        val_non_strat, test_non_strat = train_test_split(
            temp_non_strat,
            train_size=val_ratio,
            random_state=random_state
        )
        
        # Combine
        train_persons = pd.concat([train_persons, train_non_strat], ignore_index=True)
        val_persons = pd.concat([val_persons, val_non_strat], ignore_index=True)
        test_persons = pd.concat([test_persons, test_non_strat], ignore_index=True)
    
    return train_persons, val_persons, test_persons




def parse_args():
    parser = argparse.ArgumentParser(
        description="Create stratified person-level data splits for AgeDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
python create_stratified_split.py \\
-i /path/to/input.csv \\
-o /path/to/output.csv \\
--train-ratio 0.74 \\
--val-ratio 0.13 \\
--test-ratio 0.13
        """
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Input CSV file (with extracted infos, e.g., 02_agedb_paths_with_infos.csv)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output CSV file for split data'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.74,
        help='Ratio of data to use for training split (default: 0.74)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.13,
        help='Ratio of data to use for validation split (default: 0.13)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.13,
        help='Ratio of data to use for test split (default: 0.13)'
    )
    return parser.parse_args()

def main():
    

    args = parse_args()
    src = args.input
    output_path = args.output

    # Split ratios (train:val:test)
    TRAIN_RATIO = args.train_ratio
    VAL_RATIO = args.val_ratio
    TEST_RATIO = args.test_ratio


    total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if not abs(total_ratio - 1.0) < 1e-6:
        raise ValueError(f"Train/val/test ratios must sum to 1.0 (got {total_ratio:.6f})")

    if os.path.exists(output_path):
        raise FileExistsError(f"Output file already exists: {output_path}")


    # Random seed for reproducibility
    RANDOM_SEED = 42

    # Column names (update these based on your data)
    PERSON_COL = "name"  # or "person_id", "subject_id", etc.
    AGE_COL = "age"      # column containing age information
    SPLIT_COL = "split"  # column for the split assignment
    
    print(f"Configuration:")
    print(f"  Train: {TRAIN_RATIO*100:.0f}%")
    print(f"  Val:   {VAL_RATIO*100:.0f}%")
    print(f"  Test:  {TEST_RATIO*100:.0f}%")
    print(f"  Random seed: {RANDOM_SEED}")

    # ============================================================================
    # LOAD DATA
    # ============================================================================

    print(f"\n{'='*80}")
    print("LOADING DATA")
    print('='*80)

    df = pd.read_csv(src, encoding='utf-8')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())

    # ============================================================================
    # CREATE PERSON-LEVEL SUMMARY
    # ============================================================================

    print(f"\n{'='*80}")
    print("CREATING PERSON-LEVEL SUMMARY")
    print('='*80)

    # Count images per person
    person_image_counts = df[PERSON_COL].value_counts().reset_index()
    person_image_counts.columns = [PERSON_COL, 'num_images']

    # Get representative age for each person
    person_ages = df.groupby(PERSON_COL)[AGE_COL].agg(['mean', 'std', 'count']).reset_index()
    person_ages.columns = [PERSON_COL, 'age_mean', 'age_std', 'age_count']

    # Merge person info
    person_info = person_image_counts.merge(person_ages, on=PERSON_COL)

    print(f"Total unique persons: {len(person_info)}")
    print(f"Total images: {len(df)}")
    print(f"\nImages per person statistics:")
    print(person_info['num_images'].describe())
    print(f"\nAge statistics:")
    print(person_info['age_mean'].describe())

    # ============================================================================
    # CREATE STRATIFICATION BINS
    # ============================================================================

    print(f"\n{'='*80}")
    print("CREATING STRATIFICATION BINS")
    print('='*80)

    # Compute quantile edges for num_images and age_mean
    image_count_quantiles = person_info['num_images'].quantile([0.2, 0.4, 0.6, 0.8]).values
    age_mean_quantiles = person_info['age_mean'].quantile([0.2, 0.4, 0.6, 0.8]).values

    print("image_count_quantiles:: ", image_count_quantiles)
    print("age_mean_quantiles:: ", age_mean_quantiles)

    # Create bins - note: include -inf/inf to catch min/max
    image_count_bins = [-float('inf')] + list(image_count_quantiles) + [float('inf')]
    age_mean_bins = [-float('inf')] + list(age_mean_quantiles) + [float('inf')]

    # Create bin labels
    image_count_bin_labels = [f"Q{i+1}" for i in range(len(image_count_bins)-1)]
    age_mean_bin_labels = [f"Q{i+1}" for i in range(len(age_mean_bins)-1)]

    # Bin images per person using quantiles
    person_info['image_count_bin'] = pd.cut(
        person_info['num_images'],
        bins=image_count_bins,
        labels=image_count_bin_labels,
        include_lowest=True
    )

    # Bin age_mean using quantiles
    person_info['age_bin'] = pd.cut(
        person_info['age_mean'],
        bins=age_mean_bins,
        labels=age_mean_bin_labels,
        include_lowest=True
    )

    # Create combined stratification key
    person_info['strat_key'] = (person_info['image_count_bin'].astype(str) + '_' + 
                                person_info['age_bin'].astype(str))

    print(f"Number of stratification groups: {person_info['strat_key'].nunique()}")
    print(f"\nTop 20 stratification groups:")
    strat_counts = person_info['strat_key'].value_counts()
    print(strat_counts.head(20))

    # Check for small stratification groups
    small_groups = strat_counts[strat_counts < 3]
    if len(small_groups) > 0:
        print(f"\n⚠️  Warning: {len(small_groups)} stratification groups have less than 3 persons")
        print(small_groups)

    # ============================================================================
    # PERFORM STRATIFIED SPLIT
    # ============================================================================

    print(f"\n{'='*80}")
    print("PERFORMING STRATIFIED PERSON-LEVEL SPLIT")
    print('='*80)


    # Perform the split
    train_persons, val_persons, test_persons = safe_stratified_split(
        person_info,
        train_size=TRAIN_RATIO,
        val_size=VAL_RATIO,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED
    )

    print(f"\nSplit results (persons):")
    print(f"  Train: {len(train_persons)} persons ({len(train_persons)/len(person_info)*100:.1f}%)")
    print(f"  Val:   {len(val_persons)} persons ({len(val_persons)/len(person_info)*100:.1f}%)")
    print(f"  Test:  {len(test_persons)} persons ({len(test_persons)/len(person_info)*100:.1f}%)")

    # ============================================================================
    # ASSIGN SPLITS TO ALL IMAGES
    # ============================================================================

    print(f"\n{'='*80}")
    print("ASSIGNING SPLITS TO ALL IMAGES")
    print('='*80)

    # Create person -> split mapping
    train_persons['split'] = 'train'
    val_persons['split'] = 'val'
    test_persons['split'] = 'test'

    person_split_map = pd.concat([
        train_persons[[PERSON_COL, 'split']],
        val_persons[[PERSON_COL, 'split']],
        test_persons[[PERSON_COL, 'split']]
    ])

    # Merge with original dataframe (preserves ALL original columns)
    df_with_split = df.merge(person_split_map, on=PERSON_COL, how='left')

    # Verify all original columns are preserved
    print(f"\nOriginal columns: {len(df.columns)}")
    print(f"New dataframe columns: {len(df_with_split.columns)}")
    print(f"Columns in new dataframe: {list(df_with_split.columns)}")
    if 'path' in df_with_split.columns:
        print("✅ 'path' column is preserved!")

    # Verify no person appears in multiple splits
    person_split_check = df_with_split.groupby(PERSON_COL)['split'].nunique()
    multi_split_persons = person_split_check[person_split_check > 1]

    if len(multi_split_persons) > 0:
        print(f"⚠️  ERROR: {len(multi_split_persons)} persons appear in multiple splits!")
        print(multi_split_persons.head())
    else:
        print("✅ Verified: All persons appear exclusively in one split!")

    # Show split distribution (images)
    print(f"\nSplit distribution (images):")
    split_counts = df_with_split['split'].value_counts()
    print(split_counts)
    print(f"\nPercentages:")
    print(split_counts / len(df_with_split) * 100)

    # ============================================================================
    # VERIFY STRATIFICATION QUALITY
    # ============================================================================

    print(f"\n{'='*80}")
    print("VERIFYING STRATIFICATION QUALITY")
    print('='*80)

    for split_name in ['train', 'val', 'test']:
        split_persons = person_info[person_info[PERSON_COL].isin(
            df_with_split[df_with_split['split'] == split_name][PERSON_COL].unique()
        )]
        print(f"\n{split_name.upper()} statistics:")
        print(f"  Persons: {len(split_persons)}")
        print(f"  Images per person - mean: {split_persons['num_images'].mean():.2f}, std: {split_persons['num_images'].std():.2f}")
        print(f"  Age - mean: {split_persons['age_mean'].mean():.2f}, std: {split_persons['age_mean'].std():.2f}")

    # ============================================================================
    # SAVE RESULTS
    # ============================================================================

    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print('='*80)

    # Save to CSV with UTF-8 encoding
    df_with_split.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ Saved {len(df_with_split)} rows to {output_path}")

    # Save summary statistics
    summary_path = output_path.replace('.csv', '_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=== Stratified Split Summary ===\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Train: {TRAIN_RATIO*100:.0f}%\n")
        f.write(f"  Val:   {VAL_RATIO*100:.0f}%\n")
        f.write(f"  Test:  {TEST_RATIO*100:.0f}%\n")
        f.write(f"  Random seed: {RANDOM_SEED}\n\n")
        
        f.write(f"Results (persons):\n")
        f.write(f"  Train: {len(train_persons)} persons ({len(train_persons)/len(person_info)*100:.1f}%)\n")
        f.write(f"  Val:   {len(val_persons)} persons ({len(val_persons)/len(person_info)*100:.1f}%)\n")
        f.write(f"  Test:  {len(test_persons)} persons ({len(test_persons)/len(person_info)*100:.1f}%)\n\n")
        
        f.write(f"Results (images):\n")
        for split_name in ['train', 'val', 'test']:
            count = len(df_with_split[df_with_split['split'] == split_name])
            f.write(f"  {split_name.capitalize()}: {count} images ({count/len(df_with_split)*100:.1f}%)\n")
        
        f.write(f"\n✅ Verified: All persons appear exclusively in one split!\n")

    print(f"✅ Saved summary to {summary_path}")

    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================

    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print('='*80)
    print(f"\n✅ Successfully created person-level stratified split!")
    print(f"\nKey points:")
    print(f"  ✓ Each person appears exclusively in one split")
    print(f"  ✓ Stratified by number of images per person")
    print(f"  ✓ Stratified by age")
    print(f"  ✓ Split ratios approximately maintained")
    print(f"\nOutput files:")
    print(f"  - Main CSV: {output_path}")
    print(f"  - Summary:  {summary_path}")
    print(f"\n{'='*80}\n")



if __name__ == "__main__":
    main()