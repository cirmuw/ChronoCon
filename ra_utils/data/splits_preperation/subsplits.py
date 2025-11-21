### Prepare subsplits of training data on patient level for RA dataset. 
# Goal: The splits need to be used to see how well the model scales if it os only trained on a fraction of the images. 
#       This is especially important for our unsupervised case. We need to check if our unsupervised training scheme works better in low label setting. 

# 1) Read split file with patient infos (Read all for now)
# 2) Merge with images. Obtain meta infos
#    How many images per patient? 
# 3) Use only training patients and  (split = "Tr")
# Create num_buckets=5 regardings number of images. 
# E.g. bucket_0=[patient005, patient012, ... ]   contains patients with 1, ... count_b0 images 
#      bucket_1                                  contains patients with count_b0, ... count_b1 images 
#      and so on... 
# 4) Create splits according to fraction recursively
# split_buckets_0 = random_split_bucket(bucket_0, fraction = 0.5, num_splits = 6, seed=42) -> Dict
# 
# 5) Merge together again
# 6) Make folder. Make sure it does not exist yet. Otherwise error and save 





from typing import List, Dict
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import ra_utils

from ra_utils.data.dataloader_CR_patches import (
    process_several_score_groups,
    dataset_and_loader_several,
    check_duplicates_in_dataloader
)





config_data = {
    "score_groups": {
#        "ALL": ['PIPIII', 'PIPIIIED', 'PIPIIIEP', 'PIPII']
        # 'H_MSCPIPIII_JSN_ERO': ["PIPIII", "MCPIII", 'PIPIIIED', 'PIPIIIEP', 'MCPIIIED', 'MCPIIIEP'],
        'ALL': ['IPED', 'IPEP', 
                'MTPIED', 'MTPIEP', 'MTPIIED', 'MTPIIEP', 'MTPIIIED', 'MTPIIIEP', 'MTPIVED', 'MTPIVEP', 'MTPVED', 'MTPVEP', 
                'CMCIII', 'CMCIV', 'CMCV', 
                'MCPI', 'MCPII', 'MCPIII', 'MCPIV', 'MCPV', 
                'PIPII', 'PIPIII', 'PIPIV', 'PIPV', 
                'IP', 
                'MTPI', 'MTPII', 'MTPIII', 'MTPIV', 'MTPV', 
                'IPIED', 'IPIEP', 
                'MCPIED', 'MCPIEP', 'MCPIIED', 'MCPIIEP', 'MCPIIIED', 'MCPIIIEP', 'MCPIVED', 'MCPIVEP', 'MCPVED', 'MCPVEP', 
                'PIPIIED', 'PIPIIEP', 'PIPIIIED', 'PIPIIIEP', 'PIPIVED', 'PIPIVEP', 'PIPVED', 'PIPVEP',
                'Rad_Carp', 'Sca_Cap', 'Tra_Sca',
                'RadiusE', 'ScaphE', 'TrapE', 'UlnaE', 'Base_MCIE', 'LunatE'
               ]
    },

    # "mean_round_delta_bin" # "mean_ceil" # "median_score"  Null  "delta_range"   "delta_buckets" "mean_ceil"  "mean_round" "delta_bin"
    "patient_level_class_balance_aggregation_rule": "mean_ceil",   # "mean_round_delta_bin" 
    "delta_buckets_k": 5,

    "enable_tSLR": False,
    "tSLR_time_scale": 0.0,
    "tSLR_forward_scale": 1.0,
    "tSLR_backward_scale": 1.0,

    # options for how_to_deal_with_surgery:
    # - "exclude"
    # - "keep as is"
    # - "keep: map over limit to limit plus one"
    # - "keep: map over limit to limit"
    "how_to_deal_with_surgery": "keep as is",
    "image_path_folder": [
        "/home/cwatzenboeck/data/AutoPIX_local_data/dev_cw/ds_480/H_patches_size1.5/",
        "/home/cwatzenboeck/data/AutoPIX_local_data/dev_cw/ds_480/F_patches_size1.5/"
    ],
    "pattern": "*/*.npy",

    "score_path_F": "/home/cwatzenboeck/data/AutoPIX_local_data/tabular/autoscoRA_data/autoscoRA_feet.csv",
    "score_path_H": "/home/cwatzenboeck/data/AutoPIX_local_data/tabular/autoscoRA_data/autoscoRA_hands.csv",
    "splits_file": "/home/cwatzenboeck/data/AutoPIX_local_data/tabular/autoscoRA_data/splits_info/df_patient_ids_split.csv",
    "sum_wrist_points": False,
    "surgery_patientids_list_path_json": None,
    "use_splits_file": True
}





def random_split_bucket(bucket: List["str"], fraction = 0.5, num_splits = 6, seed=42) -> Dict:
    #  shuffle bucket
    bucket = np.array(bucket)
    indices = np.arange(len(bucket))
    # shuffle with seed for reproducibility
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    
    r = {}
    indices_part = indices
    for i_split in range(num_splits):
        len_new = int(len(indices_part) * (fraction))
        indices_part = indices_part[:len_new]
        r[i_split] = bucket[indices_part]
    return r
 



 
def create_buckets(df: pd.DataFrame, num_buckets: int = 5) -> Dict[int, List[str]]:
    """
    Create buckets with equal number of patients based on num_visits (number of unique dates per patient).
    Patients are sorted by num_visits first, then split into equal-sized buckets.
    
    Args:
        df: DataFrame with columns 'patient_id' and 'num_visits'
        num_buckets: Number of buckets to create
    
    Returns:
        Dictionary mapping bucket index to list of patient IDs
    """
    buckets = {}
    
    # Sort by num_visits to maintain some ordering
    sorted_df = df.sort_values("num_visits").reset_index(drop=True)
    total_patients = len(sorted_df)
    
    # Calculate bucket size (approximately equal)
    patients_per_bucket = total_patients // num_buckets
    remainder = total_patients % num_buckets
    
    # Split into buckets
    start_idx = 0
    for i_bucket in range(num_buckets):
        # Distribute remainder across first few buckets
        bucket_size = patients_per_bucket + (1 if i_bucket < remainder else 0)
        end_idx = start_idx + bucket_size
        
        # Get patient IDs for this bucket
        bucket_patients = sorted_df.iloc[start_idx:end_idx]["patient_id"].astype(str).tolist()
        buckets[i_bucket] = bucket_patients
        
        start_idx = end_idx
    
    return buckets


def merge_subsplits(all_bucket_splits: Dict[int, Dict[int, List[str]]], num_splits: int) -> Dict[int, List[str]]:
    """
    Merge subsplits from all buckets into final splits.
    
    Args:
        all_bucket_splits: Dictionary mapping bucket index to its subsplits
        num_splits: Number of splits
    
    Returns:
        Dictionary mapping split index to list of patient IDs
    """
    merged_splits = {}
    for i_split in range(num_splits):
        merged_patients = []
        for i_bucket, bucket_splits in all_bucket_splits.items():
            merged_patients.extend(bucket_splits[i_split])
        merged_splits[i_split] = merged_patients
    return merged_splits


def save_subsplits(merged_splits: Dict[int, List[str]], val_patient_ids: List[str], 
                   test_patient_ids: List[str], output_dir: str, fraction: float, 
                   num_splits: int, seed: int):
    """
    Save subsplits to CSV files in output directory.
    Each file contains train subsplit, val, and test patients combined.
    Only the training subsplit varies between files; val and test remain the same.
    
    Args:
        merged_splits: Dictionary mapping split index to list of patient IDs
        val_patient_ids: List of validation patient IDs
        test_patient_ids: List of test patient IDs
        output_dir: Output directory path
        fraction: Fraction used for splitting
        num_splits: Number of splits
        seed: Random seed used
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    
    # Check if directory already exists
    if output_path.exists():
        raise ValueError(f"Output directory already exists: {output_dir}. Please remove it first or choose a different directory.")
    
    output_path.mkdir(parents=True, exist_ok=False)
    
    # Ensure unique patient IDs for val and test
    unique_val_ids = list(set(val_patient_ids))
    unique_test_ids = list(set(test_patient_ids))
    
    # Create val and test DataFrames (same for all files)
    df_val = pd.DataFrame({
        "patient_id": unique_val_ids,
        "set_type": "Val"
    })
    df_test = pd.DataFrame({
        "patient_id": unique_test_ids,
        "set_type": "Ts"
    })
    
    # Save each combined split file
    for i_split, patient_ids in merged_splits.items():
        # Ensure unique patient IDs for train subsplit
        unique_train_ids = list(set(patient_ids))
        
        # Create train subsplit DataFrame
        df_train = pd.DataFrame({
            "patient_id": unique_train_ids,
            "set_type": "Tr"
        })
        
        # Combine train, val, and test
        df_combined = pd.concat([df_train, df_val, df_test], ignore_index=True)
        
        # Ensure unique patient_ids (should already be unique, but double-check)
        df_combined = df_combined.drop_duplicates(subset=["patient_id"], keep="first")
        
        # Save combined file
        output_file = output_path / f"splits_partial_{i_split}.csv"
        df_combined.to_csv(output_file, index=False)
        print(f"Saved {len(unique_train_ids)} train + {len(unique_val_ids)} val + {len(unique_test_ids)} test patients to {output_file}")
    
    # Save metadata
    metadata = {
        "fraction": fraction,
        "num_splits": num_splits,
        "seed": seed,
        "total_train_patients": sum(len(set(patients)) for patients in merged_splits.values()),
        "total_val_patients": len(unique_val_ids),
        "total_test_patients": len(unique_test_ids)
    }
    
    metadata_file = output_path / "metadata.txt"
    with open(metadata_file, "w") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nSaved metadata to {metadata_file}")


def main():
    """
    Main function to create subsplits of training data.
    
    Steps:
    1) Read split file with patient infos
    2) Merge with images. Obtain meta infos (num_visits per patient)
    3) Use only training patients (split = "Tr")
    4) Create num_buckets=5 regarding number of images (visits)
    5) Create splits according to fraction recursively
    6) Merge together again
    7) Make folder. Make sure it does not exist yet. Otherwise error and save
    """
    parser = argparse.ArgumentParser(description="Create subsplits of training data for RA dataset")
    parser.add_argument("--out_dir", type=str, required=False, 
                        default="/home/cwatzenboeck/data/AutoPIX_local_data/tabular/autoscoRA_data/splits_info/subsplits/run2",
                       help="Output directory for subsplits")
    parser.add_argument("--fraction", type=float, default=0.5,
                       help="Fraction for recursive splitting (default: 0.5)")
    parser.add_argument("--num_splits", type=int, default=6,
                       help="Number of subsplits to create (default: 6)")
    parser.add_argument("--num_buckets", type=int, default=5,
                       help="Number of buckets based on num_visits (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    

    from pprint import pprint
    args = parser.parse_args()
    
    print(f"Configuration:")
    pprint(args)
    
    
    # Step 1 & 2: Get patient ids and num_visits for training set
    print("Step 1-2: Processing data and obtaining patient info...")
    data_tables = process_several_score_groups(config_data)
    df = data_tables["ALL"]["df_train"]
    
    # Count unique dates (visits) per patient
    unique_dates_per_patient = df.groupby("patient_id")["date_str"].nunique()
    df_patients = unique_dates_per_patient.reset_index(name="num_visits")
    
    # Step 3: Use only training patients (already done since we use df_train)
    df_patients["set_type"] = "Tr"
    print(f"Found {len(df_patients)} training patients")
    
    # Step 4: Create buckets based on num_visits
    print(f"Step 4: Creating {args.num_buckets} buckets based on num_visits...")
    buckets = create_buckets(df_patients, num_buckets=args.num_buckets)
    
    for i_bucket, patient_list in buckets.items():
        # Convert patient_list back to same type as df_patients for comparison
        df_patients_str = df_patients.copy()
        df_patients_str["patient_id"] = df_patients_str["patient_id"].astype(str)
        num_visits_in_bucket = df_patients_str[df_patients_str["patient_id"].isin(patient_list)]["num_visits"]
        print(f"  Bucket {i_bucket}: {len(patient_list)} patients, "
              f"num_visits range: [{num_visits_in_bucket.min()}, {num_visits_in_bucket.max()}]")
    
    print(f"\nInitial bucket sizes:")
    for i_bucket, patient_list in buckets.items():
        print(f"  Bucket {i_bucket}: {len(patient_list)} patients")
    
    # Step 5: Create splits according to fraction recursively for each bucket
    print(f"Step 5: Creating subsplits with fraction={args.fraction}, num_splits={args.num_splits}, seed={args.seed}...")
    all_bucket_splits = {}
    for i_bucket, bucket_patients in buckets.items():
        print(f"  Processing bucket {i_bucket}...")
        bucket_splits = random_split_bucket(
            bucket_patients,
            fraction=args.fraction,
            num_splits=args.num_splits,
            seed=args.seed
        )
        all_bucket_splits[i_bucket] = bucket_splits
        # Print sizes for debugging
        for i_split, patients in bucket_splits.items():
            print(f"    Split {i_split}: {len(patients)} patients")
    
    # Print summary: patients per bucket at each subsplit iteration
    print(f"\nPatients per bucket at each subsplit iteration:")
    print(f"{'Subsplit':<10} " + " ".join([f"Bucket {i:<6}" for i in range(args.num_buckets)]) + " Total")
    print("-" * (10 + (args.num_buckets + 1) * 12))
    for i_split in range(args.num_splits):
        bucket_counts = []
        for i_bucket in range(args.num_buckets):
            count = len(all_bucket_splits[i_bucket][i_split])
            bucket_counts.append(count)
        total = sum(bucket_counts)
        print(f"{i_split:<10} " + " ".join([f"{count:<12}" for count in bucket_counts]) + f"{total}")
    
    # Step 6: Merge together again
    print("Step 6: Merging subsplits from all buckets...")
    merged_splits = merge_subsplits(all_bucket_splits, args.num_splits)
    
    # Read the other parts (val and test) and get unique patient IDs
    print("Reading validation and test sets...")
    df_val = data_tables["ALL"]["df_val"]
    df_test = data_tables["ALL"]["df_test"]
    
    # Get unique patient IDs for val and test
    val_patient_ids = df_val["patient_id"].unique().astype(str).tolist()
    test_patient_ids = df_test["patient_id"].unique().astype(str).tolist()
    
    print(f"Found {len(val_patient_ids)} unique validation patients")
    print(f"Found {len(test_patient_ids)} unique test patients")
    
    # Step 7: Save to folder
    print(f"Step 7: Saving to {args.out_dir}...")
    save_subsplits(merged_splits, val_patient_ids, test_patient_ids, 
                   args.out_dir, args.fraction, args.num_splits, args.seed)
    
    print("\nDone! Subsplit creation complete.")


if __name__ == "__main__": 
    main()
     