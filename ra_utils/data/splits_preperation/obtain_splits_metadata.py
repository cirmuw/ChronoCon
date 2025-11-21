### Prepare subsplits of training data on patient level for RA dataset. 
# Goal: Read in a --splits_file from the command line. The other infos are in the config_data dict (hardcoded)
# Read --out_dir=None per default (This is where the metadata is saved as yaml file) In non just printed. 
# Get metadata: 
# "Tr":
#   "num_patients": ... 
#   "num_images": ...   
#   "num_score_types": ... 
#   "avg_num_images_per_score_type": ...
#   "detailed":
#      "PIPII": 
#         "num_images": 
#      "PIPIII": 
#          "num_images": 
# ... 


from typing import Dict, List
import pandas as pd
import argparse
from pathlib import Path
import yaml
from pprint import pprint
from importlib import resources

from ra_utils.data.dataloader_CR_patches import (
    process_several_score_groups,
)





def load_score_groupings() -> Dict[str, List[str]]:
    """
    Load score type groupings from resources file.
    
    Returns:
        Dictionary mapping group names (ERO_H, ERO_F, JSN_H, JSN_F) to lists of score names
    """
    with resources.files("ra_utils.resources.scores_metadata").joinpath("roi_scores_matching.csv") as f:
        df_scores_meta = pd.read_csv(f)
    
    H_ERO_scores = sorted(df_scores_meta[(df_scores_meta["ERO_or_JSN"] == "ERO") & (df_scores_meta["region"] == "H")]["score_name"].unique())
    F_ERO_scores = sorted(df_scores_meta[(df_scores_meta["ERO_or_JSN"] == "ERO") & (df_scores_meta["region"] == "F")]["score_name"].unique())
    H_JSN_scores = sorted(df_scores_meta[(df_scores_meta["ERO_or_JSN"] == "JSN") & (df_scores_meta["region"] == "H")]["score_name"].unique())
    F_JSN_scores = sorted(df_scores_meta[(df_scores_meta["ERO_or_JSN"] == "JSN") & (df_scores_meta["region"] == "F")]["score_name"].unique())
    
    return {
        "ERO_H": H_ERO_scores,
        "ERO_F": F_ERO_scores,
        "JSN_H": H_JSN_scores,
        "JSN_F": F_JSN_scores
    }


def calculate_aggregated_score_distribution(df: pd.DataFrame, score_types: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Calculate aggregated score distribution for a group of score types.
    
    Args:
        df: DataFrame with columns including 'chosen_score' and 'score'
        score_types: List of score type names to aggregate
    
    Returns:
        Dictionary with 'score_distribution' and 'score_distribution_fractions'
    """
    # Filter dataframe for the specified score types
    df_group = df[df["chosen_score"].isin(score_types)]
    
    if len(df_group) == 0:
        return {
            "score_distribution": {},
            "score_distribution_fractions": {}
        }
    
    # Calculate aggregated score distribution
    scores = df_group["score"].dropna()
    score_counts = scores.value_counts().sort_index()
    num_images_total = len(df_group)
    
    # Convert score keys to strings for YAML compatibility
    score_distribution = {str(int(k)): int(v) for k, v in score_counts.items()}
    
    # Calculate score distribution fractions (proportions)
    score_distribution_fractions = {
        str(int(k)): float(v) / num_images_total 
        for k, v in score_counts.items()
    }
    
    return {
        "score_distribution": score_distribution,
        "score_distribution_fractions": score_distribution_fractions
    }


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
    "how_to_deal_with_surgery": "exclude", #"keep as is",
    "image_path_folder": [
        "/home/cwatzenboeck/data/AutoPIX_local_data/dev_cw/ds_480/H_patches_size1.5/",
        "/home/cwatzenboeck/data/AutoPIX_local_data/dev_cw/ds_480/F_patches_size1.5/"
    ],
    "pattern": "*/*.npy",

    "score_path_F": "/home/cwatzenboeck/data/AutoPIX_local_data/tabular/autoscoRA_data/autoscoRA_feet.csv",
    "score_path_H": "/home/cwatzenboeck/data/AutoPIX_local_data/tabular/autoscoRA_data/autoscoRA_hands.csv",
    "sum_wrist_points": False,
    "surgery_patientids_list_path_json": None,
    "use_splits_file": True
}




def calculate_split_metadata(df: pd.DataFrame, score_groupings: Dict[str, List[str]] = None) -> Dict:
    """
    Calculate metadata for a given split (Tr, Val, or Ts).
    
    Args:
        df: DataFrame with columns including 'patient_id', 'chosen_score', 'file_name', 'score'
        score_groupings: Optional dictionary mapping group names to lists of score types for aggregated statistics
    
    Returns:
        Dictionary with metadata for the split, including aggregated score distributions if score_groupings provided
    """
    if len(df) == 0:
        result = {
            "num_patients": 0,
            "num_ROIs": 0,
            "num_base_images": 0,
            "num_score_types": 0,
            "avg_num_ROIs_per_score_type": 0.0,
            "visits_per_patient_stats": {
                "median": 0.0,
                "quantile_0.25": 0.0,
                "quantile_0.75": 0.0
            },
            "detailed": {}
        }
        if score_groupings is not None:
            result["aggregated"] = {
                group_name: {"score_distribution": {}, "score_distribution_fractions": {}}
                for group_name in score_groupings.keys()
            }
        return result
    
    # Create base_image_id from patient_id, left_or_right, date_str, and extremity
    # Base image is identified by: <patient_id>_<left_or_right>_<date_str>_<extremity>
    df["base_image_id"] = (
        df["patient_id"].astype(str) + "_" +
        df["left_or_right"].astype(str) + "_" +
        df["date_str"].astype(str) + "_" +
        df["extremity"].astype(str)
    )
    
    # Basic counts
    num_patients = df["patient_id"].nunique()
    num_ROIs = len(df)  # This is actually the number of ROIs/scores, not base images
    num_base_images = df["base_image_id"].nunique()
    
    # Calculate number of visits (base images) per patient
    visits_per_patient = df.groupby("patient_id")["base_image_id"].nunique()
    visits_per_patient_stats = {
        "median": float(visits_per_patient.median()),
        "quantile_0.25": float(visits_per_patient.quantile(0.25)),
        "quantile_0.75": float(visits_per_patient.quantile(0.75))
    }
    
    # Score type information
    score_type_counts = df["chosen_score"].value_counts().to_dict()
    num_score_types = len(score_type_counts)
    avg_num_ROIs_per_score_type = num_ROIs / num_score_types if num_score_types > 0 else 0.0
    
    # Detailed breakdown by score type with per-score statistics
    detailed = {}
    for score_type, count in score_type_counts.items():
        # Filter dataframe for this score type
        df_score_type = df[df["chosen_score"] == score_type]
        
        # Calculate score distribution (support per score value)
        # Drop NaN values if any exist
        scores = df_score_type["score"].dropna()
        score_counts = scores.value_counts().sort_index()
        num_ROIs_for_score_type = int(count)
        num_base_images_for_score_type = df_score_type["base_image_id"].nunique()
        
        # Convert score keys to strings for YAML compatibility
        score_distribution = {str(int(k)): int(v) for k, v in score_counts.items()}
        
        # Calculate score distribution fractions (proportions)
        score_distribution_fractions = {
            str(int(k)): float(v) / num_ROIs_for_score_type 
            for k, v in score_counts.items()
        }
        
        # Get image_series_length statistics if available
        image_series_length_info = {}
        if "image_series_length" in df_score_type.columns:
            series_lengths = df_score_type["image_series_length"].dropna()
            if len(series_lengths) > 0:
                image_series_length_info = {
                    "mean": float(series_lengths.mean()),
                    "median": float(series_lengths.median()),
                    "min": int(series_lengths.min()),
                    "max": int(series_lengths.max())
                }
        
        detailed[score_type] = {
            "num_ROIs": num_ROIs_for_score_type,
            "num_base_images": int(num_base_images_for_score_type),
            "score_distribution": score_distribution,
            "score_distribution_fractions": score_distribution_fractions
        }
        
        if image_series_length_info:
            detailed[score_type]["image_series_length"] = image_series_length_info
    
    # Calculate aggregated score distributions for ERO_H, ERO_F, JSN_H, JSN_F
    aggregated = {}
    if score_groupings is not None:
        for group_name, group_score_types in score_groupings.items():
            aggregated[group_name] = calculate_aggregated_score_distribution(df, group_score_types)
    
    result = {
        "num_patients": int(num_patients),
        "num_ROIs": int(num_ROIs),
        "num_base_images": int(num_base_images),
        "num_score_types": int(num_score_types),
        "avg_num_ROIs_per_score_type": float(avg_num_ROIs_per_score_type),
        "visits_per_patient_stats": visits_per_patient_stats,
        "detailed": detailed
    }
    
    if aggregated:
        result["aggregated"] = aggregated
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Obtain metadata for splits in RA dataset")
    parser.add_argument("--out_dir", type=str, required=False, 
                        default=None,
                       help="Output directory for metadata (YAML file). If None, just print.")
    parser.add_argument("--splits_file", type=str, required=True, 
                       help="path to splits file .csv")
    parser.add_argument("--outname", type=str, required=False,
                        default="splits_metadata.yaml",
                       help="Output filename for metadata (default: splits_metadata.yaml)")
    parser.add_argument("--how_to_deal_with_surgery", type=str, required=False,
                        default=None,
                        choices=["exclude", "keep as is", "keep: map over limit to limit plus one", "keep: map over limit to limit"],
                       help="How to deal with surgery cases. Options: 'exclude', 'keep as is', 'keep: map over limit to limit plus one', 'keep: map over limit to limit'")

    args = parser.parse_args()
    
    print(f"Configuration:")
    pprint(args)
    
    # Update config_data to use the provided splits file
    config_data_local = config_data.copy()
    config_data_local["splits_file"] = args.splits_file
    
    # Update how_to_deal_with_surgery if provided
    if args.how_to_deal_with_surgery is not None:
        config_data_local["how_to_deal_with_surgery"] = args.how_to_deal_with_surgery
        print(f"Updated how_to_deal_with_surgery to: {args.how_to_deal_with_surgery}")
    
    # Step 1: Process data using the splits file
    print("\nStep 1: Processing data with splits file...")
    data_tables = process_several_score_groups(config_data_local)
    
    # Get the "ALL" score group (or first available)
    score_group_name = "ALL"
    if score_group_name not in data_tables:
        score_group_name = list(data_tables.keys())[0]
        print(f"Warning: 'ALL' not found, using '{score_group_name}' instead")
    
    data_group = data_tables[score_group_name]
    df_train = data_group["df_train"]
    df_val = data_group["df_val"]
    df_test = data_group["df_test"]
    
    print(f"Loaded data:")
    print(f"  Train: {len(df_train)} images, {df_train['patient_id'].nunique()} patients")
    print(f"  Val: {len(df_val)} images, {df_val['patient_id'].nunique()} patients")
    print(f"  Test: {len(df_test)} images, {df_test['patient_id'].nunique()} patients")
    
    # Create combined dataframe for cohort statistics
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    print(f"  All (combined): {len(df_all)} images, {df_all['patient_id'].nunique()} patients")
    
    # Load score groupings for aggregated statistics
    print("\nLoading score groupings from resources...")
    score_groupings = load_score_groupings()
    print(f"  Loaded groupings: {list(score_groupings.keys())}")
    for group_name, score_types in score_groupings.items():
        print(f"    {group_name}: {len(score_types)} score types")
    
    # Step 2: Calculate metadata for each split
    print("\nStep 2: Calculating metadata for each split...")
    metadata = {}
    
    metadata["Tr"] = calculate_split_metadata(df_train, score_groupings)
    metadata["Val"] = calculate_split_metadata(df_val, score_groupings)
    metadata["Ts"] = calculate_split_metadata(df_test, score_groupings)
    
    # Step 2b: Calculate combined metadata (Tr+Val+Ts) for cohort statistics
    print("\nStep 2b: Calculating combined metadata (Tr+Val+Ts)...")
    metadata["All"] = calculate_split_metadata(df_all, score_groupings)
    
    # Step 2c: Add settings to metadata
    print("\nStep 2c: Adding settings to metadata...")
    metadata["settings"] = {
        "config_data": config_data_local,
        "splits_file": args.splits_file
    }
    
    # Step 3: Print or save metadata
    print("\n" + "="*60)
    print("METADATA SUMMARY")
    print("="*60)
    
    pprint(metadata)
    
    if args.out_dir is not None:
        output_path = Path(args.out_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / args.outname
        with open(output_file, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
        
        print(f"\nMetadata saved to: {output_file}")
    else:
        print("\n(No output directory specified, metadata not saved)")

if __name__ == "__main__": 
    main()
     