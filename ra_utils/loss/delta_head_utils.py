"""
Utilities for computing class weights for DeltaHeadLoss based on score difference prevalences.

These functions analyze training data to compute prevalences of score differences
and derive class weights for balanced loss computation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Literal


def score_difference_prevalence_by_score_type(data):
    """
    Calculate score difference prevalence (distribution of score differences) per score_type,
    considering differences (i != j) within the same patient_scoretype_key.
    
    This function extracts all pairwise score differences from the training data,
    grouped by score_type, to understand the distribution of score changes.
    
    Args:
        data: Dictionary containing data loaders. Expected structure:
            {
                "ALL": {
                    "train_loader": DataLoader with dataset.data containing:
                        - "patient_scoretype_key": Patient identifier with score type
                        - "score_type": Type of score
                        - "score": Score value
                }
            }
    
    Returns:
        Dict[str, pd.Series]: Dictionary mapping score_type to a pandas Series
                             where index is score difference values and values are counts.
                             Example: {"PIPII": pd.Series({-1: 50, 0: 200, 1: 30, ...}), ...}
    """
    # Extract relevant columns from the training set
    part = ["patient_scoretype_key", "score_type", "score"]
    df = pd.DataFrame(data["ALL"]["train_loader"].dataset.data)[part]

    # Prepare results dict
    score_type_diffs = {}

    for score_type, sdf in df.groupby("score_type"):
        diffs = []
        # For each patient_scoretype_key, compute all pairwise score differences (i != j)
        for key, group in sdf.groupby("patient_scoretype_key"):
            scores = group["score"].values
            for i in range(len(scores)):
                for j in range(len(scores)):
                    if i != j:
                        diff = scores[i] - scores[j]
                        diffs.append(diff)
        # Convert to pandas Series for value_counts (prevalence)
        if diffs:
            diffs_series = pd.Series(diffs)
            prevalence = diffs_series.value_counts().sort_index()
        else:
            prevalence = pd.Series(dtype=int)
        score_type_diffs[score_type] = prevalence

    return score_type_diffs


def prevalences_weights(
    x: Dict[str, "pd.Series"], 
    option: Literal["<_or_Same_or_greater", "regression", "classification"] = "<_or_Same_or_greater",
    smoothing_factor: float = 1.0e-2,
) -> Dict[str, np.ndarray]:
    """
    Calculate per-difference-class weights (as numpy arrays) for each score_type
    based on prevalence in the score-difference distribution, normalized so that the sum
    of weights for each score_type is 1.

    Args:
        x: Dict where each value is a pd.Series mapping score differences to counts for a score_type.
           Typically the output of score_difference_prevalence_by_score_type().
        option: 
            "<_or_Same_or_greater": outputs numpy array [w[<0], w[==0], w[>0]] for 3-class comparison
            "regression": outputs weights in the order of sorted unique diff values (for regression tasks)
            "classification": same as regression (for multi-class classification)
        smoothing_factor: Add to denominator for numerical stability. Prevents division by zero
                         for rare differences. Default: 0.1.

    Returns:
        Dict[str, np.ndarray]: Dictionary mapping score_type to normalized weight array.
                              Weights are normalized so sum equals 1.0 for each score_type.
                              
    Example:
        >>> prevalences = {"PIPII": pd.Series({-1: 10, 0: 100, 1: 10})}
        >>> weights = prevalences_weights(prevalences, option="<_or_Same_or_greater")
        >>> # Returns: {"PIPII": np.array([w_neg, w_zero, w_pos])} where weights balance the classes
    """
    weights = {}
    for score_type, prevalence in x.items():
        total = prevalence.sum()
        if total == 0:
            weights[score_type] = np.array([])
            continue
        
        if option == "<_or_Same_or_greater":
            # Three-class comparison: <, ==, >
            num_lt_0 = prevalence[prevalence.index < 0].sum() if any(prevalence.index < 0) else 0
            num_eq_0 = prevalence[prevalence.index == 0].sum() if any(prevalence.index == 0) else 0
            num_gt_0 = prevalence[prevalence.index > 0].sum() if any(prevalence.index > 0) else 0

            arr = np.array([
                1.0 / ((num_lt_0 / total) + smoothing_factor),
                1.0 / ((num_eq_0 / total) + smoothing_factor),
                1.0 / ((num_gt_0 / total) + smoothing_factor)
            ])
            arr /= arr.sum()
            weights[score_type] = arr
        
        elif option in ("regression", "classification"):
            # Always sort diffs for reproducible array order
            diffs = np.array(sorted(prevalence.index))
            counts = np.array([prevalence.get(d, 0) for d in diffs])
            fracs = counts / total
            arr = 1.0 / (fracs + smoothing_factor)
            arr /= arr.sum()
            weights[score_type] = arr

        else:
            raise ValueError(f"Unknown option: {option}")

    return weights


def score_differences_to_class_indices(
    score_differences: np.ndarray,
    num_classes: int = 3,
) -> np.ndarray:
    """
    Convert continuous score differences to class indices for classification.
    
    For 3-class classification (typical case):
        - diff < 0  -> class 0 (score_i < score_j)
        - diff == 0 -> class 1 (score_i == score_j)
        - diff > 0  -> class 2 (score_i > score_j)
    
    For other numbers of classes, uses binning strategy (can be extended).
    
    Args:
        score_differences: Array of score differences (score_i - score_j)
        num_classes: Number of classes (default: 3)
    
    Returns:
        Array of class indices in range [0, num_classes-1]
    """
    if num_classes == 3:
        # 3-class comparison: <, ==, >
        class_indices = np.where(
            score_differences < 0, 0,
            np.where(score_differences == 0, 1, 2)
        )
    elif num_classes == 1:
        # Regression case - return as-is (though this shouldn't be called for regression)
        raise ValueError("num_classes=1 indicates regression, not classification")
    else:
        # For other numbers of classes, use binning
        # This is a simple implementation - can be extended for more sophisticated binning
        min_diff = score_differences.min()
        max_diff = score_differences.max()
        bins = np.linspace(min_diff, max_diff, num_classes + 1)
        class_indices = np.digitize(score_differences, bins) - 1
        # Ensure valid range [0, num_classes-1]
        class_indices = np.clip(class_indices, 0, num_classes - 1)
    
    return class_indices

