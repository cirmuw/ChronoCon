
import torch
import mlflow
import torch
import torch.nn as nn

from ra_utils.data.dataloader_CR_patches import (
    process_several_score_groups,
    dataset_and_loader_several,
    check_duplicates_in_dataloader
)

from ra_utils.training.scores_SHS.scores_SHS_training_lib_AE_v1 import (
    evaluate_and_log_testset_results_AE_v3,
    train_loop_AE_v3
)
from ra_utils.training.scores_SHS.scores_SHS_training_lib_AE_v4 import (
    evaluate_and_log_testset_results_AE_v4,
    train_loop_AE_v4
)

from ra_utils.networks.loss_function import get_score_loss_function, get_triplet_loss_fn
import torchvision.transforms.v2 as v2
from ra_utils.training.scores_SHS.model_builders import build_models_AE_v1_and2
import ra_utils.utils.utils_torch
from ra_utils.utils.verbosity_enums import *
import ra_utils.utils.utils

import ra_utils.utils.utils_torch
from pprint import pprint
import ra_utils.utils.config_parser

from ra_utils.utils.utils import datestr_to_years_since_2000


import numpy as np


from ra_utils.training.scores_SHS.run_training_main_lib import (
    check_config_consistency_and_partially_make_consistent,
    maybe_partially_init_model_from_state_dict,
)

import ra_utils.training.scores_SHS.run_training_main_lib

import ra_utils.loss.loss_fn_dict


import ra_utils.loss.loss_RnC_with_ranks_and_ids
import ra_utils.loss.loss_RnC

import ra_utils.loss.loss_RnCMono

import ra_utils.loss.online_mining_triplet_loss_MDP


import ra_utils.visualization.interactive.ra_model

from pathlib import Path
from datetime import datetime

import ra_utils.visualization
import  ra_utils.visualization.trajectories

import pandas as pd
from collections import Counter

import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.cluster.hierarchy import linkage, leaves_list
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np
from pathlib import Path 
import yaml

import ra_utils.visualization.feature_similarity_vs_delta_SvH
from ra_utils.visualization.feature_similarity_vs_delta_SvH import ( 
    plot_all_rois_similarity_boxplots,
    plot_single_roi,
    plot_similarity_matrix,
    compute_similarity_data, 
    plot_single_roi_several_models, 
    plot_all_rois_similarity_boxplots_several_models
)

from mlflow.tracking import MlflowClient

import mlflow
from typing import List


def get_experiment_id_from_name(experiment_name: str, artifact_path = "/msc/home/cwatze93/data/mlflow/mlflow_RAv2/", client = None) -> str:
    """
    Get experiment ID from experiment name.
    
    Parameters:
    - experiment_name (str): Name of the MLflow experiment
    - artifact_path (str): Path to MLflow tracking URI
    - client (MlflowClient, optional): Pre-initialized MLflow client. If None, creates a new one.
    
    Returns:
    - str: Experiment ID
    """
    # Set the MLflow tracking URI if client is not provided
    if client is None:
        mlflow.set_tracking_uri(artifact_path)
    
    # Get the experiment by name
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found in {artifact_path}")
    
    return experiment.experiment_id


def get_run_id_from_name(run_name: str, artifact_path = "/msc/home/cwatze93/data/mlflow/mlflow_RAv2/", experiment_name: str = None, client = None, experiment_id = None) -> str:
    """
    Get run ID from run name.
    
    Parameters:
    - run_name (str): Name of the MLflow run
    - artifact_path (str): Path to MLflow tracking URI
    - experiment_name (str, optional): Name of the experiment to search in. If None, searches all experiments.
    - client (MlflowClient, optional): Pre-initialized MLflow client. If None, creates a new one.
    - experiment_id (str, optional): Pre-fetched experiment ID. If provided, skips fetching it again.
    
    Returns:
    - str: Run ID
    """
    # Initialize client if not provided
    if client is None:
        mlflow.set_tracking_uri(artifact_path)
        client = MlflowClient()
    
    # Get experiment IDs to search in
    if experiment_id is not None:
        experiment_ids = [experiment_id]
    elif experiment_name is not None:
        experiment_id = get_experiment_id_from_name(experiment_name, artifact_path, client)
        experiment_ids = [experiment_id]
    else:
        # Search across all experiments
        experiments = client.list_experiments()
        experiment_ids = [exp.experiment_id for exp in experiments]
    
    # Search for runs with matching name
    runs = client.search_runs(experiment_ids=experiment_ids, filter_string=f"tags.mlflow.runName = '{run_name}'")
    
    if len(runs) == 0:
        raise ValueError(f"Run '{run_name}' not found in {artifact_path}" + 
                        (f" (experiment: {experiment_name})" if experiment_name else ""))
    
    if len(runs) > 1:
        raise ValueError(f"Multiple runs found with name '{run_name}' in {artifact_path}. "
                        f"Found {len(runs)} runs. Please specify experiment_name to narrow the search.")
    
    return runs[0].info.run_id












def load_run_paths_as_df(runs: List[str], experiment_name: str, artifact_path: str) -> pd.DataFrame: 
    """
    Load run paths and metadata into a DataFrame.
    
    Parameters:
    - runs (List[str]): List of run names to load
    - experiment_name (str): Name of the MLflow experiment
    - artifact_path (str): Path to MLflow tracking URI
    
    Returns:
    - pd.DataFrame: DataFrame with columns: experiment_name, experiment_id, run_name, run_id, run_artifacts_paths
    """
    # Initialize MLflow client once
    mlflow.set_tracking_uri(artifact_path)
    client = MlflowClient()
    
    # Get experiment ID once
    experiment_id = get_experiment_id_from_name(experiment_name, artifact_path, client)
    
    # Initialize lists to store data
    data = {
        'experiment_name': [],
        'experiment_id': [],
        'run_name': [],
        'run_id': [],
        'run_artifacts_paths': []
    }
    
    # Process each run
    for run_name in runs:
        try:
            # Get run ID from run name, passing the client and experiment_id to avoid re-initialization
            run_id = get_run_id_from_name(run_name, artifact_path, experiment_name, client, experiment_id)
            
            # Build run artifacts path: artifact_path/experiment_id/run_id
            run_artifacts_path = f"{artifact_path.rstrip('/')}/{experiment_id}/{run_id}"
            
            # Append to data
            data['experiment_name'].append(experiment_name)
            data['experiment_id'].append(experiment_id)
            data['run_name'].append(run_name)
            data['run_id'].append(run_id)
            data['run_artifacts_paths'].append(run_artifacts_path)
        except ValueError as e:
            print(f"Warning: Could not load run '{run_name}': {e}")
            # Optionally, you could still add the row with None values
            # or skip it entirely (current behavior)
            continue
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df


def load_mlflow_metrics_from_filesystem(df: pd.DataFrame, metric_names: List[str], run_artifacts_paths_column: str = "run_artifacts_paths") -> pd.DataFrame:
    """
    Load MLflow metrics from filesystem and add them to the DataFrame.
    
    MLflow stores metrics in files at <run_artifacts_path>/metrics/<metric_name>.
    Each line in the file has format: <timestamp> <value> <step>
    This function extracts the value from the last line of each metric file.
    
    Parameters:
    - df (pd.DataFrame): DataFrame with a column containing run artifacts paths
    - metric_names (List[str]): List of metric names to load (e.g., ["valFinal_macro_f1", "test_macro_f1"])
    - run_artifacts_paths_column (str): Name of the column containing run artifacts paths. Default: "run_artifacts_paths"
    
    Returns:
    - pd.DataFrame: DataFrame with additional columns for each metric
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Initialize columns for each metric with None
    for metric_name in metric_names:
        df[metric_name] = None
    
    # Process each row
    for i, row in df.iterrows():
        run_artifacts_path = row[run_artifacts_paths_column]
        
        # Load each metric
        for metric_name in metric_names:
            metric_file_path = Path(run_artifacts_path) / "metrics" / metric_name
            
            try:
                if metric_file_path.exists():
                    # Read the metric file
                    with open(metric_file_path, 'r') as f:
                        lines = f.readlines()
                    
                    if lines:
                        # Get the last line and extract the value (second column)
                        last_line = lines[-1].strip()
                        if last_line:
                            parts = last_line.split()
                            if len(parts) >= 2:
                                # Second column is the value
                                metric_value = float(parts[1])
                                df.at[i, metric_name] = metric_value
                            else:
                                print(f"Warning: Could not parse metric '{metric_name}' from {metric_file_path}: unexpected format")
                        else:
                            print(f"Warning: Empty metric file '{metric_name}' at {metric_file_path}")
                    else:
                        print(f"Warning: Empty metric file '{metric_name}' at {metric_file_path}")
                else:
                    print(f"Warning: Metric file '{metric_name}' not found at {metric_file_path}")
            except Exception as e:
                print(f"Warning: Error loading metric '{metric_name}' from {metric_file_path}: {e}")
                continue
    
    return df


def read_yaml_metrics(base_path, middle, file):
    base_path = Path(base_path)

    file_stump = Path(file).with_suffix('').as_posix()
    src = base_path / middle / file 
    with open(src, 'r') as f:
        metrics = yaml.safe_load(f)
    metrics

    metrics_mod = {f"{file_stump}: {k}":v for k,v in metrics.items()}
    return metrics_mod


def add_metrics_to_df(df, file: str, middle: str = ""): 
    for i, row in df.iterrows():
        base_path = row["run_artifacts_paths"]
        metrics_mod = read_yaml_metrics(base_path, middle, file)
        for k,v in metrics_mod.items():
            df.loc[i, k] = v
    return pd.DataFrame(df)




def bar_plot_performances(
    df_dct, 
    x_column="num_patients",
    y_column="valFinal/metrics_SvH_H+F: mae",
    y_column_upper="valFinal/metrics_SvH_H+F: mae_CI95_upper",
    y_column_lower="valFinal/metrics_SvH_H+F: mae_CI95_lower",
    axis=None,
    title=None,
    xlabel="Number of Patients (log scale)",
    ylabel="MAE (SvH H+F)",
    extra_info_column_x_axis=None,
    bar_width=None,
    legend=True, 
    colors=None, 
    no_x_ticks_base_label=True
):
    """
    Create a barplot comparing multiple models with error bars on log scale x-axis.

    Parameters:
    - df_dct: Dictionary with model names as keys and DataFrames as values
    - x_column: Column name for x-axis values
    - y_column: Column name for y-axis values (mean)
    - y_column_upper: Column name for upper error bound
    - y_column_lower: Column name for lower error bound
    - axis: Optional matplotlib axis to plot on (if None, creates new figure)
    - title: Plot title
    - xlabel: X-axis label
    - ylabel: Y-axis label
    - extra_info_column_x_axis: Optional column name containing strings to display below x-axis ticks
    - bar_width: Width of the bars in the barplot
    - legend: Whether to show a legend (default True)

    Returns:
    - fig, ax: matplotlib figure and axis objects
    """


    # # Get number of models for spacing calculation
    num_models = len(df_dct)
    # # Colors for different models
    # colors = plt.cm.tab10(np.linspace(0, 1, num_models))


    if colors is None:
        # Use matplotlib's default color cycle
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = [prop_cycle.by_key()['color'][i % len(prop_cycle.by_key()['color'])] 
                 for i in range(len(df_dct))]

    # Create figure/axis if not provided
    if axis is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        ax = axis
        fig = ax.figure


    # Get all unique x values from all dataframes to determine spacing
    all_x_values = set()
    for df_part_to_plot in df_dct.values():
        all_x_values.update(df_part_to_plot[x_column].values)
    all_x_values = np.sort(np.array(list(all_x_values)))

    # Convert to log2 space for positioning
    x_values_log2 = np.log2(all_x_values)

    # Calculate bar width based on minimum spacing in log space
    if bar_width is None:
        if len(x_values_log2) > 1:
            min_spacing = np.min(np.diff(x_values_log2))
            bar_width = min_spacing * 0.7 / num_models  # Use 70% of spacing, divided by number of models
            print(f"Bar width: {bar_width}")
        else:
            bar_width = 0.3
            print(f"Bar width: {bar_width}")


    # Plot each model
    for i, (name, df_part_to_plot) in enumerate(df_dct.items()):
        x = df_part_to_plot[x_column]
        y = df_part_to_plot[y_column]

        # Handle error bars - if y_column_upper or y_column_lower is None, skip error bars
        if y_column_upper is not None and y_column_lower is not None:
            y_lower = df_part_to_plot[y_column_lower]
            y_upper = df_part_to_plot[y_column_upper]
            # Calculate error bar heights (distance from mean to bounds)
            yerr_lower = y - y_lower
            yerr_upper = y_upper - y
            yerr = [yerr_lower.values, yerr_upper.values]
        else:
            yerr = None

        # Convert x values to log2 space
        x_values_log2_model = np.log2(x.values)

        # Calculate shift to center bars around each x position
        # For 2 models: shift = -bar_width/2, +bar_width/2
        # For 3 models: shift = -bar_width, 0, +bar_width
        shift_offset = (i - (num_models - 1) / 2) * bar_width

        # Map x values to their positions in all_x_values
        x_positions = []
        for x_val in x.values:
            idx = np.where(all_x_values == x_val)[0]
            if len(idx) > 0:
                x_positions.append(x_values_log2[idx[0]] + shift_offset)
            else:
                # If x value not in all_x_values, use direct log2 conversion
                x_positions.append(np.log2(x_val) + shift_offset)

        x_positions = np.array(x_positions)

        # Plot bars with label for legend
        if yerr is not None:
            bars = ax.bar(
                x_positions, y, yerr=yerr, capsize=5, alpha=0.7,
                edgecolor='black', width=bar_width, label=name, color=colors[i]
            )
        else:
            bars = ax.bar(
                x_positions, y, alpha=0.7,
                edgecolor='black', width=bar_width, label=name, color=colors[i]
            )

    # Set x-axis labels
    ax.set_xticks(x_values_log2)

    # Handle extra info column if provided
    if extra_info_column_x_axis is not None:
        # Create a mapping from x values to extra info strings
        x_to_extra_info = {}
        for df_part_to_plot in df_dct.values():
            if extra_info_column_x_axis in df_part_to_plot.columns:
                for x_val, extra_info in zip(df_part_to_plot[x_column], df_part_to_plot[extra_info_column_x_axis]):
                    if x_val not in x_to_extra_info:
                        x_to_extra_info[x_val] = str(extra_info)

        # Create tick labels with extra info
        tick_labels = []
        for x_val in all_x_values:
            base_label = str(x_val)
            if x_val in x_to_extra_info:
                extra_info = x_to_extra_info[x_val]
                # Combine with line break
                if no_x_ticks_base_label:
                    tick_labels.append(extra_info)
                else:
                    tick_labels.append(f"{base_label} {extra_info}")
                #tick_labels.append(f"{base_label} {extra_info}")
            else:
                tick_labels.append(base_label)

        ax.set_xticklabels(tick_labels)
    else:
        ax.set_xticklabels(all_x_values)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)

    # Set y-ticks on left only, not right; set ticks facing inside.
    ax.tick_params(axis='y', left=True, right=False, labelleft=True, labelright=False, direction='in')
    # Set x-ticks facing inside
    ax.tick_params(axis='x', direction='in')

    # Add legend if legend=True
    if legend:
        ax.legend(loc='best', fontsize=10)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    return fig, ax
