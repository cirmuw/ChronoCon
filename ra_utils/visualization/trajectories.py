# Separate computation and plotting functions
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
from typing import Tuple, Optional

try:
    from joblib import dump as joblib_dump, load as joblib_load
except Exception:
    joblib_dump = None
    joblib_load = None

def compute_2d_projection(pack, reduction_method="tsne", seed=43):
    """
    Compute 2D projection of embeddings.
    
    Args:
        pack: The data pack containing embeddings and metadata
        reduction_method: Dimensionality reduction method ("pca", "umap", or "tsne")
        seed: Random seed for reproducibility
        
    Returns:
        coords_2d: 2D coordinates (N, 2)
        reducer: The fitted reducer object
    """
    print(f"Computing {reduction_method.upper()} projection with seed {seed}...")
    print(f"Input embeddings shape: {pack['Z'].shape}")
    
    if reduction_method.lower() == "pca":
        reducer = PCA(n_components=2, random_state=seed)
        coords_2d = reducer.fit_transform(pack['Z'])
        
    elif reduction_method.lower() == "umap":
        reducer = umap.UMAP(n_components=2, random_state=seed)
        coords_2d = reducer.fit_transform(pack['Z'])
        
    elif reduction_method.lower() == "tsne":
        reducer = TSNE(n_components=2, random_state=seed, perplexity=min(30, max(5, len(pack['Z']) // 10)))
        coords_2d = reducer.fit_transform(pack['Z'])
        
    else:
        raise ValueError(f"Unknown reduction method: {reduction_method}. Choose from 'pca', 'umap', or 'tsne'")
    
    print(f"2D projection completed. Shape: {coords_2d.shape}")
    return coords_2d, reducer

from copy import deepcopy

def plot_trajectories_from_coords_sorted(
    coords_2d,
    pack,
    trajectories_to_connect,
    reduction_method="tsne",
    figsize=(5, 4),
    no_legend=True,
    background_color_key=None,
    background_color_key_title=None,
    background_colormap="viridis",
    traj_colors_columns="t_rel",
    traj_colors_colorbar=False,
    axis=None,
    traj_line_color=None,
    markersize_fn=None,
    markersize_col="score_gt",
    alpha_fn=None,
    alpha_col="score_gt",
    plot_order_column=None,   # <-- NEW
    flip_x=False,                   # <-- NEW
    hide_frame=True                 # <-- NEW
):
    """
    Plot trajectories from pre-computed 2D coordinates.
    """
    pack = deepcopy(pack)
    coords_2d=deepcopy(coords_2d)

    print(f"Plotting {len(trajectories_to_connect)} trajectories...")

    # ------------------- OPTIONAL X-FLIP -------------------
    if flip_x:
        coords_2d = coords_2d.copy()
        coords_2d[:, 0] *= -1
        print("Flipping X-axis (coords[:,0] *= -1).")

    # Marker cycle
    marker_cycle = ['o', '^', 'D', 'v', 'P', 'X', '*', 'h', 'p', 's']

    # Set up figure
    if axis is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = axis
        fig = ax.figure

    # ----------------------- GROUPS ------------------------
    # These preserve chronological order → DO NOT TOUCH THIS
    from collections import defaultdict
    groups = defaultdict(list)
    for i, key in enumerate(pack["patient_scoretype_key"]):
        groups[key].append(i)

    for key in groups:
        groups[key].sort(key=lambda i: pack["years_since_2000"][i])  # keep temporal trajectories

    # ------------------ SORT BACKGROUND PLOTTING ORDER ------------------
    # Only sort for background scatter → Does NOT affect trajectories
    if plot_order_column in pack:
        try:
            background_order = np.argsort(pack[plot_order_column])
            print(
                f"Sorting background points by '{plot_order_column}' "
                f"(low → high, high will be drawn last and appear on top)."
            )
        except Exception:
            background_order = np.arange(len(coords_2d))
            print(f"Warning: Could not sort by {plot_order_column}. Plotting in original order.")
    else:
        background_order = np.arange(len(coords_2d))
        print(f"Warning: plot_order_column '{plot_order_column}' not found. No sorting applied.")

    # ------------------ Marker sizes for background -------------------
    marker_sizes = None
    if markersize_fn is not None and markersize_col in pack:
        x = pack[markersize_col]
        marker_sizes = np.array([markersize_fn(val) for val in x])
    elif markersize_fn is not None:
        print(f"Warning: markersize_col '{markersize_col}' not found. Skipping.")

    # ------------------ Alpha for background -------------------
    alpha_values = None
    if alpha_fn is not None and alpha_col in pack:
        x = pack[alpha_col]
        alpha_values = np.array([alpha_fn(val) for val in x])
    elif alpha_fn is not None:
        print(f"Warning: alpha_col '{alpha_col}' not found. Skipping.")

    # ------------------ Compute trajectory colors -------------------
    if len(trajectories_to_connect) > 0:
        colors = plt.cm.Set1(np.linspace(0, 1, len(trajectories_to_connect)))
        patient_colors = {patient: colors[i] for i, patient in enumerate(trajectories_to_connect)}

    # ------------------ BACKGROUND SCATTER -------------------
    # Draw in sorted order so high-score points appear ON TOP
    print("Plotting background points...")

    bg_x = coords_2d[background_order, 0]
    bg_y = coords_2d[background_order, 1]

    if background_color_key is None:
        ax.scatter(
            bg_x,
            bg_y,
            c='lightgray',
            s=marker_sizes[background_order] if marker_sizes is not None else 20,
            alpha=alpha_values[background_order] if alpha_values is not None else 0.3,
        )
    else:
        if background_color_key not in pack:
            print(f"Warning: '{background_color_key}' not found.")
            ax.scatter(
                bg_x,
                bg_y,
                c='lightgray',
                s=marker_sizes[background_order] if marker_sizes is not None else 20,
                alpha=alpha_values[background_order] if alpha_values is not None else 0.3,
            )
        else:
            values = np.array(pack[background_color_key])
            try:
                values = values.astype(float)
                scatter = ax.scatter(
                    bg_x,
                    bg_y,
                    c=values[background_order],
                    cmap=background_colormap,
                    s=marker_sizes[background_order] if marker_sizes is not None else 20,
                    alpha=alpha_values[background_order] if alpha_values is not None else 0.3,
                )
                if axis is None:
                    plt.colorbar(scatter, ax=ax, label=background_color_key_title or background_color_key)
            except Exception:
                # categorical
                unique = np.unique(values)
                mapping = {cat: i for i, cat in enumerate(unique)}
                mapped = np.array([mapping[v] for v in values])
                scatter = ax.scatter(
                    bg_x,
                    bg_y,
                    c=mapped[background_order],
                    cmap='tab10',
                    s=marker_sizes[background_order] if marker_sizes is not None else 20,
                    alpha=alpha_values[background_order] if alpha_values is not None else 0.3,
                )
                if axis is None:
                    cbar = plt.colorbar(scatter, ax=ax, ticks=np.arange(len(unique)))
                    cbar.ax.set_yticklabels(unique)

    # ------------------ TRAJECTORIES -------------------
    last_scatter = None

    for idx, patient_key in enumerate(trajectories_to_connect):

        if patient_key not in groups or len(groups[patient_key]) < 2:
            print(f"Skipping trajectory '{patient_key}' (less than 2 points).")
            continue

        indices = groups[patient_key]
        patient_coords = coords_2d[indices]

        # line color (same for all if traj_line_color set)
        if traj_line_color is not None:
            line_color = traj_line_color
        else:
            line_color = patient_colors[patient_key]

        # plot line
        ax.plot(
            patient_coords[:, 0],
            patient_coords[:, 1],
            color=line_color,
            linewidth=3,
            alpha=0.8,
        )

        # marker cycle
        marker = marker_cycle[idx % len(marker_cycle)]

        # marker color
        if traj_colors_columns in pack:
            cvals = pack[traj_colors_columns][indices]
        else:
            cvals = np.linspace(0, 1, len(indices))

        last_scatter = ax.scatter(
            patient_coords[:, 0],
            patient_coords[:, 1],
            c=cvals,
            cmap='viridis',
            s=30,
            marker=marker,
            edgecolors=line_color,
            linewidth=1.5,
            alpha=0.95,
            zorder=5,
        )

    # colorbar for trajectory points
    if traj_colors_colorbar and last_scatter is not None and axis is None:
        plt.colorbar(last_scatter, ax=ax, label=f'Trajectory: {traj_colors_columns}')

    ax.set_xlabel(f'{reduction_method.upper()} Dimension 1')
    ax.set_ylabel(f'{reduction_method.upper()} Dimension 2')

    # Legend
    if not no_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.grid(False)
    ax.tick_params(axis='both', direction='in')

    # ------------------ HIDE FRAME -------------------
    if hide_frame:
        for s in ax.spines.values():
            s.set_visible(False)
        # uncomment if you also want no ticks:
        # ax.set_xticks([])
        # ax.set_yticks([])

    if axis is None:
        plt.tight_layout()

    return fig, ax




def plot_trajectories_from_coords(
    coords_2d,
    pack,
    trajectories_to_connect,
    reduction_method="tsne",
    figsize=(5, 4),
    no_legend=True,
    background_color_key=None,
    background_color_key_title=None,
    background_colormap="viridis",
    traj_colors_columns="t_rel",
    traj_colors_colorbar=False,
    axis=None,
    traj_line_color=None,  # <-- NEW KEYWORD
    markersize_fn=None,  # Optional function to compute marker sizes: markersize_fn(x, size=20)
    markersize_col="score_gt",  # Column name to use for marker size computation
    alpha_fn=None,  # Optional function to compute alpha values: alpha_fn(x, base=0.3)
    alpha_col="score_gt",  # Column name to use for alpha computation
    flip_x=False
):
    """
    Plot trajectories from pre-computed 2D coordinates.

    Args:
        coords_2d: Pre-computed 2D coordinates (N, 2)
        pack: The data pack containing metadata
        trajectories_to_connect: List of patient_scoretype_key to connect with lines
        reduction_method: Method name for axis labels
        figsize: Figure size
        no_legend: If True, hide the legend (default: True)
        background_color_key: Key in pack to use for background coloring (default: None = gray)
        background_color_key_title: Title for background colorbar (default: None)
        background_colormap: Colormap for numeric background coloring (default: "viridis")
        traj_colors_columns: Key in pack to use for trajectory point coloring (default: "t_rel")
        traj_colors_colorbar: If True, show colorbar for trajectory point colors (default: False)
        axis: Optional matplotlib axes. If provided, plot on this axis and do not create a new figure.
        traj_line_color: Color to use for connecting lines between markers for all trajectories (default: None = color per trajectory).
                         If set, all lines will be plotted with the same color.
        markersize_fn: Optional function to compute marker sizes for background points only. Function signature: markersize_fn(x, size=20)
                       where x is a single value from the pack column and size is the base size (default: 20).
                       Should return a single marker size value. If None, uses default size of 20 for background points.
                       Note: Trajectory points always use a fixed size of 30.
        markersize_col: Column name in pack to use for marker size computation (default: "score_gt")
        alpha_fn: Optional function to compute alpha (transparency) values for background points only. Function signature: alpha_fn(x, base=0.3)
                  where x is a single value from the pack column and base is the base alpha value.
                  Should return a single alpha value. If None, uses default alpha of 0.3 for background points.
                  Note: Trajectory points always use a fixed alpha of 0.95.
        alpha_col: Column name in pack to use for alpha computation (default: "score_gt")

    Returns:
        fig, ax: Matplotlib figure and axis objects (so you can save the plot later)
    """
    print(f"Plotting {len(trajectories_to_connect)} trajectories...")
    pack = deepcopy(pack)
    coords_2d = deepcopy(coords_2d)
    if flip_x: 
        factor_x= -1
    else: 
        factor_x=1
    # Define a marker cycle for up to 10 unique marker styles, then cycle through if more needed
    marker_cycle = ['o',  '^', 'D', 'v', 'P', 'X', '*', 'h', 'p', 's']

    # Handle figure and axis creation
    if axis is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = axis
        fig = ax.figure

    # Group data by patient_scoretype_key
    groups = defaultdict(list)
    for i, key in enumerate(pack['patient_scoretype_key']):
        groups[key].append(i)

    # Sort each group by years_since_2000
    for key in groups:
        groups[key].sort(key=lambda i: pack['years_since_2000'][i])

    # Compute marker sizes for background points if markersize_fn is provided
    marker_sizes = None
    if markersize_fn is not None:
        # Get the column data
        if markersize_col in pack:
            x = pack[markersize_col]
            marker_sizes = np.array([markersize_fn(val) for val in x])
            print(f"Using markersize_fn with column '{markersize_col}' for background points, size range: {marker_sizes.min():.2f} to {marker_sizes.max():.2f}")
        else:
            print(f"Warning: Column '{markersize_col}' not found in pack for markersize_fn. Using default sizes.")
            marker_sizes = None

    # Compute alpha values for background points if alpha_fn is provided
    alpha_values = None
    if alpha_fn is not None:
        # Get the column data
        if alpha_col in pack:
            x = pack[alpha_col]
            alpha_values = np.array([alpha_fn(val) for val in x])
            print(f"Using alpha_fn with column '{alpha_col}' for background points, alpha range: {alpha_values.min():.3f} to {alpha_values.max():.3f}")
        else:
            print(f"Warning: Column '{alpha_col}' not found in pack for alpha_fn. Using default alpha.")
            alpha_values = None

    # Define colors for the trajectories to connect (for markers and edges)
    if len(trajectories_to_connect) > 0:
        colors = plt.cm.Set1(np.linspace(0, 1, len(trajectories_to_connect)))
        patient_colors = {patient: colors[i] for i, patient in enumerate(trajectories_to_connect)}

    # Plot ALL points (background) with optional coloring
    if background_color_key is not None:
        if background_color_key in pack:
            background_values = pack[background_color_key]
            print(f"Using '{background_color_key}' for background coloring with colormap '{background_colormap}'")

            # Check if values are numeric
            try:
                background_values = np.asarray(background_values, dtype=float)
                print(f"Background values range: {background_values.min():.3f} to {background_values.max():.3f}")
                
                # Use colormap for numeric values
                scatter = ax.scatter(
                    coords_2d[:, 0] * factor_x,
                    coords_2d[:, 1],
                    c=background_values,
                    cmap=background_colormap,
                    s=marker_sizes if marker_sizes is not None else 20,
                    alpha=alpha_values if alpha_values is not None else 0.3,
                    label='All other samples'
                )

                # Add colorbar for background if axis was not provided
                if axis is None:
                    if background_color_key_title is not None:
                        cbar = plt.colorbar(scatter, ax=ax, label=f'{background_color_key_title}')
                    else:
                        cbar = plt.colorbar(scatter, ax=ax, label=f'{background_color_key}')

            except (ValueError, TypeError):
                # Non-numeric values, treat as categorical
                print(f"Non-numeric values detected, using categorical coloring")

                # Convert categorical values to numeric indices
                unique_categories = np.unique(background_values)
                category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
                numeric_categories = np.array([category_to_idx[val] for val in background_values])

                print(f"Found {len(unique_categories)} unique categories: {unique_categories}")

                # Use a discrete colormap
                n_categories = len(unique_categories)
                if n_categories <= 10:
                    cmap = plt.cm.get_cmap('tab10', n_categories)
                elif n_categories <= 20:
                    cmap = plt.cm.get_cmap('tab20', n_categories)
                else:
                    cmap = plt.cm.get_cmap(background_colormap, n_categories)

                scatter = ax.scatter(
                    coords_2d[:, 0] * factor_x,
                    coords_2d[:, 1],
                    c=numeric_categories,
                    cmap=cmap,
                    s=marker_sizes if marker_sizes is not None else 20,
                    alpha=alpha_values if alpha_values is not None else 0.3,
                    vmin=-0.5,
                    vmax=n_categories - 0.5
                )

                # Add colorbar with category labels if axis is None (no custom axis)
                if axis is None:
                    cbar = plt.colorbar(
                        scatter,
                        ax=ax,
                        label=f'Background: {background_color_key}',
                        ticks=np.arange(n_categories)
                    )
                    cbar.ax.set_yticklabels(unique_categories)

        else:
            print(f"Warning: '{background_color_key}' not found in pack!")
            print(f"Available keys in pack: {list(pack.keys())}")
            print("Using default gray background")
            ax.scatter(
                coords_2d[:, 0] * factor_x,
                coords_2d[:, 1],
                c='lightgray',
                s=marker_sizes if marker_sizes is not None else 20,
                alpha=alpha_values if alpha_values is not None else 0.3,
                label='All other samples'
            )
    else:
        ax.scatter(
            coords_2d[:, 0] * factor_x,
            coords_2d[:, 1],
            c='lightgray',
            s=marker_sizes if marker_sizes is not None else 20,
            alpha=alpha_values if alpha_values is not None else 0.3,
            label='All other samples'
        )

    last_scatter = None  # For colorbar

    # Plot trajectories for specified patients
    for idx, patient_key in enumerate(trajectories_to_connect):
        if patient_key not in groups or len(groups[patient_key]) < 2:
            print(f"Warning: {patient_key} has less than 2 points, skipping trajectory")
            continue

        indices = groups[patient_key]
        patient_coords = coords_2d[indices]

        # Plot trajectory line -- color determined by new keyword
        if traj_line_color is not None:
            line_color = traj_line_color
        else:
            line_color = patient_colors[patient_key] if len(trajectories_to_connect) > 0 else None

        ax.plot(
            patient_coords[:, 0] * factor_x,
            patient_coords[:, 1],
            color=line_color,
            linewidth=3,
            alpha=0.8,
            label=f'{patient_key} ({len(indices)} points)'
        )

        # Determine marker style (cycle if more patients than markers)
        marker = marker_cycle[idx % len(marker_cycle)]

        # Plot points for this patient (smaller and with distinct style)
        # Note: Trajectory points use fixed size (30), marker sizes only apply to background
        if traj_colors_columns is None:
            patient_t_rel = "gray"
            last_scatter = ax.scatter(
                patient_coords[:, 0] * factor_x,
                patient_coords[:, 1],
                c=patient_t_rel,
                cmap=None,
                s=30,  # Fixed size for trajectory points
                marker=marker,
                edgecolors=patient_colors[patient_key] if len(trajectories_to_connect) > 0 else None,
                linewidth=1.5,
                alpha=0.95,
                zorder=5
            )
        else:
            patient_t_rel = (
                pack[traj_colors_columns][indices]
                if traj_colors_columns in pack
                else np.linspace(0, 1, len(indices))
            )
            last_scatter = ax.scatter(
                patient_coords[:, 0] * factor_x,
                patient_coords[:, 1],
                c=patient_t_rel,
                cmap='viridis',
                s=30,  # Fixed size for trajectory points
                marker=marker, # Distinct style for each trajectory
                edgecolors=patient_colors[patient_key] if len(trajectories_to_connect) > 0 else None,
                linewidth=1.5,
                alpha=0.95,
                zorder=5
            )

    # Add colorbar for trajectory colors if requested
    if traj_colors_colorbar and len(trajectories_to_connect) > 0 and last_scatter is not None and axis is None:
        cbar = plt.colorbar(last_scatter, ax=ax, label=f'Trajectory: {traj_colors_columns}')
        # Set appropriate ticks based on the data range
        if traj_colors_columns in pack:
            values = pack[traj_colors_columns]
            if np.issubdtype(values.dtype, np.number):
                min_val, max_val = values.min(), values.max()
                if max_val - min_val <= 1.0:  # Likely 0-1 range
                    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
                else:  # Other numeric ranges
                    cbar.set_ticks([min_val, (min_val + max_val) / 2, max_val])

    ax.set_xlabel(f'{reduction_method.upper()} Dimension 1')
    ax.set_ylabel(f'{reduction_method.upper()} Dimension 2')
    #ax.set_title(f'All Embeddings with {len(trajectories_to_connect)} Connected Trajectories ({reduction_method.upper()})\n(Gray: all samples, Colored lines: specified trajectories)')

    # Add legend only if no_legend is False
    if not no_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Remove the grid and set ticks to face inwards
    ax.grid(False)
    ax.tick_params(axis='both', direction='in')

    # Only use tight_layout if figure was created here
    if axis is None:
        plt.tight_layout()

    # Do NOT plt.show(); the user will save/plot explicitly.
    return fig, ax


# Combined function for backward compatibility
def plot_embeddings_with_trajectories_v3(pack, trajectories_to_connect, n_most_common=None, reduction_method="tsne", seed=43, figsize=(15, 12), no_legend=True, background_color_key=None, background_colormap="viridis", traj_colors_columns="t_rel", traj_colors_colorbar=False):
    """
    Combined function that computes projection and plots trajectories.
    For backward compatibility - use separate functions for better performance.
    """
    # Use n_most_common if specified
    if n_most_common is not None:
        patient_counts = Counter(pack["patient_scoretype_key"])
        trajectories_to_connect = [patient for patient, count in patient_counts.most_common(n_most_common)]
        print(f"Using top {n_most_common} most common trajectories: {trajectories_to_connect}")
    
    print(f"Total samples: {len(pack['patient_scoretype_key'])}")
    print(f"Trajectories to connect: {trajectories_to_connect}")
    
    # Compute projection
    coords_2d, reducer = compute_2d_projection(pack, reduction_method, seed)
    
    # Plot trajectories
    return plot_trajectories_from_coords(coords_2d, pack, trajectories_to_connect, reduction_method, figsize, no_legend, background_color_key, background_colormap, traj_colors_columns, traj_colors_colorbar)




# ===== Helpers to cache/load 2D projections =====
def _method_to_filenames(method: str) -> Tuple[str, str]:
    """
    Returns (coords_filename, reducer_filename) for a given method.
    Example for "umap": ("umap2d.npy", "umap_reducer.joblib")
    """
    m = method.lower()
    if m not in ("umap", "pca", "tsne"):
        raise ValueError(f"Unknown method: {method}")
    coords_name = f"{m}2d.npy"
    reducer_name = f"{m}_reducer.joblib"
    return coords_name, reducer_name


def save_projection(coords_2d: np.ndarray,
                    reducer,
                    out_dir: str,
                    method: str,
                    save_reducer: bool = False) -> None:
    """
    Save 2D coordinates (+ optional reducer) to disk.
    - coords saved to <out_dir>/{method}2d.npy
    - reducer saved to <out_dir>/{method}_reducer.joblib if save_reducer=True
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    coords_name, reducer_name = _method_to_filenames(method)

    np.save(out / coords_name, np.asarray(coords_2d))

    if save_reducer and reducer is not None:
        if joblib_dump is None:
            raise RuntimeError("joblib is not available to save reducer. Install joblib or set save_reducer=False.")
        joblib_dump(reducer, out / reducer_name)


def load_projection(out_dir: str,
                    method: str,
                    load_reducer: bool = False):
    """
    Load 2D coordinates (+ optional reducer) from disk.
    Returns (coords_2d, reducer_or_None).
    Raises FileNotFoundError if the coords file is missing.
    """
    p = Path(out_dir)
    coords_name, reducer_name = _method_to_filenames(method)

    coords_path = p / coords_name
    if not coords_path.exists():
        raise FileNotFoundError(str(coords_path))

    coords_2d = np.load(coords_path)
    reducer = None
    if load_reducer:
        if joblib_load is None:
            raise RuntimeError("joblib is not available to load reducer. Install joblib or set load_reducer=False.")
        reducer_path = p / reducer_name
        if not reducer_path.exists():
            raise FileNotFoundError(str(reducer_path))
        reducer = joblib_load(reducer_path)

    return coords_2d, reducer


def load_or_compute_projection(pack,
                               out_dir: str,
                               method: str = "umap",
                               seed: int = 43,
                               save_reducer: bool = False):
    """
    Convenience: try to load cached 2D coords from out_dir; if missing, compute
    with compute_2d_projection(pack, method, seed), save to cache, and return.

    Returns (coords_2d, reducer_or_None). For most use cases, reducer is None
    and not needed; coords are sufficient for plotting.
    """
    try:
        coords, reducer = load_projection(out_dir, method, load_reducer=save_reducer)
        print(f"[load_or_compute_projection] Loaded {method} coords from: {out_dir}")
        return coords, reducer
    except FileNotFoundError:
        pass

    coords, reducer = compute_2d_projection(pack, reduction_method=method, seed=seed)
    save_projection(coords, reducer, out_dir, method, save_reducer=save_reducer)
    print(f"[load_or_compute_projection] Computed and cached {method} coords to: {out_dir}")
    return coords, reducer

