# Separate computation and plotting functions
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import numpy as np
from collections import defaultdict

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

def plot_trajectories_from_coords(coords_2d, pack, trajectories_to_connect, 
                                  reduction_method="tsne", 
                                  figsize=(5, 4), 
                                  no_legend=True, 
                                  background_color_key=None, 
                                  background_colormap="viridis",
                                  traj_colors_columns="t_rel",
                                  traj_colors_colorbar=False):
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
        background_colormap: Colormap for numeric background coloring (default: "viridis")
        traj_colors_columns: Key in pack to use for trajectory point coloring (default: "t_rel")
        traj_colors_colorbar: If True, show colorbar for trajectory point colors (default: False)
        
    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    print(f"Plotting {len(trajectories_to_connect)} trajectories...")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Group data by patient_scoretype_key
    groups = defaultdict(list)
    for i, key in enumerate(pack['patient_scoretype_key']):
        groups[key].append(i)
    
    # Sort each group by years_since_2000
    for key in groups:
        groups[key].sort(key=lambda i: pack['years_since_2000'][i])
    
    # Define colors for the trajectories to connect
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
                scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                                   c=background_values, cmap=background_colormap, s=20, alpha=0.3, 
                                   label='All other samples')
                
                # Add colorbar for background
                cbar = plt.colorbar(scatter, ax=ax, label=f'Background: {background_color_key}')
                
            except (ValueError, TypeError):
                # Non-numeric values, use as categorical
                print(f"Non-numeric values detected, using categorical coloring")
                ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                          c=background_values, s=20, alpha=0.3, label='All other samples')
        else:
            print(f"Warning: '{background_color_key}' not found in pack!")
            print(f"Available keys in pack: {list(pack.keys())}")
            print("Using default gray background")
            ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                      c='lightgray', s=20, alpha=0.3, label='All other samples')
    else:
        ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                  c='lightgray', s=20, alpha=0.3, label='All other samples')
    
    # Plot trajectories for specified patients
    for patient_key in trajectories_to_connect:
        if patient_key not in groups or len(groups[patient_key]) < 2:
            print(f"Warning: {patient_key} has less than 2 points, skipping trajectory")
            continue
        
        indices = groups[patient_key]
        patient_coords = coords_2d[indices]
        
        # Plot trajectory line
        ax.plot(patient_coords[:, 0], patient_coords[:, 1], 
                color=patient_colors[patient_key], linewidth=3, alpha=0.8, 
                label=f'{patient_key} ({len(indices)} points)')
        
        # Plot points for this patient
        if traj_colors_columns == None: 
            patient_t_rel = "gray"
        else: 
            patient_t_rel = pack[traj_colors_columns][indices] if traj_colors_columns in pack else np.linspace(0, 1, len(indices))
        scatter = ax.scatter(patient_coords[:, 0], patient_coords[:, 1], 
                           c=patient_t_rel, cmap='viridis', s=100, 
                           edgecolors=patient_colors[patient_key], linewidth=2,
                           alpha=0.9, zorder=5)
    
    # Add colorbar for trajectory colors if requested
    if traj_colors_colorbar and len(trajectories_to_connect) > 0:
        cbar = plt.colorbar(scatter, ax=ax, label=f'Trajectory: {traj_colors_columns}')
        # Set appropriate ticks based on the data range
        if traj_colors_columns in pack:
            values = pack[traj_colors_columns]
            if np.issubdtype(values.dtype, np.number):
                min_val, max_val = values.min(), values.max()
                if max_val - min_val <= 1.0:  # Likely 0-1 range
                    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
                else:  # Other numeric ranges
                    cbar.set_ticks([min_val, (min_val + max_val)/2, max_val])
    
    ax.set_xlabel(f'{reduction_method.upper()} Dimension 1')
    ax.set_ylabel(f'{reduction_method.upper()} Dimension 2')
    #ax.set_title(f'All Embeddings with {len(trajectories_to_connect)} Connected Trajectories ({reduction_method.upper()})\n(Gray: all samples, Colored lines: specified trajectories)')
    
    # Add legend only if no_legend is False
    if not no_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
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



