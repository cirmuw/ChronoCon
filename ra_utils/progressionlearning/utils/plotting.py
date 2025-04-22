import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def plot_metric(metric_name, scores, mean, std, savepath):
    """Helper function to generate metric plots."""
    plt.figure(figsize=(8, 6))
    plt.plot(scores, marker='o', label=f'{metric_name} Scores', linestyle='-', color='blue')
    plt.axhline(mean, color='red', linestyle='--', label=f'Mean {metric_name} ({mean:.3f})')
    plt.fill_between(
        range(len(scores)),
        [mean - std] * len(scores),
        [mean + std] * len(scores),
        color='red',
        alpha=0.2,
        label=f'Standard Deviation ({std:.3f})'
    )
    plt.title(f'{metric_name} Across Batches', fontsize=16)
    plt.xlabel('Batch', fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.5)
    plt.savefig(savepath + f'metrics_{metric_name}.jpg')
    plt.close()


def do_reduction(frozen_features_dict, reduction='tsne', timepoints=['T0', 'T1', 'T2', 'T3']):
    """
    Perform dimensionality reduction on frozen features from different timepoints.

    Parameters
    ----------
    frozen_features_dict : dict
        Contains the frozen features for each timepoint (T0, T1, T2, T3).
    reduction : str
        Type of dimensionality reduction to perform. Options are 'tsne' or 'pca'.

    Returns
    -------
    reduced : np.ndarray
        Reduced features with shape (n_samples, 2).
    """
    data = np.concatenate([frozen_features_dict[t] for t in timepoints], axis=0)
    if reduction == 'tsne':
        reduced = TSNE(n_components=2, random_state=45, metric='cosine').fit_transform(data)
    elif reduction == 'pca':
        reduced = PCA(n_components=2).fit_transform(data)
    return reduced 


def plt_timelines_with_trajectories(features,  axs, timepoints, reduction='tsne', num_patients=10):
    # Number of points per timepoint (e.g., 100 patients)
    """
    Plots temporal trajectories of features across multiple timepoints.

    Parameters
    ----------
    features : np.ndarray
        Array of shape (n_samples, 2) containing reduced feature coordinates.
    axs : matplotlib.axes.Axes
        Matplotlib axes object to plot the timelines.
    timepoints : np.ndarray
        Array of shape (n_samples,) indicating the timepoint for each sample.
    reduction : str, optional
        Type of dimensionality reduction performed, used in plot title. Default is 'tsne'.
    num_patients : int, optional
        Number of patients to randomly highlight in the plot. Default is 10.

    Notes
    -----
    - Highlights the trajectories of selected patients with black lines.
    - Scatter plots all data points with reduced opacity and highlights selected patients with full opacity.
    - Adds a colorbar indicating the timepoint and sets a title based on the reduction method.
    """
    num_points = features.shape[0] // 4  # Assuming pca_results has points for T0, T1, T2, T3 stacked
    # Pick 5 random patients (or specific ones) to highlight their trajectories
    highlight_indices = np.random.choice(num_points, size=num_patients, replace=False) 
    highlight_indices = [1,2,3,4,5]

    # 1. Plot the highlighted trajectories for 5 patients with black lines
    for i in highlight_indices:
        # Indices for the current patient (p_i) across T0, T1, T2, T3
        indices = [i, i + num_points, i + 2 * num_points, i + 3 * num_points]
        
        # Plot lines connecting p_i across timepoints with black lines
        axs.plot(features[indices, 0], features[indices, 1], c='black', alpha=1.0, linestyle='-', marker='o', markersize=1)


    # Scatter plot for all points with reduced opacity (alpha=0.2) to fade them, but still colored by timepoint
    scatter = axs.scatter(features[:, 0], features[:, 1], c=timepoints, cmap='viridis', alpha=0.2)

    # 3. Plot the highlighted 5 patients' points again with full opacity
    for i in highlight_indices:
        # Scatter the 5 selected patients with their original colors but full opacity
        indices = [i, i + num_points, i + 2 * num_points, i + 3 * num_points]
        axs.scatter(features[indices, 0], features[indices, 1], c=timepoints[indices], cmap='viridis', alpha=1.0, s=70, edgecolors='black')
    # Add a colorbar and set titles
    plt.colorbar(scatter, ax=axs, label="Timepoint (T0-T3)")
    axs.set_title(f"{reduction}: Temporal Trajectories")



def plt_timelines(features,  axs, timepoints, reduction='tsne'):
    # Scatter plot for all points with reduced opacity (alpha=0.2) to fade them, but still colored by timepoint
    """
    Scatter plots the reduced features, coloring each point by its timepoint.

    Parameters
    ----------
    features : np.ndarray
        Array of shape (n_samples, 2) containing reduced feature coordinates.
    axs : matplotlib.axes.Axes
        Matplotlib axes object to plot the timelines.
    timepoints : np.ndarray
        Array of shape (n_samples,) indicating the timepoint for each sample.
    reduction : str, optional
        Type of dimensionality reduction performed, used in plot title. Default is 'tsne'.
    """
    scatter = axs.scatter(features[:, 0], features[:, 1], c=timepoints, cmap='viridis', alpha=0.2)
    plt.colorbar(scatter, ax=axs, label="Timepoint (T0-T3)")
    axs.set_title(f"{reduction}: Temporal Trajectories")


def plt_label(features,  axs, labels, reduction='tsne'):   
    """
    Scatter plots the reduced features, coloring each point by its label.

    Parameters
    ----------
    features : np.ndarray
        Array of shape (n_samples, 2) containing reduced feature coordinates.
    axs : matplotlib.axes.Axes
        Matplotlib axes object to plot the label distribution.
    labels : np.ndarray
        Array of shape (n_samples,) indicating the label for each sample.
    reduction : str, optional
        Type of dimensionality reduction performed, used in plot title. Default is 'tsne'.
    """
    unique_labels = np.unique(labels)
    
    for i, label in enumerate(unique_labels):
        idx = np.where(labels == label)
        axs.scatter(features[idx, 0], features[idx, 1], label=f'Class {label}', alpha=0.7)
    axs.legend(['T3 pcr ==1', 'other'])
    axs.set_title(f"{reduction}: Label Distribution")
  