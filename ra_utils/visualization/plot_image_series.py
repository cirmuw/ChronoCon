
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime


def plot_patch_series(data_batch, pat_id: str, axes=None):

    # Select entries for that patient
    m = np.array(data_batch["patient_id"]) == pat_id
    pat_ids = np.array(data_batch["patient_id"])[m]
    dates_str = np.array(data_batch["date_str"])[m]
    scores = np.array(data_batch["score"])[m]
    sides = np.array(data_batch["left_or_right"])[m]
    score_types = np.array(data_batch["score_type"])[m]
    roi_names = np.array(data_batch["roi_name"])[m]
    images = [img for (cnt, img) in enumerate(data_batch["img"]) if m[cnt]]

    # Convert dates to datetime objects for sorting
    dates_dt = np.array([datetime.strptime(d, "%Y%m%d") for d in dates_str])
    sorted_indices = np.argsort(dates_dt)

    # Sort everything
    dates_sorted = dates_str[sorted_indices]
    scores_sorted = scores[sorted_indices]
    sides_sorted = sides[sorted_indices]
    score_types_sorted = score_types[sorted_indices]
    roi_names_sorted = roi_names[sorted_indices]
    images_sorted = [images[i] for i in sorted_indices]

    # Separate left and right images
    L_indices = [i for i, side in enumerate(sides_sorted) if side == "L"]
    R_indices = [i for i, side in enumerate(sides_sorted) if side == "R"]

    n_L, n_R = len(L_indices), len(R_indices)
    max_n = max(n_L, n_R)

    fig, axes = plt.subplots(
        2, max_n,
        figsize=(4 * max_n, 10),
        gridspec_kw={"height_ratios": [1, 1]}
    )

    # Ensure axes is 2D
    if max_n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    # Plot left images (row 0)
    for i, idx in enumerate(L_indices):
        ax = axes[0, i]
        ax.imshow(images_sorted[idx].cpu().numpy().squeeze(), cmap="gray")
        ax.set_title(f"Date: {dates_sorted[idx]}\nScore: {scores_sorted[idx]}\nType: {score_types_sorted[idx]}")
        ax.axis('off')
    for j in range(n_L, max_n):
        axes[0, j].axis('off')

    # Plot right images (row 1)
    for i, idx in enumerate(R_indices):
        ax = axes[1, i]
        ax.imshow(images_sorted[idx].cpu().numpy().squeeze(), cmap="gray")
        ax.set_title(f"Date: {dates_sorted[idx]}\nScore: {scores_sorted[idx]}\nType: {score_types_sorted[idx]}")
        ax.axis('off')
    for j in range(n_R, max_n):
        axes[1, j].axis('off')

    # Add row labels "L" and "R"
    fig.text(0.03, 0.73, "L", fontsize=30, va='center', ha='center', weight='bold')
    fig.text(0.03, 0.27, "R", fontsize=30, va='center', ha='center', weight='bold')

    fig.suptitle(f'Patient ID: {pat_id}', fontsize=18)
    plt.subplots_adjust(hspace=0.5, left=0.05)
    plt.show()
    
    
    
    

