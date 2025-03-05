import numpy as np
import pandas as pd
import torch
import pydicom
import matplotlib.pyplot as plt

from pathlib import Path

import ra_utils.data.data_utils




def plot_landmarks(image: str | Path | torch.Tensor, 
                   landmarks, 
                   landmarks_targets=None, 
                   figsize=(12, 10)):
    """
    Plots an image (either from a file path, a NumPy array, or a PyTorch tensor) with landmarks.

    Args:
        image (Union[str, np.ndarray, torch.Tensor]): Image source, either a DICOM file path, 
                                                      a NumPy array, or a PyTorch tensor.
        landmarks (np.ndarray): Numpy array of shape (N, 2), where N is the number of landmarks.
                                Each row represents (x, y) coordinates.
        landmarks_targets (np.ndarray, optional): Ground truth landmarks of the same shape (N, 2).
                                                  If provided, they will be plotted in blue.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """

    # Load the image if a path is provided
    if isinstance(image, (str, Path)):  
        dcm = pydicom.dcmread(image)
        image = dcm.pixel_array

    # Convert tensor to numpy array if necessary
    if isinstance(image, torch.Tensor):
        image = image.squeeze().cpu().numpy()

    # Ensure image is in 2D format
    assert len(image.shape) == 2, "Image should be a 2D grayscale array"

    # Ensure landmarks are properly shaped
    assert landmarks.shape[1] == 2, "Landmarks should have shape (N, 2)"

    if landmarks_targets is not None:
        assert landmarks_targets.shape == landmarks.shape, \
            "landmarks_targets must have the same shape as landmarks"

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, cmap='gray')

    # Plot detected landmarks
    ax.scatter(landmarks[:, 0], landmarks[:, 1], s=20, color='red', marker='x', label="Predicted")

    # Plot target landmarks if provided
    if landmarks_targets is not None:
        ax.scatter(landmarks_targets[:, 0], landmarks_targets[:, 1], 
                   s=20, color='blue', marker='o', label="Target")

    # Hide axes and add legend if needed
    ax.axis("off")
    if landmarks_targets is not None:
        ax.legend()

    return fig  # Return the figure object


def plot_landmarks_from_df(dfm, image_idx=0):
    """
    Extracts the image path and landmarks from the DataFrame and plots them.

    Args:
        dfm (pd.DataFrame): DataFrame containing image paths and landmarks.
        image_idx (int): Index of the image to be plotted.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    # Extract image path and landmarks
    image_path = dfm["image"].iloc[image_idx]
    landmarks = ra_utils.data.data_utils.extract_landmarks_from_df(dfm, image_idx)

    # Plot the image with landmarks
    return plot_landmarks(image_path, landmarks)