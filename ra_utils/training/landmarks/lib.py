import numpy as np
import pandas as pd
import torch
import pydicom
import matplotlib.pyplot as plt

from pathlib import Path

# MONAI imports
import monai
from monai.data import Dataset, CacheDataset, DataLoader, PILReader
from monai.transforms import (
    LoadImage, LoadImaged, Resized, Compose, SaveImage,
    Spacingd, SpatialCropd, ResizeWithPadOrCropd
)

import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    Transposed,
    NormalizeIntensityd,
    MapTransform,
    ScaleIntensityRangePercentilesd,
    RandAffined, RandGaussianNoised,
    RandStdShiftIntensityd, RandScaleIntensityd, RandAdjustContrastd, RandHistogramShiftd,
    ScaleIntensityd, Lambdad,
    LoadImage, Transpose
)

from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

import landmarker
import landmarker.datasets
from landmarker.datasets import get_cepha_landmark_datasets
from landmarker.heatmap import GaussianHeatmapGenerator
from landmarker.models import OriginalSpatialConfigurationNet
from landmarker.losses import GaussianHeatmapL2Loss
from torch.utils.data import DataLoader
from landmarker.visualize import inspection_plot
from landmarker.visualize.utils import prediction_inspect_plot, prediction_inspect_plot_transposed
from landmarker.heatmap.generator import GaussianHeatmapGenerator
from landmarker.data import LandmarkDataset
from landmarker.heatmap.decoder import heatmap_to_coord
from landmarker.metrics import point_error

import copy


#   My stuff
import ra_utils
import ra_utils.data.data_utils
from ra_utils.data.data_utils import (
    extract_extras_from_filename,
    extract_extras_from_abspath
)

import ra_utils.utils
import ra_utils.visualization.plot_landmarks
import ra_utils.data
import ra_utils.data.data_handler
import ra_utils.data.dataloader_CR_landmarks
import ra_utils.utils.utils_mlflow
import pydicom
import numpy as np
import pandas as pd

import numpy as np
from sklearn.model_selection import KFold

from ra_utils.data.splits_utils import generate_split_dictionary

from ra_utils.utils.config_parser import load_config
import os
import mlflow
import mlflow.pytorch


# Define transforms
def get_transforms(config):
    fn_keys = ('image',)
    spatial_transformd = [
        RandAffined(
            fn_keys,
            prob=1,
            rotate_range=(-np.pi/12, np.pi/12),
            translate_range=(-10, 10),
            scale_range=(-0.1, 0.1),
            shear_range=(-0.1, 0.1)
        )
    ]

    config_Tr = config["training_parameters"]
    train_transformd = Compose([
        Transposed(keys=["image"], indices=(0, 2, 1)),  # CW added
        RandGaussianNoised(('image', ), prob=0.2, mean=0, std=0.1),
        RandScaleIntensityd(('image', ), factors=0.25, prob=0.2),
        RandAdjustContrastd(('image', ), prob=0.2, gamma=(
            config_Tr["RandAdjustContrast_gamma_lower"], config_Tr["RandAdjustContrast_gamma_upper"])),
        RandHistogramShiftd(('image', ), prob=0.2),
        ScaleIntensityd(('image', )),
    ] + spatial_transformd)

    inference_transformd = Compose([
        Transposed(keys=["image"], indices=(0, 2, 1)),  # CW added
        ScaleIntensityd(('image', )),
    ])

    return train_transformd, inference_transformd



def train_epoch(model, heatmap_generator, train_loader, criterion, optimizer, device):
    running_loss = 0
    model.train()
    for i, batch in enumerate(tqdm(train_loader)):
        images = batch["image"].to(device)
        landmarks = batch["landmark"].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        heatmaps = heatmap_generator(landmarks)
        loss = criterion(outputs, heatmap_generator.sigmas, heatmaps)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def val_epoch(model, heatmap_generator, val_loader, criterion, device, method="local_soft_argmax"):
    eval_loss = 0
    eval_mpe = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            images = batch["image"].to(device)
            landmarks = batch["landmark"].to(device)
            outputs = model(images)
            dim_orig = batch["dim_original"].to(device)
            pixel_spacing = batch["spacing"].to(device)
            padding = batch["padding"].to(device)
            heatmaps = heatmap_generator(landmarks)
            loss = criterion(
                outputs, heatmap_generator.sigmas, heatmaps)
            pred_landmarks = heatmap_to_coord(outputs, method=method)
            eval_loss += loss.item()
            eval_mpe += point_error(
                landmarks,
                pred_landmarks,
                images.shape[-2:],
                dim_orig,
                pixel_spacing,
                padding,
                reduction="mean"
            )
    return eval_loss / len(val_loader), eval_mpe / len(val_loader)


def train_loop(
    model,
    heatmap_generator,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=1000,
    patience=10,
    scheduler=None,
    run_full_epochs=False, 
    save_model = True
):
    """
    If run_full_epochs=True, training proceeds through all `epochs` without early stopping,
    and the function returns the model/heatmap_generator in their final state.
    Otherwise, early stopping is used and the "best" weights (lowest val_mpe) are restored
    before returning.
    """

    best_val_mpe = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_heatmap_wts = copy.deepcopy(heatmap_generator.state_dict())
    epochs_no_improve = 0

    for epoch in range(epochs):
        # ---- Train for one epoch ----
        train_loss = train_epoch(
            model, heatmap_generator, train_loader, criterion, optimizer, device
        )

        # ---- Validate for one epoch ----
        val_loss, val_mpe = val_epoch(
            model, heatmap_generator, val_loader, criterion, device
        )

        # ---- Print to console ----
        print(f"Epoch {epoch+1}/{epochs} - "
            f"Train loss: {train_loss:.4f} - "
            f"Val loss: {val_loss:.4f} - "
            f"Val mpe: {val_mpe:.4f}")

        # ---- Log metrics to MLflow ----
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_mpe", val_mpe, step=epoch)

        # ---- Update scheduler (if provided) ----
        if scheduler is not None:
            scheduler.step(val_loss)

        # ---- If we are not forcing full epochs, do early stopping checks ----
        if not run_full_epochs:
            if val_mpe < best_val_mpe:
                best_val_mpe = val_mpe
                best_model_wts = copy.deepcopy(model.state_dict())
                best_heatmap_wts = copy.deepcopy(heatmap_generator.state_dict())
                epochs_no_improve = 0

                # Log these weights as the best so far
                if save_model: 
                    mlflow.pytorch.log_model(model, artifact_path="best_model")
                    mlflow.pytorch.log_model(heatmap_generator, "best_heatmap_generator")
            else:
                epochs_no_improve += 1

            # If no improvement for 'patience' epochs, stop
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

        # ---- Otherwise, if forcing full epochs, just keep going until the end ----
        # (No early stopping logic here.)

    # ---- After training loop ----
    if not run_full_epochs:
        # Restore the best model/heatmap weights when early stopping was in use
        model.load_state_dict(best_model_wts)
        heatmap_generator.load_state_dict(best_heatmap_wts)

    return model, heatmap_generator



import ra_utils.data.data_handler
#### Data:::
def init_datahandler_from_config(config: dict, 
                                 base_dir="/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/"):
    base_dir = Path(base_dir)
    dataHandler = ra_utils.data.data_handler.DataHandler_CR_autoscoRA(
        folder_H_images=base_dir / "autoscoRA_images/H_images_of_interest_2_renamed_mirrored_inverted_dicoms",
        folder_F_images=base_dir / "autoscoRA_images/F_images_of_interest_2_renamed_mirrored_inverted_dicoms",
        df_lm_labels_H=config.get("data_settings", {}).get("landmarks_csv_H",  "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/landmark_data/100_all_H_joints36/points_with_names.csv"),
        df_lm_labels_F= config.get("data_settings", {}).get("landmarks_csv_F", "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/landmark_data/100_all_F_joints27/points_with_names.csv"),
        df_autoscoRA_labels_F=config.get("data_settings", {}).get("landmarks_csv_F", base_dir / "autoscoRA_data/autoscoRA_feet.csv"),
        df_autoscoRA_labels_H=config.get("data_settings", {}).get("landmarks_csv_H", base_dir / "autoscoRA_data/autoscoRA_hands.csv"),
        
        training_test_splits_json_H=config["data_settings"]["training_test_splits_json_H"],
        training_test_splits_json_F=config["data_settings"]["training_test_splits_json_F"],
        df_autoscoRA_labels_F_header = config["data_settings"].get("df_autoscoRA_labels_F_header", None),  # infer is default; use None when no header is available
        df_autoscoRA_labels_H_header = config["data_settings"].get("df_autoscoRA_labels_H_header", None)
    )
    return dataHandler





