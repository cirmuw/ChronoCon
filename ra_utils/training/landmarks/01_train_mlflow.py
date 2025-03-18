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

#   My stuff
import ra_utils
import ra_utils.data.data_utils
from  ra_utils.data.data_utils import (
    extract_extras_from_filename, 
    extract_extras_from_abspath
)

import ra_utils.utils
import ra_utils.visualization.plot_landmarks
import ra_utils.data
import ra_utils.data.data_handler
import ra_utils.data.dataloader_CR_landmarks
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
def get_transforms(*args, **kwargs):
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

    train_transformd = Compose([
        Transposed(keys=["image"], indices=(0, 2, 1)),  # CW added
        RandGaussianNoised(('image', ), prob=0.2, mean=0, std=0.1),
        RandScaleIntensityd(('image', ), factors=0.25, prob=0.2),
        RandAdjustContrastd(('image', ), prob=0.2, gamma=(0.5,4.5)),
        RandHistogramShiftd(('image', ), prob=0.2),
        ScaleIntensityd(('image', )),
    ] + spatial_transformd)

    inference_transformd = Compose([
        Transposed(keys=["image"], indices=(0, 2, 1)),  # CW added
        ScaleIntensityd(('image', )),
    ])
    
    return train_transformd, inference_transformd


def main():
    config = load_config(
        default_config="/home/cwatzenboeck/code/RA/ra_utils/runs/config_landmarks/train_landmarks_01.yaml", 
        debugging_in_jupyter_nb=False
    )

    # Debugging option:
    # Set different MLFLOW location
    if config["debugging"]:
        home_dir = Path.home()
        mlflow_debugging_path = home_dir / "data/tmp/mlflow_debugging"
        if not mlflow_debugging_path.exists():
            raise RuntimeError(
                f"Directory {mlflow_debugging_path} (debugging = True) does not exist. "
                f"Please create it or set a valid path."
            )
        else:
            MLFLOW_TRACKING_URI = f"file://{mlflow_debugging_path}"
            print(f"Debugging = True! Setting MLFLOW_TRACKING_URI to {MLFLOW_TRACKING_URI}")
            os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    else: 
        os.environ["MLFLOW_TRACKING_URI"] = config["mlflow_runs_dir"]

    #------------------------------
    experiment_id = ra_utils.utils.utils_mlflow.get_or_create_experiment(config["experiment_name"])
    
    with mlflow.start_run(experiment_id=experiment_id, run_name=config["run_name"], nested=True):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on ", device)

        # Log some parameters from config
        mlflow.log_param("experiment_name", config["experiment_name"])
        mlflow.log_param("run_name", config["run_name"])
        mlflow.log_param("lr", config["training_parameters"]["lr"])
        mlflow.log_param("batch_size", config["training_parameters"]["batch_size"])
        mlflow.log_param("epochs", config["training_parameters"]["epochs"])

        # Load paths 
        base_dir = Path("/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/")
        dataHandler = ra_utils.data.data_handler.DataHandler_CR_autoscoRA(
            folder_H_images = base_dir / "autoscoRA_images/H_images_of_interest_2_renamed_mirrored_inverted_dicoms",
            folder_F_images = base_dir / "autoscoRA_images/F_images_of_interest_2_renamed_mirrored_inverted_dicoms",
            df_lm_labels_H = base_dir / "landmark_data/100_all_H_joints36/points.csv",
            df_lm_labels_F = base_dir / "landmark_data/100_all_F_joints27/points.csv",
            df_autoscoRA_labels_F = base_dir / "autoscoRA_data/autoscoRA_feet.csv",
            df_autoscoRA_labels_H = base_dir / "autoscoRA_data/autoscoRA_hands.csv",
            training_test_splits_json_H = base_dir / "landmark_data/splits/splits_H_TD_25-03-05.json",
            training_test_splits_json_F = base_dir / "landmark_data/splits/splits_F_TD_25-03-05.json",
        )

        (
            image_paths_train,
            image_paths_test1,
            image_paths_test2,
            landmarks_train,
            landmarks_test1,
            landmarks_test2,
            pixel_spacings_train,
            pixel_spacings_test1,
            pixel_spacings_test2
        ) = dataHandler.get_landmarks_dataset_H(
            get_pixel_spacing=config["model_settings"].get("get_pixel_spacing", False)
        )

        # Make dataset 
        dim_image = config["model_settings"]["dim_image"]
        train_transformd, inference_transformd = get_transforms(config)

        ds_train, ds_test1, ds_test2 = ra_utils.data.dataloader_CR_landmarks.get_landmark_datasets(
            image_paths_train = image_paths_train,
            image_paths_test1 = image_paths_test1,
            image_paths_test2 = image_paths_test2,
            landmarks_train = landmarks_train,
            landmarks_test1 = landmarks_test1,
            landmarks_test2 = landmarks_test2,
            pixel_spacings_train=(0.1,0.1),  # pixel_spacings_train if needed
            pixel_spacings_test1=(0.1,0.1),
            pixel_spacings_test2=(0.1,0.1),
            train_transform=train_transformd,
            inference_transform=inference_transformd,
            dim_img=dim_image
        )
        
        N_landmarks = landmarks_train.shape[1]
        heatmap_generator = GaussianHeatmapGenerator(
            nb_landmarks=N_landmarks,
            sigmas=5,
            gamma=10,
            heatmap_size=dim_image,
            learnable=True
        )

        # Init model, optimizer, ...
        model = OriginalSpatialConfigurationNet(
            in_channels=1, 
            out_channels=N_landmarks
        ).to(device)

        print("Number of learnable parameters: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ))

        lr = config["training_parameters"]["lr"]
        batch_size = config["training_parameters"]["batch_size"]
        epochs = config["training_parameters"]["epochs"]

        optimizer = torch.optim.SGD(
            [
                {'params': model.parameters(), "weight_decay": 1e-3},
                {'params': heatmap_generator.sigmas},
                {'params': heatmap_generator.rotation}
            ],
            lr=lr, 
            momentum=0.99, 
            nesterov=True
        )

        criterion = GaussianHeatmapL2Loss(alpha=5)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5,
            patience=10, 
            verbose=True, 
            cooldown=10
        )

        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader   = DataLoader(ds_test1, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader  = DataLoader(ds_test2, batch_size=batch_size, shuffle=False, num_workers=0)

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
                    loss = criterion(outputs, heatmap_generator.sigmas, heatmaps)
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

        def train_loop(model, heatmap_generator, train_loader, val_loader, criterion, optimizer, device, epochs=1000):
            for epoch in range(epochs):
                train_loss = train_epoch(model, heatmap_generator, train_loader, criterion, optimizer, device)
                val_loss, val_mpe = val_epoch(model, heatmap_generator, val_loader, criterion, device)
                
                # Print to console
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train loss: {train_loss:.4f} - "
                      f"Val loss: {val_loss:.4f} - "
                      f"Val mpe: {val_mpe:.4f}")
                
                # Log metrics to MLflow
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_mpe", val_mpe, step=epoch)

                # Step the scheduler
                lr_scheduler.step(val_loss)

        # Run training
        train_loop(
            model=model, 
            heatmap_generator=heatmap_generator, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            device=device,
            epochs=epochs
        )

        # Optionally evaluate on test set
        # test_loss, test_mpe = val_epoch(
        #     model, heatmap_generator, test_loader, criterion, device
        # )
        # print(f"Final Test loss: {test_loss:.4f} - Test mpe: {test_mpe:.4f}")
        # mlflow.log_metric("test_loss", test_loss)
        # mlflow.log_metric("test_mpe", test_mpe)

        # Log the final model as artifact
        # mlflow.pytorch.log_model(model, "model_checkpoint")
        # # If you want to log the heatmap generator state as well:
        # mlflow.pytorch.log_model(heatmap_generator, "heatmap_generator_checkpoint")

        artifact_uri = mlflow.get_artifact_uri()
        print("ARTIFACTS URI = ", artifact_uri)


if __name__ == "__main__":
    # For debugging
    if False:
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ["nnUNet_n_proc_DA"] = '1'
        import multiprocessing
        multiprocessing.set_start_method("spawn")
    main()
