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


from ra_utils.training.landmarks.lib import (
    get_transforms, 
    train_epoch, val_epoch, train_loop,
    init_datahandler_from_config
)


from ra_utils.data.data_handler_landmarks_generic import DataHandler_CR_autoscoRA_generic

def main():
    config = load_config(
        default_config="/home/cwatzenboeck/code/RA/ra_utils/runs/config_landmarks/feet/F_train_landmarks_102_debugging.yaml",
        debugging_in_jupyter_nb=False, 
        silencium=False
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
            print(
                f"Debugging = True! Setting MLFLOW_TRACKING_URI to {MLFLOW_TRACKING_URI}")
            os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    else:
        os.environ["MLFLOW_TRACKING_URI"] = config["mlflow_runs_dir"]

    # ------------------------------
    experiment_id = ra_utils.utils.utils_mlflow.get_or_create_experiment(
        config["experiment_name"])

    with mlflow.start_run(experiment_id=experiment_id, run_name=config["run_name"], nested=True):
        package_info_parameters = {
            "package_infos -- ra_utils": ra_utils.utils.utils.package_infos(ra_utils),
            "package_infos -- landmarker": ra_utils.utils.utils.package_infos(landmarker)
        }
        mlflow.log_params(package_info_parameters)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on ", device)

        # Log some parameters from config
        mlflow.log_param("experiment_name", config["experiment_name"])
        mlflow.log_param("run_name", config["run_name"])

        mlflow.log_params(config["model_settings"])
        mlflow.log_params(config["training_parameters"])

        # Log config file
        mlflow.log_dict(config, "config.yml")

        # Get the splits        
        settings = config["data_settings"]
        dataHandler = DataHandler_CR_autoscoRA_generic(
            folder_images=settings["folder_images"],
            df_lm_labels=settings["landmarks_csv"],
            training_test_splits_json=settings["training_test_splits_json"],
            df_autoscoRA_labels_header=settings.get("df_autoscoRA_labels_header", "infer")
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
        ) = dataHandler.get_landmarks_dataset(
            get_pixel_spacing=config["model_settings"].get("get_pixel_spacing", False), 
            pixel_spacing_default = config["model_settings"].get("pixel_spacing_default", [0.1, 0.1])
        )
        lm_names = dataHandler.landmark_names
        mlflow.log_param("landmark_names", lm_names)


        if not isinstance(pixel_spacings_train, (list, tuple, np.ndarray)) or pixel_spacings_train is None:
            pixel_spacings_train = (0.1, 0.1)
        if not isinstance(pixel_spacings_test1, (list, tuple, np.ndarray)) or pixel_spacings_test1 is None:
            pixel_spacings_test1 = (0.1, 0.1)
        if not isinstance(pixel_spacings_test2, (list, tuple, np.ndarray)) or pixel_spacings_test2 is None:
            pixel_spacings_test2 = (0.1, 0.1)

        # Make dataset
        dim_image = config["model_settings"]["dim_image"]
        train_transformd, inference_transformd = get_transforms(config)

        ds_train, ds_test1, ds_test2 = ra_utils.data.dataloader_CR_landmarks.get_landmark_datasets_v2(
            image_paths_train=image_paths_train,
            image_paths_test1=image_paths_test1,
            image_paths_test2=image_paths_test2,
            landmarks_train=landmarks_train,
            landmarks_test1=landmarks_test1,
            landmarks_test2=landmarks_test2,
            pixel_spacings_train=pixel_spacings_train,
            pixel_spacings_test1=pixel_spacings_test1,
            pixel_spacings_test2=pixel_spacings_test2,
            train_transform=train_transformd,
            inference_transform=inference_transformd,
            dim_img=dim_image
        )

        N_landmarks = landmarks_train.shape[1]
        heatmap_generator = GaussianHeatmapGenerator(
            nb_landmarks=N_landmarks,
            sigmas=torch.tensor([config["model_settings"].get(
                "heatmap_sigma", 3.0)], device=device),
            gamma=config["model_settings"]["gamma"],
            heatmap_size=dim_image,
            learnable=config["model_settings"]["learnable_sigmas"],
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


        optimizer_choice = config["training_parameters"].get("optimizer", "adamW")
        if optimizer_choice == "adamW":
            optimizer = torch.optim.AdamW(
                [
                    {'params': model.parameters(), 'weight_decay': 1e-3},
                    {'params': heatmap_generator.sigmas, 'weight_decay': 0.0},
                    {'params': heatmap_generator.rotation, 'weight_decay': 0.0},
                ],
                lr=lr
            )
        elif optimizer_choice == "SGD":
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
        else:
            raise ValueError("Optimizer choice not known ('adamW', 'SGD', ...) ")


        criterion = GaussianHeatmapL2Loss(alpha=config["model_settings"].get("alpha", 5))

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True,
            cooldown=10
        )

        train_loader = DataLoader(
            ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(
            ds_test1, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Testset is not needed 
        # test_loader = DataLoader(
        #     ds_test2, batch_size=batch_size, shuffle=False, num_workers=0)



        # Run training
        model, heatmap_generator = train_loop(
            model=model,
            heatmap_generator=heatmap_generator,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=epochs,
            patience=config["training_parameters"].get("early_stopping_patience", 10),
            scheduler=lr_scheduler, 
            run_full_epochs=config["training_parameters"].get("run_full_epochs", False), 
            save_model=config["SAVE_MODEL"]
        )

        if config["SAVE_MODEL"]: 
            mlflow.pytorch.log_model(model, artifact_path="final_model")
            mlflow.pytorch.log_model(heatmap_generator, "final_heatmap_generator")

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
