
#%%
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
from landmarker.heatmap import GaussianHeatmapGenerator, LaplacianHeatmapGenerator
from landmarker.models import OriginalSpatialConfigurationNet
from landmarker.losses import GaussianHeatmapL2Loss
from torch.utils.data import DataLoader
from landmarker.visualize import inspection_plot
from landmarker.visualize.utils import prediction_inspect_plot, prediction_inspect_plot_transposed, inspection_plot_numbers
from landmarker.visualize import detection_report
from landmarker.visualize.evaluation import detection_report, convert_to_report_df, evaluate_model_on_loader
from landmarker.data import LandmarkDataset

#   My stuff
import ra_utils
import ra_utils.data.data_utils
from  ra_utils.data.data_utils import (
    extract_extras_from_filename, 
    extract_extras_from_abspath
)

import ra_utils.visualization.plot_landmarks
import ra_utils.data
import ra_utils.data.data_handler
import ra_utils.data.dataloader_CR_landmarks
import ra_utils.visualization.plot_landmarks #.plot_landmarks
import pydicom
import numpy as np

import pydicom
import numpy as np
import pandas as pd


import numpy as np
from sklearn.model_selection import KFold

from ra_utils.data.splits_utils import generate_split_dictionary

import mlflow
from mlflow.tracking import MlflowClient
import yaml
import mlflow.pytorch

from ra_utils.utils.config_parser import load_config
import os 
from landmarker.heatmap.decoder import heatmap_to_coord
from landmarker.metrics import point_error

import ra_utils.inference
import ra_utils.inference.landmarks_inference_utils

from pprint import pprint
from ra_utils.utils.config_parser import load_config
import mlflow
import mlflow.pytorch

def load_models_and_settings(config):
    if config["model"]["reload_from_state_dict"]:
        raise NotADirectoryError()
    else: 
        # define mlflow runs directory
        mlflow_runs_dir = config["model"]["mlflow_runs_dir"] # "/home/cwatzenboeck/data/mlflow_cirpc_tmp/RA/data/" 
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_runs_dir
        run_id = config["model"]["run_id"]
        logged_model_uri = f"runs:/{run_id}/best_model"  # or the path you used
        model = mlflow.pytorch.load_model(logged_model_uri)
        model.eval()

        logged_heatmap_uri = f"runs:/{run_id}/best_heatmap_generator"  # or the path you used
        heatmap_generator = mlflow.pytorch.load_model(logged_heatmap_uri)
        heatmap_generator.eval();

        # client = MlflowClient()
        # local_path = client.download_artifacts(run_id, "config")

        # # Load YAML config
        # with open(local_path, 'r') as f:
        #     config_landmarks = yaml.safe_load(f)
        # pprint(config_landmarks)
        # # TODO 
        # # Get img_size, ...
         
        settings = {
            "dim_image": config["input"]["dim_image"],
            "N_landmarks": config["input"]["N_landmarks"]
        } 
        return model, heatmap_generator, settings
    

def main():

    config = load_config(default_config="/home/cwatzenboeck/code/RA/ra_utils/runs/config_landmarks/inference/F_inference.yaml",
                        debugging_in_jupyter_nb="False",
                        silencium=False)

    #%%
    model, heatmap_generator, settings = load_models_and_settings(config)
    N_landmarks = settings["N_landmarks"]
    dim_image = settings["dim_image"]


    out_path = config["output_dst"]
    if os.path.exists(out_path):
        raise FileExistsError(f"The output file already exists: {out_path}")


    images_folder = Path(config["input"]["image_folder"])
    image_paths = list(images_folder.glob("*.dcm"))
    image_paths = [str(i) for i in image_paths]
    print(f"Files in: {images_folder}   {len(image_paths) = }")

    # Debugging: 
    #image_paths = image_paths[:3]

    # Define transforms
    inference_transformd = Compose([
        #UseOnlyFirstChannel(('image', )),
        Transposed(keys=["image"], indices=(0, 2, 1)),  # CW added
        ScaleIntensityd(('image', )),
    ])


    # Define loader with dummy landmarks: 
    # Create dummy landmarks to reuse the datasetclass of landmarks
    # And to invert the linear transfrom which was used during training 
    # (e.g. translation, transposition, rescaling)
    dummy_landmarks = np.zeros((len(image_paths), N_landmarks, 2), dtype=np.uint16)
    dummy_landmarks[:, 1,0] = 1 # make dummy landmark 1 the unit vector x
    dummy_landmarks[:, 2,1] = 1 # make dummy landmark 2 the unit vector y

    ds = LandmarkDataset(
        image_paths,
        dummy_landmarks,
        pixel_spacing=None,
        transform=inference_transformd,
        dim_img=dim_image
    )

    loader = DataLoader(
            ds,
            batch_size=1,
            num_workers=0,
            shuffle=False
        )



    (all_pred_landmarks, all_pred_landmarks_transformed, all_dim_origs, all_pixel_spacings, all_paddings
    ) = ra_utils.inference.landmarks_inference_utils.predict_landmarks(model, loader, device="cuda")
    
    
    
    column_names = ["image_path"] + [f"landmark__{i}_{axis}" for i in range(len(all_pred_landmarks_transformed[0])) for axis in ["x", "y"]]
    landmarks_flat = [list(np.array(landmark).flatten()) for landmark in all_pred_landmarks_transformed]
    landmarks_df_separate = pd.DataFrame(
        [[image_paths[i]] + landmarks_flat[i] for i in range(len(image_paths))],
        columns=column_names
    )
    landmarks_df_separate["file_name"] = landmarks_df_separate["image_path"].apply(lambda x: Path(x).name)
    landmarks_df_separate.to_csv(out_path, index=False)    
    print("Saving csv to ", out_path)



if __name__ == "__main__":
    main()