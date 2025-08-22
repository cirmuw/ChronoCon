"""
Successor of  05_training_mlflow_triplet.py: 
Change interface of loss functions. 
"""


import os
from pathlib import Path
import torch
import mlflow
import torch
import torch.nn as nn

import ra_utils
import ra_utils.utils.config_parser
import ra_utils.utils.utils_mlflow
import ra_utils.utils.optuna
from ra_utils.utils.utils_SHS_scoring import get_classes
import ra_utils.utils.utils

from ra_utils.data.dataloader_CR_patches import (
    process_several_score_groups,
    dataset_and_loader_several,
    check_duplicates_in_dataloader
)

import ra_utils.networks.loss_function
from ra_utils.networks.loss_function import get_score_loss_function, get_triplet_loss_fn

import torchvision.transforms.v2 as v2
import ra_utils.utils.config_parser


from ra_utils.training.scores_SHS.model_builders import build_models_AE_v2, build_models_AE, build_models_AE_v1_and2
from typing import Optional

import  ra_utils.utils.multiprocessing 
from ra_utils.utils.verbosity_enums import *


from ra_utils.training.scores_SHS.run_training_main_lib import run_training_v2

# --------------------------------------------------------------#
# --------------------------  main -----------------------------#
# --------------------------------------------------------------#
def main():

    # Load the configuration
    config, config_name = ra_utils.utils.config_parser.load_config(
        #default_config="/home/cwatzenboeck/code/RA/ra_utils/runs/config_scoring/DEV_DELTA/dev.yml", 
        default_config="/home/cwatzenboeck/code/RA/ra_utils/runs/config_scoring/Exp31_hps/dev_triplet.yml",
        debugging_in_jupyter_nb=False, silencium=False, return_config_name=True, 
        # default_path_substitution_config="/home/cwatzenboeck/code/RA/ra_utils/runs/path_sustitution/cirpc_to_msc.yml"
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

    if config["debugging"]:
        print("overwriting some config params")
        config["training"]["epochs"] = 1
        config["SAVE_MODEL_FULL"] = False
        config["SAVE_MODEL_state_dct"] = False
        if config.get("RESET_IMAGE_FOLDER_WHEN_DEBUGGING", False):
            config["data"]["image_path_folder"] = [
                "/home/cwatzenboeck/data/AutoPIX_local_data/dev_cw/ds_480/H_patches_size1.5_DEBUGGING/",
                "/home/cwatzenboeck/data/AutoPIX_local_data/dev_cw/ds_480/F_patches_size1.5_DEBUGGING/"
            ]
        


    
    
    experiment = mlflow.set_experiment(config["experiment_name"])
    experiment_id = experiment.experiment_id
    

    with mlflow.start_run(experiment_id=experiment_id, run_name=config["run_name"], nested=True) as run:

        artifact_uri = mlflow.get_artifact_uri()
        print("ARTIFACTS URI = ", artifact_uri)

        # Log package info
        package_info_parameters = {
            "package_infos -- ra_utils": ra_utils.utils.utils.package_infos(ra_utils)
        }
        mlflow.log_params(package_info_parameters)
        run_id = run.info.run_id
        print("RUN ID: ", run_id)

        # Log config file
        mlflow.log_params(ra_utils.utils.utils.flatten_dict(config))
        mlflow.log_params(ra_utils.utils.utils.log10_params_dct(config))
        mlflow.log_dict(config, "config.yml")


        metrics = run_training_v2(config = config,  
                               mlflow_logging=True, 
                               verbose=VerboseLevel.CHATTY,
                               config_name=config_name)
        

        artifact_uri = mlflow.get_artifact_uri()
        print("ARTIFACTS URI = ", artifact_uri)
        
        run_id = run.info.run_id
        print("RUN ID: ", run_id)

if __name__ == "__main__":
    ra_utils.utils.multiprocessing.set_multiprocessing_strategy()
    main()
    print("Done")
