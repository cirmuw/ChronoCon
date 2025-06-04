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

import datetime

import  ra_utils.utils.multiprocessing 
from ra_utils.utils.verbosity_enums import *

from pprint import pprint




import ra_utils.utils.optuna
import optuna 
from copy import deepcopy
# from ra_utils.training.scores_SHS.



from ra_utils.training.scores_SHS.run_training_main_lib import run_training_with_cleanup




# Always use last! 
def generate_objective(
        config,
        experiment_id=None,
        verbose: VerboseLevel = VerboseLevel.QUIET, 
        catch_exceptions = False, 
        exception_cost_value = 10, 
        extract_objective_value_from_validation_metrics_dct = lambda metrics_dct: metrics_dct[-1]["Ly"]
    ):



    def objective(trial):
        with mlflow.start_run(nested=True, experiment_id=experiment_id):
            try: # catch cuda.OOM 
                # Define hyperparameters
                config_trial = deepcopy(config)
                ra_utils.utils.optuna.recursive_suggest_trial_parameters(trial, config_trial, 
                                                                         treat_dot_params_special = config.get("hps_treat_dot_params_special", False))


                if verbose in [PRINT_PARAMS, CHATTY]:
                    pprint(config_trial)

                metrics = run_training_with_cleanup(config_trial, 
                                                    mlflow_logging=True, 
                                                    verbose=verbose,
                                                    append_BEST_VAL_as_last=True
                                                    )
                
                error = extract_objective_value_from_validation_metrics_dct(metrics["metrics Val"])

            except torch.cuda.OutOfMemoryError as e:
                if catch_exceptions:
                    print("OOM event occured")
                    print(e)
                    error = exception_cost_value
                else:
                    raise e

            # Log additional information and config
            package_info_parameters = {"package_infos -- ra_utils": ra_utils.utils.utils.package_infos(ra_utils)}
            mlflow.log_params(package_info_parameters)

            # Log config and parameters
            mlflow.log_params(ra_utils.utils.utils.flatten_dict(config_trial))
            mlflow.log_params(ra_utils.utils.utils.log10_params_dct(config_trial))
            mlflow.log_metric("VAL SEARCH_METRIC", error)
            
            artifact_uri = mlflow.get_artifact_uri()
            print("ARTIFACTS URI = ", artifact_uri)
            mlflow.log_dict(config_trial, artifact_file="config_files/config_trial.yml")
        return error

    return objective



# --------------------------------------------------------------#
# --------------------------  main -----------------------------#
# --------------------------------------------------------------#

def run_HP_search_study(verbose : VerboseLevel = PRINT_PARAMS):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"{device = }")

    # Load the configuration
    config, config_name = ra_utils.utils.config_parser.load_config(
        default_config="/home/cwatzenboeck/code/RA/ra_utils/runs/config_scoring/Exp16_reg_head/dev.yml", 
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
        

    #------------------------------
    experiment = mlflow.set_experiment(config["experiment_name"])
    experiment_id = experiment.experiment_id
    # experiment_id = ra_utils.utils.utils_mlflow.get_or_create_experiment(config["experiment_name"])
    config_hps = config["hps"]
    


    # Initiate the parent run and call the hyperparameter tuning child run logic
    with mlflow.start_run(experiment_id=experiment_id, run_name=config["run_name"], nested=False):
        # Initialize the Optuna study

        direction = config_hps.get("search_direction", "minimize")
        search_metric = config_hps.get("search_metric", "Ly")
        n_trials = config_hps.get("n_trials", 10)
        exception_cost_value = config_hps.get("exception_cost_value", 10)

        print(f"Running hyperparameter search with: direction={direction}, search_metric={search_metric}, n_trials={n_trials}")
        study = optuna.create_study(direction=direction)

        # Execute the hyperparameter optimization trials.
        # Note the addition of the `champion_callback` inclusion to control our logging
        extract_objective_value_from_validation_metrics_dct = lambda metrics_dct: metrics_dct[-1][search_metric]
        objective = generate_objective(config,
                                       experiment_id=experiment_id, 
                                       verbose=VerboseLevel.CHATTY, #  VerboseLevel.QUIET, #PRINT_PARAMS,# CHATTY, # change later
                                       catch_exceptions=True, 
                                       exception_cost_value=exception_cost_value,
                                       extract_objective_value_from_validation_metrics_dct = extract_objective_value_from_validation_metrics_dct
                                    )
        
        # objective = generate_objective(config, device) # no early stopping, but average over last couple of epochs
        study.optimize(objective, n_trials=n_trials, 
                       callbacks=[ra_utils.utils.optuna.champion_callback])
        
        
        # log the parameters of the HP search (not seperated)
        mlflow.log_params(study.best_params)
               
        # Extract other paramters from config and save these as well 
        package_info_parameters = {"package_infos -- ra_utils":  ra_utils.utils.utils.package_infos(ra_utils)}
        mlflow.log_params(package_info_parameters)
        
        
        # log the config
        keys_to_ignore = ["transform_params", "model_params", "optimizer_params", "scheduler_params"]
        config_slim = {k: v for k,v in config.items() if k not in [keys_to_ignore]}
        mlflow.log_params(config_slim)


        # Log tags
        mlflow.set_tags(
            tags={
                "project": config["project"],
                "optimizer_engine": "optuna",
            }
        )
        
        artifact_uri = mlflow.get_artifact_uri()
        print("ARTIFACTS URI = ", artifact_uri)
        # NOTE
        # One could log best parameters as artifact (see other HPS file)      
        mlflow.log_dict(study.best_params, artifact_file="config_files/best_params.yml")
        print(f"DONE with full HPS direction={direction}, search_metric={search_metric}, n_trials={n_trials}")
        pprint(study.best_params)

        mlflow.log_dict(config, artifact_file="config_files/hps_config.yml")
        

    
    return None
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    ra_utils.utils.multiprocessing.set_multiprocessing_strategy()
    t1 = datetime.datetime.now()         
    run_HP_search_study()
    print("Done")



    print("Total runtime time:",datetime.datetime.now()-t1)