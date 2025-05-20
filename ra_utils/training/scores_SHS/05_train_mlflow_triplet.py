import os
from pathlib import Path
import torch
import mlflow
import torch.nn as nn

import ra_utils
import ra_utils.utils.config_parser
import ra_utils.utils.utils_mlflow
import ra_utils.utils.optuna
from ra_utils.utils.utils_SHS_scoring import get_classes
import ra_utils.utils.utils

from ra_utils.data.dataloader_CR_patches import (
    process_several_score_groups,
    dataset_and_loader_several
)

from ra_utils.training.scores_SHS.scores_SHS_training_lib_AE_v1 import (
    evaluate_and_log_testset_results_AE_v2,
    train_loop_AE_v2,
    train_loop_AE_v3
)

import ra_utils.networks.loss_function
from ra_utils.networks.loss_function import get_score_loss_function, get_triplet_loss_fn

import torchvision.transforms.v2 as v2
import ra_utils.utils.config_parser


from ra_utils.training.scores_SHS.model_builders import build_models_AE_v2, build_models_AE, build_models_AE_v1_and2

import online_triplet_loss.losses
from typing import Optional




# --------------------------------------------------------------#
# --------------------------  main -----------------------------#
# --------------------------------------------------------------#
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the configuration
    config, config_name = ra_utils.utils.config_parser.load_config(
        default_config="/home/cwatzenboeck/code/RA/ra_utils/runs/config_scoring/Exp11/H_JSN_ResNetMTANAE_lx100_FL.yml", 
        debugging_in_jupyter_nb=False, silencium=False, return_config_name=True)
    

    

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
        

    # ------------------------------
    experiment_id = ra_utils.utils.utils_mlflow.get_or_create_experiment(
        config["experiment_name"])

    with mlflow.start_run(experiment_id=experiment_id, run_name=config["run_name"], nested=True):
        artifact_uri = mlflow.get_artifact_uri()
        print("ARTIFACTS URI = ", artifact_uri)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on ", device)


        # Log config file
        mlflow.log_params(ra_utils.utils.utils.flatten_dict(config))
        mlflow.log_dict(config, "config.yml")

        # Load tables with paths and scores (+ split)
        # data_tables = load_img_SHS_patch_data(config["data"])
        data_tables = process_several_score_groups(config["data"])

        # Make dataset and dataloaders
        # data = dataset_and_loader(data_tables, config)
        data = dataset_and_loader_several(data_tables, config)

        # Load/ make model
        attention_paths_dct = config["data"].get("network_score_groups")
        if attention_paths_dct is None:
            print("'network_score_groups' not found in config file. Using default attention paths: 'score_groups'")
            attention_paths_dct = config["data"]["score_groups"]

        classifier_head_infos = config["data"]["classifier_head_infos"]
        if config["data"].get("how_to_deal_with_surgery")  == "keep: map over limit to limit plus one":
            print("Adding one to the output dimension of the classifier heads! (surgery class)")
            for k in classifier_head_infos.keys():
                v = classifier_head_infos[k]["out_dim"]
                classifier_head_infos[k]["out_dim"] = v + 1

        model_name = config["model_name"]
        model_AE, model_c = build_models_AE_v1_and2(
                                            model_name, config, 
                                            classifier_head_infos = classifier_head_infos, 
                                            attention_paths_dct = attention_paths_dct
                                            )

        model_AE.to(device)
        model_c.to(device)

        # define loss function
        loss_fn_y = get_score_loss_function(config["loss"]["score"])
        loss_fn_x = nn.MSELoss()
        loss_fn_z = nn.L1Loss()
        loss_fn_z_triplet_classes = get_triplet_loss_fn(config["loss"].get("triplet_scores", {}))

        # ---- joint optimiser with separate lrs --------------------------------
        opt_cfg = config["optimizer_params"].copy()
        lr_ae  = opt_cfg["learning_rates"]["encoder__OR__decoder"]
        lr_clf = opt_cfg["learning_rates"]["classifier"]
        param_groups = [
            {"params": model_AE.parameters(), "lr": lr_ae},
            {"params": model_c.parameters(),  "lr": lr_clf},
        ]
        print(opt_cfg["other_optimizer_kwargs"])
        optimizer = torch.optim.AdamW(param_groups, **opt_cfg["other_optimizer_kwargs"])
        
        
        transform_AE = v2.GaussianNoise(mean=0, sigma = 0.05, clip=True)

        # scheduler
        scheduler = None
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)


        train_dataloaders = {k: data[k]["train_loader"] for k in data.keys()}
        val_loaders = {k: data[k]["val_loader"] for k in data.keys()}
        test_loaders = {k: data[k]["test_loader"] for k in data.keys()}
        epochs = config["training"]["epochs"]

        classes = None # get_classes(config) # TO be removed

        print("Start training for: ", config_name)
        model_AE, model_c = train_loop_AE_v3(
            model_AE=model_AE,
            model_classifier=model_c,
            train_dataloaders=train_dataloaders,
            val_loaders=val_loaders,
            loss_fn_x=loss_fn_x,
            loss_fn_y=loss_fn_y,
            loss_fn_z=loss_fn_z,
            loss_fn_z_triplet_classes = loss_fn_z_triplet_classes,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=epochs,
            patience=config["training"].get("early_stopping_tol", 100),
            lambda_x=config.get('loss_weights', {}).get('lambda_x', 0.0),   # used to be 1...   
            lambda_y=config.get('loss_weights', {}).get('lambda_y', 0.0),   # used to be 1...   
            lambda_z=config.get('loss_weights', {}).get('lambda_z', 0.0),   # used to be 1...   
            lambda_z_triplet_classes = config.get('loss_weights', {}).get('lambda_z_triplet_classes', 0.0),  # used to be 1...              
            transform=transform_AE,
            classes=classes,
            log_model=config["SAVE_MODEL"],
            verbose=3,
            ES_metric_key=config["training"].get("early_stopping_metric_key", "L")
        )



        artifact_uri = mlflow.get_artifact_uri()
        print("ARTIFACTS URI = ", artifact_uri)
        print("Done with training for: ", config_name)

        evaluate_on_testset = config.get("evaluate_on_testset", False)
        if evaluate_on_testset:
            print("Evaluating on test set")
            evaluate_and_log_testset_results_AE_v2( # TODO maybe add triplet loss
                model_AE=model_AE,
                model_classifier=model_c,
                dataloaders=test_loaders,
                loss_fn_x=loss_fn_x,
                loss_fn_y=loss_fn_y,
                loss_fn_z=loss_fn_z,
                device=device,
                classes=classes,
                transform=transform_AE,
                prefix="test_",
            )

        if config.get("eval_on_val_set_once_more", True):
            print("Evaluating on val set once more")
            evaluate_and_log_testset_results_AE_v2(
                model_AE=model_AE,
                model_classifier=model_c,
                dataloaders=val_loaders,
                loss_fn_x=loss_fn_x,
                loss_fn_y=loss_fn_y,
                loss_fn_z=loss_fn_z,
                device=device,
                classes=classes,
                transform=transform_AE,
                prefix="valFinal_",
                skip_metrics_logging=False # These are already logged in the training loop
            )



if __name__ == "__main__":
    main()
    print("Done")
