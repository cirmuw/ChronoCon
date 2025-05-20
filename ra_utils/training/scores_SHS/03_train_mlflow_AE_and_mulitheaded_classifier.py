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
    load_img_SHS_patch_data,
    dataset_and_loader
)

from ra_utils.networks.assemble_model import (
    get_model_for_SHS_scoring,
    get_model_for_SHS_scoring_with_imports,
    make_params_a_la_Paul
)

from ra_utils.training.scores_SHS.scores_SHS_training_lib import (
#    train_loop, # no longer needed
    evaluate_and_log_testset_results
)

from ra_utils.training.scores_SHS.scores_SHS_training_lib_AE_v1 import (
    training_epoch_AE_v1, 
    val_epoch_AE_v1, 
    train_loop_AE_v1,
    evaluate_and_log_testset_results_AE_v1,
)

import ra_utils.networks.loss_function
from ra_utils.networks.loss_function import get_score_loss_function


from ra_utils.progressionlearning.models.builder import (
    build_MTANAE
)

import torchvision.transforms.v2 as v2
import ra_utils.utils.config_parser


from ra_utils.training.scores_SHS.model_builders import build_models_AE




# --------------------------------------------------------------#
# --------------------------  main -----------------------------#
# --------------------------------------------------------------#
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the configuration
    config, config_name = ra_utils.utils.config_parser.load_config(
        default_config="/home/cwatzenboeck/code/RA/ra_utils/runs/config_scoring/Exp03_ResNetAE_MLP.yml",   
        # default_config="/home/cwatzenboeck/code/RA/ra_utils/runs/config_scoring/ERO_H_PIP_SM_ResNet18.yml",        
        debugging_in_jupyter_nb=False, silencium=False, return_config_name=True)
    
    classifier_head_infos = config["data"]["classifier_head_infos"]
    

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

        # Log the parts of the config which are not a dict
        # basic_config = {f"{k}": v for k, v in config.items() if not isinstance(v, dict)}
        # mlflow.log_params(basic_config)
        mlflow.log_params(ra_utils.utils.utils.flatten_dict(config))

        # Log the parts of the config which are a dict
        # for n in ["data", "transforms", "training", "optimizer_params"]:
        #     mlflow.log_params({f"{n}.{k}": v for k, v in config[n].items()})

        # Log config file
        mlflow.log_dict(config, "config.yml")

        # Load tables with paths and scores (+ split)
        data_tables = load_img_SHS_patch_data(config["data"])

        # Make dataset and dataloaders
        data = dataset_and_loader(data_tables, config)

        # Load/ make model
        model_name = config["model_name"]
        model_AE, model_c = build_models_AE(model_name, config, classifier_head_infos)

        model_AE.to(device)
        model_c.to(device)

        # define loss function
        loss_fn_y = get_score_loss_function(config["loss"]["score"])
        loss_fn_x = nn.MSELoss()
        loss_fn_z = nn.L1Loss()

        # ---- joint optimiser with separate lrs --------------------------------
        if model_name in ["ResNetAE + MultiHeadClassifier", 
                          "ResNetAE_v2 + MultiHeadClassifier", 
                          "ResNetAE_v2p1 + MultiHeadClassifier"]: 
            opt_cfg = config["optimizer_params"].copy()
            lr_e  = opt_cfg["learning_rates"]["encoder"]
            lr_d  = opt_cfg["learning_rates"]["decoder"]
            lr_clf = opt_cfg["learning_rates"]["classifier"]
            param_groups = [
                {"params": model_AE.encoder.parameters(), "lr": lr_e},
                {"params": model_AE.decoder.parameters(), "lr": lr_d},
                {"params": model_c.parameters(),  "lr": lr_clf},
            ]
            print(opt_cfg["other_optimizer_kwargs"])
            optimizer = torch.optim.AdamW(param_groups, **opt_cfg["other_optimizer_kwargs"])
        else:
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

        # model_params = ra_utils.utils.utils.unflatten_dict(config["model_params"])
        # model = get_model_for_SHS_scoring(config, model_params)
        # model = get_model_for_SHS_scoring_with_imports(config)
        # model = model.to(device)

        # scheduler
        scheduler = None
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)



        
        

        train_dataloader = data["train_loader"]
        val_loader = data["val_loader"]
        test_loader = data["test_loader"]
        epochs = config["training"]["epochs"]
        model_forward_interface_option = config.get(
            "model_forward_interface_option", "image only")


        classes = get_classes(config) # TO be removed

        print("Start training for: ", config_name)
        model_AE, model_c = train_loop_AE_v1(
            model_AE=model_AE,
            model_classifier=model_c,
            train_dataloader=train_dataloader,
            val_loader=val_loader,
            loss_fn_x=loss_fn_x,
            loss_fn_y=loss_fn_y,
            loss_fn_z=loss_fn_z,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=epochs,
            patience=config["training"].get("early_stopping_tol", 100),
            lambda_x=config.get('loss_weights', {}).get('lambda_x', 1.0), 
            lambda_y=config.get('loss_weights', {}).get('lambda_y', 1.0), 
            lambda_z=config.get('loss_weights', {}).get('lambda_z', 1.0), 
            transform=transform_AE,
            classes=classes,
            log_model=config["SAVE_MODEL"],
            verbose=2,
            ES_metric_key=config["training"].get("early_stopping_metric_key", "L")
        )



        artifact_uri = mlflow.get_artifact_uri()
        print("ARTIFACTS URI = ", artifact_uri)
        print("Done with training for: ", config_name)

        evaluate_on_testset = config.get("evaluate_on_testset", False)
        if evaluate_on_testset:
            print("Evaluating on test set")
            evaluate_and_log_testset_results_AE_v1(
                model_AE=model_AE,
                model_classifier=model_c,
                dataloader=test_loader,
                loss_fn_x=loss_fn_x,
                loss_fn_y=loss_fn_y,
                loss_fn_z=loss_fn_z,
                device=device,
                classes=classes,
                transform=transform_AE,
                prefix="test_",
            )



if __name__ == "__main__":
    main()
    print("Done")
