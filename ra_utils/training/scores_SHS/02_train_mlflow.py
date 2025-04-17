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
    make_params_a_la_Paul
)

from ra_utils.training.scores_SHS.scores_SHS_training_lib import (
    train_loop,
    evaluate_and_log_testset_results
)

# --------------------------------------------------------------#
# --------------------------  foos -----------------------------#
# --------------------------------------------------------------#


# --------------------------------------------------------------#
# -------------------------  model -----------------------------#
# --------------------------------------------------------------#




# --------------------------------------------------------------#
# --------------------------  loss -----------------------------#
# --------------------------------------------------------------#


# --------------------------------------------------------------#
# --------------------------  TODO  ----------------------------#
# --------------------------------------------------------------#
# read model_params from config
# Test more layers, ... 
# 
# Try different loss function 
# 








# --------------------------------------------------------------#
# --------------------------  main -----------------------------#
# --------------------------------------------------------------#
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the configuration
    config = ra_utils.utils.config_parser.load_config(
        default_config="/home/cwatzenboeck/code/RA/ra_utils/runs/config_scoring/old/scoring_H_XX_dev_MultiModal.yml",
        # default_config="/home/cwatzenboeck/code/RA/ra_utils/runs/config_scoring/ERO_H_PIP_SM_ResNet18.yml",        
        debugging_in_jupyter_nb=False, silencium=False)

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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on ", device)

        # Log the parts of the config which are not a dict
        basic_config = {f"{k}": v for k,
                        v in config.items() if not isinstance(v, dict)}
        mlflow.log_params(basic_config)

        # Log the parts of the config which are a dict
        for n in ["data", "transforms", "training", "optimizer_params"]:
            mlflow.log_params({f"{n}.{k}": v for k, v in config[n].items()})

        # Log config file
        mlflow.log_dict(config, "config.yml")

        # Load tables with paths and scores (+ split)
        data_tables = load_img_SHS_patch_data(config["data"])

        # Make dataset and dataloaders
        data = dataset_and_loader(data_tables, config)

        # Load/ make model
        # model_params = ra_utils.utils.optuna.update_dot_dicts_with_sub_dicts(config["model_params"])

        model_params = ra_utils.utils.utils.unflatten_dict(config["model_params"])
        model = get_model_for_SHS_scoring(config, model_params)
        model = model.to(device)

        # optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), **config["optimizer_params"])

        # scheduler
        scheduler = None
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

        # For now
        model_name = config["model_name"]
        if model_name == "AutoscorRA":
            params = make_params_a_la_Paul({'data': {"scores": config["data"]["scores"]},
                                            "model_params": model_params})
            classes = params["classes"]
        else:
            classes = get_classes(config)

        # define loss function
        # loss_fn_no_reduce = get_loss_no_reduction(config)
        criterion = nn.CrossEntropyLoss()

        train_dataloader = data["train_loader"]
        val_loader = data["val_loader"]
        test_loader = data["test_loader"]
        epochs = config["training"]["epochs"]
        model_forward_interface_option = config.get(
            "model_forward_interface_option", "image only")

        model = train_loop(model=model,
                           train_dataloader=train_dataloader,
                           val_loader=val_loader,
                           criterion=criterion,
                           optimizer=optimizer,
                           device=device,
                           epochs=epochs,
                           patience=config["training"].get(
                               "early_stopping_tol", 100),
                           scheduler=scheduler,
                           run_full_epochs=False,
                           classes=classes,
                           log_model=config["SAVE_MODEL"],
                           interface_option=model_forward_interface_option
                           )

        artifact_uri = mlflow.get_artifact_uri()
        print("ARTIFACTS URI = ", artifact_uri)

        evaluate_on_testset = config.get("evaluate_on_testset", False)
        if evaluate_on_testset:
            print("Evaluating on test set")
            _, _ = evaluate_and_log_testset_results(
                model=model,
                dataloader=test_loader,
                criterion=criterion,
                device=device,
                classes=classes,
                interface_option=model_forward_interface_option,
                prefix="test_"
            )




if __name__ == "__main__":
    main()
    print("Done")
