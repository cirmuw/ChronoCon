import os
import sys

from torchvision import transforms
from torch.utils.data import DataLoader
import torch


import os
import numpy as np
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import random
from pathlib import Path 


# from ra_utils.autoscora.autoscorRA_Pipeline.scoring.src.io_scoring_method import io_scoring
# from ra_utils.autoscora.autoscorRA_Pipeline.scoring.src.run_utils import (
#     paths_list_scores_list_from_score_types,
#     restructure_paths_and_scores,
#     restructure_paths_and_scores_v2
# )

import pandas as pd
from typing import List 

import ra_utils
import ra_utils.utils.config_parser

import ra_utils.data.dataloader_CR_patches
from ra_utils.data.dataloader_CR_patches import (
    load_img_SHS_patch_data,
    dataset_and_loader,
    df_scores_to_dct_list
)

import ra_utils.data.dataloader_CR_patches
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose
)
import torch

import ra_utils.utils.utils_mlflow
import mlflow
import mlflow.pytorch
import ra_utils.utils.optuna
import ra_utils.utils.utils_torch

from ra_utils.autoscora.autoscorRA_Pipeline.scoring.src.train import train, validate, test

import ra_utils.networks.architecture
from ra_utils.networks.architecture import (
    ResNet18Encoder,
    ResNet34Encoder,
    ResNet50Encoder,
    make_mlp,
    EncoderClassifierNetwork
)
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
import mlflow

#--------------------------------------------------------------#
#--------------------------  foos -----------------------------#
#--------------------------------------------------------------#


#--------------------------------------------------------------#
#-------------------------  model -----------------------------#
#--------------------------------------------------------------#


def get_classes(config):
    # TODO 
    print("HACK Assuming Hand erosion problem -> 6 classes")
    return np.array([0., 1., 2., 3., 4., 5.])

def make_params_a_la_Paul(config):
    params = {"chosen_score": config["data"]["scores"]}
    params = {**params, **config["model_params"]}
    
    if params["binary"] == 0:
        #params["n_classes"] = len(unique_tmp)
        #params["classes"] = unique_tmp
        params["n_classes"] = 6
        params["classes"] = np.array([0., 1., 2., 3., 4., 5.])
        if params["chosen_score"] == ["LunatE", "RadiusE", "ScaphE", "TrapE", "UlnaE"]:
            params["n_classes"] = 26
            params["classes"] = np.arange(26.0)
        if params["chosen_score"] == ["Rad_Carp", "Sca_Cap", "Tra_Sca"]:
            params["n_classes"] = 16
            params["classes"] = np.arange(16.0)
    elif params["binary"] == 1:
        params["n_classes"] = 2
        params["classes"] = np.array([0., 1.])
    elif params["binary"] == 2:
        params["n_classes"] = 2
        params["classes"] = np.array([0., 1.])
    else:
        print("please define \"binary\" ")
    return params


from ra_utils.autoscora.autoscorRA_Pipeline.scoring.src.network import Custom_VGG
def get_model_for_SHS_scoring(config, model_params):
    model_name = config["model_name"]
    if model_name == "AutoscorRA":
        params = make_params_a_la_Paul({'data': {"scores": config["data"]["scores"]},
                                        "model_params": model_params})

        model = Custom_VGG(ipt_size=(128, 128),
                            pretrained=True,
                            num_classes=params["n_classes"],
                            vgg_type = params["vgg_type"],
                            regression = params["regression"],
                            ordinal = params["ordinal"])
        return model
    if model_name == "ResNet18":
        out_dim = model_params["N_classes"]
        encoder = ResNet18Encoder(weights='ResNet18_Weights.DEFAULT')
        mlp = make_mlp(latent_dim=512, depth=2, 
                       dropout_op = None,  # TODO read in  
                       out_dim=out_dim)
        model = EncoderClassifierNetwork(
            encoder=encoder,
            classifier=mlp,
            return_latent_representation=False, 
            preprocessor=None
        )
        return model 
    
    else:
        raise ValueError(f"Model {model_name} not implemented yet.")

#--------------------------------------------------------------#
#--------------------------  loss -----------------------------#
#--------------------------------------------------------------#



class GeneralizedCrossEntropyLoss(nn.Module): 
    def __init__(self, lam = 0):
        super(GeneralizedCrossEntropyLoss, self).__init__()
        self.lam = lam    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.cross_entropy(pred, target, reduction="none")
        if self.lam > 1.0e-8:
            w = torch.abs(torch.argmax(pred, dim=1) - target) + 1
            loss = loss*(w** self.lam)
        return loss



def get_loss_no_reduction(config):
    if config["loss_fn_params"]["name"] == "GeneralizedCrossEntropyLoss":
        lam = config["loss_fn_params"]["lam"]
        def loss_fn_no_reduce(pred, target, lam = lam): 
            loss_all = F.cross_entropy(pred, target, reduction="none")
            if lam > 1.0e-8:
                w = torch.abs(torch.argmax(pred, dim=1) - target) + 1
                loss_all = loss_all*(w** lam)
            return loss_all
        
    elif config["loss_fn_params"]["name"] == "CrossEntropyLoss":
        def loss_fn_no_reduce(pred, target): 
            loss_all = F.cross_entropy(pred, target, reduction="none")
            return loss_all


    elif config["loss_fn_params"]["name"] == "MSELoss":
        def loss_fn_no_reduce(pred, target): 
            loss_all = F.mse_loss(pred, target, reduction="none")
            return loss_all

    else: 
        raise ValueError(f"Loss function {config['loss_fn_params']['name']} not implemented yet.")
    return loss_fn_no_reduce

from ra_utils.autoscora.autoscorRA_Pipeline.scoring.src.train import get_individual_loss_v2


def train_epoch(model, 
               dataloader, 
               optimizer, 
               criterion, 
               device="cpu"
               ):
    running_loss = 0
    model.train()
    for i, batch in enumerate(dataloader):
        X = batch["img"].to(device)
        Y = batch["score"].to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def val_epoch(model, 
               dataloader, 
               criterion, 
               device="cpu", 
               classes = None
               ):
    running_loss = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            X = batch["img"].to(device)
            Y = batch["score"].to(device)
            outputs = model(X)
            loss = criterion(outputs, Y)
            running_loss += loss.item()
            
            # TODO score metrics: 
            # confusion matrix, 
            # F1 score, ...
            
    loss = running_loss / len(dataloader)
    metrics = {"loss": loss}
    # TODO add class specific metrics
    
    return metrics


# OLD!!!!
def train_step(model,
               optimizer,
               data_train_x,
               data_train_y,
               classes,
               loss_fn_no_reduce=lambda pred, target: F.cross_entropy(pred, target, reduction="none")
               ): 
    
    # map the classes to the indices    
    for i in range(len(classes)):
        if classes[i] != i:
            data_train_y[data_train_y == classes[i]] = i

    model.train()
    optimizer.zero_grad()
    pred = model(data_train_x)
    loss_all = loss_fn_no_reduce(pred, data_train_y)
    loss = torch.mean(loss_all)
    loss.backward()
    optimizer.step()
    cur_loss_batch, count_values_batch = get_individual_loss_v2(unique_classes = classes,
                                                classes = data_train_y,
                                                values = loss_all,
                                                device=data_train_x.device)
    return loss, cur_loss_batch, count_values_batch


    
def train_loop(
        model,
        train_dataloader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs=1000,
        patience=10,
        scheduler=None,
        run_full_epochs=False,
        classes = None,
        log_model = False
    ):
        
    best_loss = float('inf')
    best_model = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    for epoch in range(epochs):
        train_loss = train_epoch(model, 
                           train_dataloader, 
                           optimizer,
                           criterion,
                           device=device)
        
        # Validation: 
        val_metrics = val_epoch(model=model, 
                                dataloader=val_loader,
                                criterion=criterion,
                                device=device,
                                classes=classes)
        val_loss = val_metrics["loss"]
        val_metrics = {f"val_{k}": v for k,v in val_metrics.items()}
        
        print(f"Epoch {epoch}/{epochs}, Tr Loss: {train_loss:.4f} |  {val_loss:.4f}")

        # ---- Log metrics to MLflow ----
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        #mlflow.log_metrics(val_metrics, step=epoch)


        # ---- Update scheduler (if provided) ----
        if scheduler is not None:
            scheduler.step(val_loss)
      
        # ---- If we are not forcing full epochs, do early stopping checks ----
        if not run_full_epochs:
            if val_loss < best_loss:
                best_val_mpe = val_loss
                best_model = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0

                # Log these weights as the best so far
                if log_model: 
                    mlflow.pytorch.log_model(model, artifact_path="best_model")
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
        model.load_state_dict(best_model)

    return model





#--------------------------------------------------------------#
#--------------------------  main -----------------------------#
#--------------------------------------------------------------#
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the configuration
    config = ra_utils.utils.config_parser.load_config(
    default_config="/home/cwatzenboeck/code/RA/ra_utils/runs/config_scoring/scoring_H_XX_dev_resnet.yml", 
    debugging_in_jupyter_nb=True, silencium=True)



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

        # Log some parameters from config
        mlflow.log_param("experiment_name", config["experiment_name"])
        mlflow.log_param("run_name", config["run_name"])

        mlflow.log_params(config["data"])
        mlflow.log_params(config["transforms"])
        mlflow.log_params(config["training"])
        # TODO     
    
        # Log config file
        mlflow.log_dict(config, "config.yml")




        # Load tables with paths and scores (+ split)
        data_tables = load_img_SHS_patch_data(config["data"])

        # Make dataset and dataloaders
        data = dataset_and_loader(data_tables, config)


        # Load/ make model
        model_params = ra_utils.utils.optuna.update_dot_dicts_with_sub_dicts(config["model_params"])
        model = get_model_for_SHS_scoring(config, model_params)
        model = model.to(device)


        # optimizer
        optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer_params"])

        # scheduler
        scheduler = None

        # For now
        model_name = config["model_name"]
        if model_name == "AutoscorRA":
            params = make_params_a_la_Paul({'data': {"scores": config["data"]["scores"]},
                                            "model_params": model_params})
            classes = params["classes"]
        else: 
            classes = get_classes(config)
        
        
        # define loss function
        #loss_fn_no_reduce = get_loss_no_reduction(config)
        criterion = nn.CrossEntropyLoss()

        train_dataloader = data["train_loader"]
        val_loader = data["val_loader"] 
        epochs = config["training"]["epochs"]
        
        model =  train_loop(model = model, 
                            train_dataloader = train_dataloader,
                            val_loader = val_loader,
                            criterion = criterion, 
                            optimizer = optimizer,
                            device = device,
                            epochs=epochs,
                            patience=config["training"].get("early_stopping_tol", 100),
                            scheduler=scheduler,
                            run_full_epochs=False,
                            classes = classes
                            )
        








        
            
            # loss_epoch = 0
            # for i, batch in enumerate(train_dataloader):
            #     data_train_x = batch["img"]
            #     data_train_y = batch["score"]
            #     data_train_x = data_train_x.to(device)
            #     data_train_y = data_train_y.to(device)

            #     loss_mean_batch, cur_loss_batch, count_values_batch = train_step(
            #         model=model,
            #         optimizer=optimizer,
            #         data_train_x=data_train_x,
            #         data_train_y=data_train_y,
            #         loss_fn_no_reduce=loss_fn_no_reduce, 
            #         classes=classes
            #     )
            #     loss_epoch += loss_mean_batch.item()
            #     #print(f"     Batch {i}, Loss: {loss_mean_batch.item()}")
            # loss_epoch /= len(train_dataloader)
            # print(f"Epoch {epoch}, Loss: {loss_epoch:.4f}")


    








        # optimizer_class = globals()[config['optimizer_name']]
        # scheduler_class = globals()[config['scheduler_name']] if config['scheduler_name'] else None


        # scheduler_params = config["scheduler_params"]
        # optimizer_params = config["optimizer_params"]
        # optimizer, scheduler = ra_utils.utils.utils_torch.plan_optimization_v3(
        #     model, optimizer_class=optimizer_class, optimizer_params=optimizer_params,
        #     scheduler_class=scheduler_class, scheduler_params=scheduler_params
        # )



if __name__ == "__main__":
    main()
    print("Done")