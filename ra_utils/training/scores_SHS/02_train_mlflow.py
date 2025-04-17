from ra_utils.autoscora.autoscorRA_Pipeline.scoring.src.train import get_individual_loss_v2
from ra_utils.autoscora.autoscorRA_Pipeline.scoring.src.network import Custom_VGG
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
from tqdm import tqdm


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
    EncoderClassifierNetwork,
    MultiModalImageScoreTypeNetwork,
    ROI_type_encoder
)

import torch.nn as nn
import torch.nn.functional as F
import copy
import os
import mlflow
import tempfile


import yaml
from importlib import resources
import torch

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import mlflow
import json
import os
import tempfile
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from typing import Literal


# --------------------------------------------------------------#
# --------------------------  foos -----------------------------#
# --------------------------------------------------------------#


# --------------------------------------------------------------#
# -------------------------  model -----------------------------#
# --------------------------------------------------------------#


def get_classes(config):
    with resources.files("ra_utils.resources.scores_metadata").joinpath("score_abbreviations_info_dct_w_scores.yml").open("r") as f:
        score_abbreviations_info_dct = yaml.safe_load(f)
    chosen_score = config["data"]["scores"] 
    sum_wrist_points = config["data"]["sum_wrist_points"]
    if (chosen_score == ["LunatE", "RadiusE", "ScaphE", "TrapE", "UlnaE"]) and sum_wrist_points:
        print("HACK Assuming LunatE, RadiusE, ScaphE, TrapE, UlnaE -> 5*5+1 classes")
        return np.array([str(i) for i in range(26)])
    elif (chosen_score == ["Rad_Carp", "Sca_Cap", "Tra_Sca"]) and sum_wrist_points:
        print("HACK Assuming Rad_Carp, Sca_Cap, Tra_Sca -> 4*3+1 classes")
        return np.array([str(i) for i in range(13)])
    else:
        score_ranges = []
        first = True
        for score in chosen_score:
            if first:
                first = False
                score_ranges = score_abbreviations_info_dct[score]["scores"]
            else:
                assert score_ranges == score_abbreviations_info_dct[score][
                    "scores"], f"Score ranges do not match for {score} and {chosen_score[0]}"
        return score_ranges


def make_params_a_la_Paul(config):
    params = {"chosen_score": config["data"]["scores"]}
    params = {**params, **config["model_params"]}

    if params["binary"] == 0:
        # params["n_classes"] = len(unique_tmp)
        # params["classes"] = unique_tmp
        params["n_classes"] = 6
        params["classes"] = np.array([0., 1., 2., 3., 4., 5.])
        if params["chosen_score"] == ["LunatE", "RadiusE", "ScaphE", "TrapE", "UlnaE"]:
            params["n_classes"] = 26
            params["classes"] = np.arange(26.0)
        # CW: 16 Makes no sense to me (ERO 0, ... 4)
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


# def check_score_types_can_be_combined(config):
#     # Load data classes:
#     with resources.files("ra_utils.resources.scores_metadata").joinpath("score_abbreviations_info_dct.yml").open("r") as f:
#         score_abbreviations_info_dct = yaml.safe_load(f)
#     df = pd.DataFrame(score_abbreviations_info_dct).transpose()


#     m = (df["extremity"] == "H")  & (df["score_type"] == "ERO")
#     classes_ERO_H = list(df[m].index)

#     m = (df["extremity"] == "F")  & (df["score_type"] == "ERO")
#     classes_ERO_F = list(df[m].index)

#     m = (df["extremity"] == "H")  & (df["score_type"] == "JSN")
#     classes_JSN_H = list(df[m].index)

#     m = (df["extremity"] == "F")  & (df["score_type"] == "JSN")
#     classes_JSN_F = list(df[m].index)

#     # differentiate between classes according to number of output classes (scores)
#     classes_JSN = classes_JSN_F + classes_JSN_H   # 0, ..., 4 as output

#     classes = config["data"]["scores"]
#     # Check that all belong to same output number of classes:
#     if len(set(classes) & set(classes_JSN)) != 0:
#         assert len(set(classes) & set(classes_ERO_F)) == 0, f"ERO_F and JSN classes overlap: {set(classes) & set(classes_ERO_F)}"
#         assert len(set(classes) & set(classes_ERO_H)) == 0, f"ERO_H and JSN classes overlap: {set(classes) & set(classes_ERO_H)}"
#         score_lenght = 6
#     elif len(set(classes) & set(classes_ERO_F)) != 0:
#         assert len(set(classes) & set(classes_JSN)) == 0, f"JSN and ERO_F classes overlap: {set(classes) & set(classes_JSN)}"
#         assert len(set(classes) & set(classes_ERO_H)) == 0, f"ERO_H and ERO_F classes overlap: {set(classes) & set(classes_ERO_H)}"
#     elif len(set(classes) & set(classes_ERO_H)) != 0:
#         assert len(set(classes) & set(classes_JSN)) == 0, f"JSN and ERO_H classes overlap: {set(classes) & set(classes_JSN)}"
#         assert len(set(classes) & set(classes_ERO_F)) == 0, f"ERO_F and ERO_H classes overlap: {set(classes) & set(classes_ERO_F)}"
#     else:
#         raise ValueError(f"Classes {classes} do not match any of the known classes.")
#     return None


def get_model_for_SHS_scoring(config, model_params):
    model_name = config["model_name"]
    if model_name == "AutoscorRA":
        params = make_params_a_la_Paul({'data': {"scores": config["data"]["scores"]},
                                        "model_params": model_params})

        model = Custom_VGG(ipt_size=(128, 128),
                           pretrained=True,
                           num_classes=params["n_classes"],
                           vgg_type=params["vgg_type"],
                           regression=params["regression"],
                           ordinal=params["ordinal"])
        return model
    if model_name == "ResNet18":
        out_dim = model_params["N_classes"]
        encoder = ResNet18Encoder(weights='ResNet18_Weights.DEFAULT')
        mlp = make_mlp(latent_dim=512, depth=2,
                       dropout_op=None,  # TODO read in
                       out_dim=out_dim)
        model = EncoderClassifierNetwork(
            encoder=encoder,
            classifier=mlp,
            return_latent_representation=False,
            preprocessor=None
        )
        return model

    if model_name == "ResNet34":
        out_dim = model_params["N_classes"]
        encoder = ResNet34Encoder(weights='ResNet34_Weights.DEFAULT')
        mlp = make_mlp(latent_dim=512, depth=2,
                       dropout_op=None,  # TODO read in
                       out_dim=out_dim)
        model = EncoderClassifierNetwork(
            encoder=encoder,
            classifier=mlp,
            return_latent_representation=False,
            preprocessor=None
        )
        return model

    if model_name == "ResNet50":
        out_dim = model_params["N_classes"]
        encoder = ResNet50Encoder(weights='ResNet50_Weights.DEFAULT')
        mlp = make_mlp(latent_dim=512, depth=2,
                       dropout_op=None,  # TODO read in
                       out_dim=out_dim)
        model = EncoderClassifierNetwork(
            encoder=encoder,
            classifier=mlp,
            return_latent_representation=False,
            preprocessor=None
        )
        return model

    if model_name == "MultiModalImageScoreTypeNetwork__ResNet18":
        # check_score_types_can_be_combined(config)
        scores_names = get_classes(config)
        n_classes = len(scores_names)
        score_types = config["data"]["scores"]

        encoder = ResNet18Encoder(weights='ResNet18_Weights.DEFAULT')
        encoder_tab = ROI_type_encoder(
            score_types, out_dim=None, normalized=False)
        latent_dim_tab = encoder_tab.output_dim

        classifier = make_mlp(latent_dim=512 + latent_dim_tab,
                              depth=2,
                              dropout_op=None,
                              out_dim=n_classes)

        model_multimodal = MultiModalImageScoreTypeNetwork(
            image_encoder=encoder,
            score_type_encoder=encoder_tab,
            classifier=classifier,
            return_latent_representation=False
        )
        return model_multimodal

    if model_name == "MultiModalImageScoreTypeNetwork__ResNet34":
        # check_score_types_can_be_combined(config)
        scores_names = get_classes(config)
        n_classes = len(scores_names)
        score_types = config["data"]["scores"]

        encoder = ResNet34Encoder(weights='ResNet34_Weights.DEFAULT')
        encoder_tab = ROI_type_encoder(
            score_types, out_dim=None, normalized=False)
        latent_dim_tab = encoder_tab.output_dim

        classifier = make_mlp(latent_dim=512 + latent_dim_tab,
                              depth=2,
                              dropout_op=None,
                              out_dim=n_classes)

        model_multimodal = MultiModalImageScoreTypeNetwork(
            image_encoder=encoder,
            score_type_encoder=encoder_tab,
            classifier=classifier,
            return_latent_representation=False
        )
        return model_multimodal

 
    if model_name == "MultiModalImageScoreTypeNetwork__ResNet50":
        # check_score_types_can_be_combined(config)
        scores_names = get_classes(config)
        n_classes = len(scores_names)
        score_types = config["data"]["scores"]

        encoder = ResNet50Encoder(weights='ResNet50_Weights.DEFAULT')
        encoder_tab = ROI_type_encoder(
            score_types, out_dim=None, normalized=False)
        latent_dim_tab = encoder_tab.output_dim

        classifier = make_mlp(latent_dim=512 + latent_dim_tab,
                              depth=2,
                              dropout_op=None,
                              out_dim=n_classes)

        model_multimodal = MultiModalImageScoreTypeNetwork(
            image_encoder=encoder,
            score_type_encoder=encoder_tab,
            classifier=classifier,
            return_latent_representation=False
        )
        return model_multimodal

    else:
        raise ValueError(f"Model {model_name} not implemented yet.")





# --------------------------------------------------------------#
# --------------------------  loss -----------------------------#
# --------------------------------------------------------------#


class GeneralizedCrossEntropyLoss(nn.Module):
    def __init__(self, lam=0):
        super(GeneralizedCrossEntropyLoss, self).__init__()
        self.lam = lam

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.cross_entropy(pred, target, reduction="none")
        if self.lam > 1.0e-8:
            w = torch.abs(torch.argmax(pred, dim=1) - target) + 1
            loss = loss*(w ** self.lam)
        return loss


def get_loss_no_reduction(config):
    if config["loss_fn_params"]["name"] == "GeneralizedCrossEntropyLoss":
        lam = config["loss_fn_params"]["lam"]

        def loss_fn_no_reduce(pred, target, lam=lam):
            loss_all = F.cross_entropy(pred, target, reduction="none")
            if lam > 1.0e-8:
                w = torch.abs(torch.argmax(pred, dim=1) - target) + 1
                loss_all = loss_all*(w ** lam)
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
        raise ValueError(
            f"Loss function {config['loss_fn_params']['name']} not implemented yet.")
    return loss_fn_no_reduce


def model_interface_forward(model: nn.Module, batch: dict, device="cpu",
                            options: Literal["image only", "image + score_type"] = "image only"):
    if options == "image only":
        X = batch["img"].to(device)
        return model(X)

    elif options == "image + score_type":
        X = batch["img"].to(device)
        score_types = batch["score_type"]   # List of strings (N_batch)
        return model(images=X, score_types=score_types)


def train_epoch(model,
                dataloader,
                optimizer,
                criterion,
                device="cpu",
                interface_option="image only"
                ):
    running_loss = 0
    model.train()
    for i, batch in tqdm(enumerate(dataloader), desc="train_epoch :: batch"):
        Y = batch["score"].to(device)
        optimizer.zero_grad()
        outputs = model_interface_forward(
            model, batch, device, options=interface_option)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

from sklearn.metrics import balanced_accuracy_score

def calculate_some_classification_metrics(all_preds, all_labels):
    metrics = {}
    mse = np.mean((all_preds - all_labels) ** 2)
    mae = np.mean(np.abs(all_preds - all_labels))
    rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
    metrics["rmse"] = rmse
    metrics["mse"] = mse
    metrics["mae"] = mae



    # 1. Accuracy: Percentage of predictions that exactly match the true labels.
    accuracy = np.mean(all_preds == all_labels)
    metrics["accuracy"] = accuracy

    # 2. Error > 1: Percentage of predictions where the absolute error is greater than 1.
    accuracy_lt_2 = np.mean(np.abs(all_preds - all_labels) < 2)
    metrics["accuracy (error < 2)"] = accuracy_lt_2

    # 3. Balanced Accuracy: Using scikit-learn's balanced_accuracy_score.
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    metrics["balanced acc."] = balanced_acc

    # 4. Balanced (error < 2): For each unique class, calculate the percentage of predictions with an error less than 2 and average.
    unique_labels = np.unique(all_labels)
    balanced_error_less_2 = np.mean([
        np.mean(np.abs(all_preds[all_labels == label] - label) < 2)
        for label in unique_labels
    ])
    metrics["balanced acc. (error < 2)"] = balanced_error_less_2

    return metrics



def val_epoch(model,
              dataloader,
              criterion,
              device="cpu",
              classes=None,
              interface_option="image only",
              return_all_predictions=False):

    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_logits = []
    extra_keys = ['file_name', 'score_type', 'JSN_or_ERO', 'extremity', 'patient_id']
    all_extras = {k: [] for k in extra_keys}

    model.eval()
    n_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            # Get inputs and targets
            Y = batch["score"].to(device)

            # Forward pass
            logits = model_interface_forward(
                model, batch, device, options=interface_option)
            loss = criterion(logits, Y)
            running_loss += loss.item() * len(Y)
            n_samples += len(Y)

            # Convert logits to predicted class indices
            preds = logits.argmax(dim=1)

            # Save extra information
            for key in extra_keys:
                if key in batch:
                    all_extras[key].extend(batch[key])
            
            # Save predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(Y.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())

    # Average loss over the validation set.
    avg_loss = running_loss / n_samples
    metrics = {"loss": avg_loss}

    # Convert to numpy arrays.
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    all_probs = F.softmax(torch.from_numpy(all_logits), dim=1).numpy()
    
    for key in extra_keys:
        all_extras[key] = np.array(all_extras[key])
    
    
    m = calculate_some_classification_metrics(all_preds, all_labels)
    metrics = {**metrics, **m}

    top2 = np.argsort(all_probs, axis=1)[:, -2:]
    top2_correct = sum(label in top2[idx] for idx, label in enumerate(all_labels))
    metrics['top2_accuracy'] = top2_correct / len(all_labels)        


    # Compute confusion matrix.
    cm = confusion_matrix(all_labels, all_preds)
    # convert to list for JSON/MLflow logging
    metrics["confusion_matrix"] = cm.tolist()

    # If classes is provided, supply full label list to ensure all classes are included.
    if classes is not None:
        # Assumes classes are indexed 0, 1, ..., len(classes)-1
        labels = list(range(len(classes)))
        report = classification_report(
            all_labels,
            all_preds,
            labels=labels,
            target_names=classes,
            output_dict=True,
            zero_division=0.0
        )
    else:
        report = classification_report(
            all_labels, all_preds, output_dict=True, zero_division=0.0)
    metrics["classification_report"] = report

    outputs_all_samples = {}
    if return_all_predictions:
        outputs_all_samples["labels"] = all_labels
        outputs_all_samples["preds"] = all_preds
        #outputs_all_samples["logits"] = all_logits
        outputs_all_samples["probs"] = all_probs
        outputs_all_samples = {**outputs_all_samples, **all_extras}
        return metrics, outputs_all_samples
    else:
        return metrics


def log_metrics_mlflow(metrics: dict, 
                       prefix: str,
                       classes: list, 
                       step: None | int = None,
                       log_report_and_confusion_matrix_as_artifact=False):
    # ---- Log Scalar Test Metrics ----
    mlflow.log_metric(f"{prefix}loss", metrics["loss"], step=step)
    mlflow.log_metric(f"{prefix}accuracy", metrics.get("accuracy", 0.0), step=step)
    mlflow.log_metric(f"{prefix}mse", metrics.get("mse", 9999), step=step)
    mlflow.log_metric(f"{prefix}mae", metrics.get("mae", 9999), step=step)
    mlflow.log_metric(f"{prefix}rmse", metrics.get("rmse", 9999), step=step)
    mlflow.log_metric(f"{prefix}accuracy", metrics.get("accuracy", 0.0), step=step)
    mlflow.log_metric(f"{prefix}accuracy lt_2", metrics.get("accuracy (error < 2)", 0.0), step=step)    
    mlflow.log_metric(f"{prefix}balanced acc.", metrics.get("balanced acc.", 0.0), step=step)
    mlflow.log_metric(f"{prefix}balanced acc. lt_2", metrics.get("balanced acc. (error < 2)", 0.0), step=step)
    mlflow.log_metric(f"{prefix}top2_accuracy", metrics.get("top2_accuracy", 0.0), step=step)
    

    # ---- Log Classification Report & Confusion Matrix as Artifacts ----
    classification_report_dict = metrics.get("classification_report", {})
    confusion_matrix_list = metrics.get("confusion_matrix", {})
    if log_report_and_confusion_matrix_as_artifact: 
        mlflow.log_dict(classification_report_dict, f"metrics/{prefix}classification_report.json")
        mlflow.log_dict(confusion_matrix_list, f"metrics/{prefix}confusion_matrix.json")

    # ---- Log Class-Specific Metrics ----
    if classes is not None:
        for cls in classes:
            cls_metrics = classification_report_dict.get(cls)
            if cls_metrics:
                mlflow.log_metric(f"{prefix}{cls}_precision", cls_metrics["precision"])
                mlflow.log_metric(f"{prefix}{cls}_recall", cls_metrics["recall"])
                mlflow.log_metric(f"{prefix}{cls}_f1", cls_metrics["f1-score"])

    # ---- Log Macro-Average Metrics ----
    macro_avg = classification_report_dict.get("macro avg")
    if macro_avg:
        mlflow.log_metric(f"{prefix}macro_precision", macro_avg["precision"])
        mlflow.log_metric(f"{prefix}macro_recall", macro_avg["recall"])
        mlflow.log_metric(f"{prefix}macro_f1", macro_avg["f1-score"])

    # ---- Log Weighted-Average Metrics ----
    weighted_avg = classification_report_dict.get("weighted avg")
    if weighted_avg:
        mlflow.log_metric(f"{prefix}weighted_precision", weighted_avg["precision"])
        mlflow.log_metric(f"{prefix}weighted_recall", weighted_avg["recall"])
        mlflow.log_metric(f"{prefix}weighted_f1", weighted_avg["f1-score"])

    # ---- Log Per-Class Accuracy Derived from the Confusion Matrix ----
    cm_array = np.array(confusion_matrix_list)
    if classes is not None and cm_array.shape[0] == len(classes):
        for idx, cls in enumerate(classes):
            total = cm_array[idx].sum()
            class_accuracy = float(cm_array[idx, idx]) / total if total > 0 else 0.0
            mlflow.log_metric(f"{prefix}{cls}_accuracy", class_accuracy)



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
    classes=None,
    log_model=False,
    verbose=True,
    interface_option="image only",
    log_classspecific_metrics=False,
):

    best_loss = float('inf')
    best_model = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    for epoch in tqdm(range(epochs), desc="train_loop :: epochs"):
        train_loss = train_epoch(model,
                                 train_dataloader,
                                 optimizer,
                                 criterion,
                                 device=device,
                                 interface_option=interface_option)

        # At the end of each training epoch, after you get your validation metrics:
        val_metrics = val_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            classes=classes,  # list of class names, e.g. ['0', '1', '2']
            interface_option=interface_option,
            return_all_predictions=False
        )
        val_loss = val_metrics["loss"]

        print(f"Epoch {epoch}/{epochs}, Tr Loss: {train_loss:.4f} |  Val Loss: {val_loss:.4f}")
        # ---- Log scalar metrics to MLflow ----
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        
        log_metrics_mlflow(val_metrics, prefix="val_", classes=classes, step=epoch,
                           log_report_and_confusion_matrix_as_artifact=False)
        
        # mlflow.log_metric("val_loss", val_metrics["loss"], step=epoch)
        # mlflow.log_metric("val_accuracy", val_metrics.get("accuracy", 0.0), step=epoch)
        # mlflow.log_metric("val_mse", val_metrics.get("mse", 9999), step=epoch)
        # mlflow.log_metric("val_mae", val_metrics.get("mae", 9999), step=epoch)
        # mlflow.log_metric("val_rmse", val_metrics.get("rmse", 9999), step=epoch)
        # mlflow.log_metric("val_accuracy", val_metrics.get("accuracy", 0.0), step=epoch)
        # mlflow.log_metric("val_accuracy_lt_2", val_metrics.get("accuracy (error < 2)", 0.0), step=epoch)
        # mlflow.log_metric("val_balanced_acc", val_metrics.get("balanced acc.", 0.0), step=epoch)
        # mlflow.log_metric("val_balanced_acc_error_lt_2", val_metrics.get("balanced acc. (error < 2)", 0.0), step=epoch)


        # report = val_metrics["classification_report"]
        # if classes is not None:
        #     for cls in classes:
        #         cls_metrics = report.get(cls)
        #         if cls_metrics:
        #             # Log precision, recall, and F1-score for each class
        #             mlflow.log_metric(
        #                 f"val_{cls}_precision", cls_metrics["precision"], step=epoch)
        #             mlflow.log_metric(
        #                 f"val_{cls}_recall", cls_metrics["recall"], step=epoch)
        #             mlflow.log_metric(
        #                 f"val_{cls}_f1", cls_metrics["f1-score"], step=epoch)

        # # Log macro-average metrics (calculated across all classes)
        # macro_avg = report.get("macro avg")
        # if macro_avg:
        #     mlflow.log_metric("val_macro_precision",
        #                       macro_avg["precision"], step=epoch)
        #     mlflow.log_metric("val_macro_recall",
        #                       macro_avg["recall"], step=epoch)
        #     mlflow.log_metric(
        #         "val_macro_f1", macro_avg["f1-score"], step=epoch)

        # # Log weighted-average metrics (weighted by support)
        # weighted_avg = report.get("weighted avg")
        # if weighted_avg:
        #     mlflow.log_metric("val_weighted_precision",
        #                       weighted_avg["precision"], step=epoch)
        #     mlflow.log_metric("val_weighted_recall",
        #                       weighted_avg["recall"], step=epoch)
        #     mlflow.log_metric("val_weighted_f1",
        #                       weighted_avg["f1-score"], step=epoch)

        # # Log accuracy
        # mlflow.log_metric("val_accuracy", report.get("accuracy"), step=epoch)

        # # ---- Optionally log per-class accuracy (using the confusion matrix) ----
        # # Compute per-class accuracy manually:
        # # for each class, accuracy = (true positives) / (total actual for that class)
        # cm_array = np.array(val_metrics["confusion_matrix"])
        # if classes is not None and cm_array.shape[0] == len(classes):
        #     for idx, cls in enumerate(classes):
        #         total = cm_array[idx].sum()
        #         # Avoid division by zero when there are no samples for this class.
        #         class_accuracy = float(
        #             cm_array[idx, idx]) / total if total > 0 else 0.0
        #         mlflow.log_metric(f"val_{cls}_accuracy",
        #                           class_accuracy, step=epoch)




        # ---- Update scheduler (if provided) ----
        if scheduler is not None:
            scheduler.step(val_loss)

        # ---- If we are not forcing full epochs, do early stopping checks ----
        if not run_full_epochs:
            if val_loss < best_loss:
                if verbose:
                    print(
                        f"  YEAH!! New best validation loss! Improving from {best_loss:.4f} by {((best_loss - val_loss)/best_loss*100):.2f}%.")

                best_loss = val_loss
                best_model = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0

                # Log these weights as the best so far
                if log_model:
                    mlflow.pytorch.log_model(model, artifact_path="best_model")
                    
                # log classification report of best model
                # Using mlflow.log_dict (requires MLflow >=1.18.0) to log the classification report and confusion matrix.
                mlflow.log_dict(val_metrics.get("classification_report", {}), "metrics/classification_report.json")
                mlflow.log_dict(val_metrics.get("confusion_matrix", {}),
                                "metrics/confusion_matrix.json")
                
            else:
                epochs_no_improve += 1

            # If no improvement for 'patience' epochs, stop
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping triggered. No improvement for {patience} epochs.")
                break

        # ---- Otherwise, if forcing full epochs, just keep going until the end ----
        # (No early stopping logic here.)

    # ---- After training loop ----
    if not run_full_epochs:
        # Restore the best model/heatmap weights when early stopping was in use
        model.load_state_dict(best_model)

    return model




def evaluate_and_log_testset_results(
    model,
    dataloader,
    criterion,
    device,
    classes,
    interface_option="image only",
    prefix="test_"
):

    # Evaluate the test set to obtain metrics and predictions.
    test_metrics_and_predictions, test_outputs_all_samples = val_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
        classes=classes,
        interface_option=interface_option,
        return_all_predictions=True
    )

    log_metrics_mlflow(test_metrics_and_predictions, prefix=prefix, classes=classes, step=None,
                       log_report_and_confusion_matrix_as_artifact=True)
    


 

    # ---- Log Raw Predictions and Additional Infos ----
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        np.savez_compressed(tmp.name, **test_outputs_all_samples)
        npz_path = tmp.name
    mlflow.log_artifact(npz_path, artifact_path=f"predictions/{prefix}")
    os.remove(npz_path)
    
    if False: 
        mlflow.log_dict(test_outputs_all_samples, f"predictions/{prefix}raw_predictions.json")


    return test_metrics_and_predictions, test_outputs_all_samples



# --------------------------------------------------------------#
# --------------------------  main -----------------------------#
# --------------------------------------------------------------#
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the configuration
    config = ra_utils.utils.config_parser.load_config(
        default_config="/home/cwatzenboeck/code/RA/ra_utils/runs/config_scoring/old/scoring_H_XX_dev_MultiModal.yml",
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
        model_params = ra_utils.utils.optuna.update_dot_dicts_with_sub_dicts(
            config["model_params"])
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
