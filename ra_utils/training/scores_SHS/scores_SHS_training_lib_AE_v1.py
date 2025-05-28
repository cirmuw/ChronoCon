import os
import copy
import tempfile
from collections import defaultdict
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
)
import mlflow
import matplotlib.pyplot as plt
from ra_utils.training.scores_SHS.scores_SHS_training_lib import (
    calculate_some_classification_metrics,
    log_metrics_mlflow
)
from ra_utils.networks.architecture import (
    DummyReturnZeroLoss,
    make_score_type_2_head_name_dct
)


from typing import Literal
import torchvision
from typing import Callable


# ---------------------------------------------------------------------------
# Training loop (AutoEncoder + multi‑head classifier)
# ---------------------------------------------------------------------------

def _log_scalar_dict(prefix: str, dct: Dict[str, float], step: int | None = None):
    """Utility: log every key/value pair in *dct* to MLflow with *prefix*."""
    for k, v in dct.items():
        mlflow.log_metric(f"{prefix}{k}", v, step=step)


def train_loop_AE_v1(
    model_AE: torch.nn.Module,
    model_classifier: Optional[torch.nn.Module],
    train_dataloader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    *,
    # loss functions
    loss_fn_x: Optional[torch.nn.Module] = None,
    loss_fn_y: Optional[torch.nn.Module] = None,
    loss_fn_z: Optional[torch.nn.Module] = None,
    # optimiser etc.
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cuda",
    # training config
    epochs: int = 1000,
    patience: int = 10,
    run_full_epochs: bool = False,
    lambda_x: float = 1.0,
    lambda_y: float = 1.0,
    lambda_z: float = 1.0,
    transform=lambda x: x,
    classes: Optional[List[str]] = None,
    log_model: bool = False,
    verbose: bool = True,
    ES_metric_key = "Ly"  # which metric to use for early stopping
):
    """Orchestrates the whole training process using *training_epoch_AE_v1* and
    the newly defined *val_epoch_AE_v1*.
    """

    best_val_loss = float("inf")
    best_AE_state = copy.deepcopy(model_AE.state_dict())
    best_clf_state = (
        copy.deepcopy(model_classifier.state_dict())
        if model_classifier is not None
        else None
    )
    epochs_no_improve = 0
    print("Starting training")
    for epoch in range(epochs):
        # ------------------------------------------------------------------
        # 1) Train one epoch -------------------------------------------------
        train_loss_dct = training_epoch_AE_v1(
            model_AE,
            optimizer,
            train_dataloader,
            model_classifier=model_classifier,
            loss_fn_x=loss_fn_x,
            loss_fn_y=loss_fn_y,
            loss_fn_z=loss_fn_z,
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            lambda_z=lambda_z,
            transform=transform,
            device=device,
        )

        # log training scalar losses
        
        _log_scalar_dict("train_", train_loss_dct, step=epoch)

        # ------------------------------------------------------------------
        # 2) Validation ------------------------------------------------------
        val_metrics = val_epoch_AE_v1(
            model_AE,
            val_loader,
            model_classifier=model_classifier,
            loss_fn_x=loss_fn_x,
            loss_fn_y=loss_fn_y,
            loss_fn_z=loss_fn_z,
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            lambda_z=lambda_z,
            transform = lambda x: x, # No denoising in val. loop  #transform,
            device=device,
            classes=classes,
            return_all_predictions=False,
        )
        val_loss = val_metrics["L"]

        # basic console printout -------------------------------------------
        if verbose:
            print(
                (
                    f"Epoch {epoch}/{epochs} | "
                    f"train L: {train_loss_dct['L']:.4f} "
                    f"val L: {val_loss:.4f}"
                )
            )
        if verbose > 1:
            train_losses = "  Train:      " + " | ".join(
            f"{key}: {value:.3f}" for key, value in train_loss_dct.items() if key != "L"
            )
            val_losses = "  Validation: " + " | ".join(
            f"{key}: {value:.3f}" for key, value in val_metrics.items() if (
                key.startswith("L") and 
                key != "L")
            )
            print(train_losses)
            print(val_losses)

        # log validation classification metrics with legacy helper ----------
        log_metrics_mlflow(
            val_metrics,
            prefix="val_",
            classes=classes,
            step=epoch,
            log_report_and_confusion_matrix_as_artifact=False,
        )

        # also log reconstruction / latent / per‑head losses
        extra_loss_keys = {
            k: v for k, v in val_metrics.items() if k.startswith("L")
        }
        _log_scalar_dict("val_", extra_loss_keys, step=epoch)

        # ------------------------------------------------------------------
        # 3) Scheduler step --------------------------------------------------
        if scheduler is not None:
            scheduler.step(val_loss)

        # ------------------------------------------------------------------
        # 4) Early stopping --------------------------------------------------
        val_loss_ES = val_metrics[ES_metric_key]  # used to be val_loss
        if not run_full_epochs:
            improved = val_loss_ES < best_val_loss
            if improved:
                if verbose:
                    improvement = (-(val_loss_ES - best_val_loss) / best_val_loss)*100
                    print(
                        f"    YEAH!! New best validation {ES_metric_key} ↓ "
                        f"{best_val_loss:.4f} → {val_loss_ES:.4f}  this is a {improvement:.1f}% improvement \n"
                    )
                best_val_loss = val_loss_ES
                best_AE_state = copy.deepcopy(model_AE.state_dict())
                if model_classifier is not None:
                    best_clf_state = copy.deepcopy(model_classifier.state_dict())
                epochs_no_improve = 0

                # optional MLflow model snapshot
                if log_model:
                    mlflow.pytorch.log_model(model_AE, "best_model_AE")
                    if model_classifier is not None:
                        mlflow.pytorch.log_model(
                            model_classifier, "best_model_classifier"
                        )

                # artifacts: save best classification report & CM
                mlflow.log_dict(
                    val_metrics["classification_report"],
                    "metrics/best_val_classification_report.json",
                )
                mlflow.log_dict(
                    val_metrics["confusion_matrix"],
                    "metrics/best_val_confusion_matrix.json",
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(
                        f"Early stopping – no improvement for {patience} epochs."
                    )
                    break

    # ----------------------------------------------------------------------
    # 5) Wrap‑up: restore best weights -------------------------------------
    if not run_full_epochs:
        model_AE.load_state_dict(best_AE_state)
        if model_classifier is not None and best_clf_state is not None:
            model_classifier.load_state_dict(best_clf_state)

    return model_AE, model_classifier



def train_loop_AE_v2(
    model_AE: torch.nn.Module,
    model_classifier: Optional[torch.nn.Module],
    train_dataloaders: Dict[str, torch.utils.data.DataLoader],
    val_loaders: Dict[str, torch.utils.data.DataLoader],
    *,
    # loss functions
    loss_fn_x: Optional[torch.nn.Module] = None,
    loss_fn_y: Optional[torch.nn.Module] = None,
    loss_fn_z: Optional[torch.nn.Module] = None,
    # optimiser etc.
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cuda",
    # training config
    epochs: int = 1000,
    patience: int = 10,
    run_full_epochs: bool = False,
    lambda_x: float = 1.0,
    lambda_y: float = 1.0,
    lambda_z: float = 1.0,
    transform=lambda x: x,
    classes: Optional[List[str]] = None,
    log_model: bool = False,
    verbose: bool = True,
    ES_metric_key = "Ly"  # which metric to use for early stopping
):
    """
    Similar to v1 but Dict of dataloaders.
    """

    best_val_loss = float("inf")
    best_AE_state = copy.deepcopy(model_AE.state_dict())
    best_clf_state = (
        copy.deepcopy(model_classifier.state_dict())
        if model_classifier is not None
        else None
    )
    epochs_no_improve = 0
    print("Starting training")
    for epoch in range(epochs):
        # ------------------------------------------------------------------
        # 1) Train one epoch -------------------------------------------------
        train_loss_dct = training_epoch_AE_v2(
            model_AE,
            optimizer,
            train_dataloaders,
            model_classifier=model_classifier,
            loss_fn_x=loss_fn_x,
            loss_fn_y=loss_fn_y,
            loss_fn_z=loss_fn_z,
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            lambda_z=lambda_z,
            transform=transform,
            device=device,
        )

        # log training scalar losses
        
        _log_scalar_dict("train_", train_loss_dct, step=epoch)

        # ------------------------------------------------------------------
        # 2) Validation ------------------------------------------------------
        val_metrics = val_epoch_AE_v2(
            model_AE,
            val_loaders,
            model_classifier=model_classifier,
            loss_fn_x=loss_fn_x,
            loss_fn_y=loss_fn_y,
            loss_fn_z=loss_fn_z,
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            lambda_z=lambda_z,
            transform = lambda x: x, # No denoising in val. loop  #transform,
            device=device,
            classes=classes,
            return_all_predictions=False,
        )
        val_loss = val_metrics["L"]

        # basic console printout -------------------------------------------
        if verbose:
            print(
                (
                    f"Epoch {epoch}/{epochs} | "
                    f"train L: {train_loss_dct['L']:.4f} "
                    f"val L: {val_loss:.4f}"
                )
            )
        if verbose > 1:
            train_losses = "  Train:      " + " | ".join(
            f"{key}: {value:.3f}" for key, value in train_loss_dct.items() if key != "L"
            )
            val_losses = "  Validation: " + " | ".join(
            f"{key}: {value:.3f}" for key, value in val_metrics.items() if (
                key.startswith("L") and 
                key != "L")
            )
            print(train_losses)
            print(val_losses)

        # log validation classification metrics with legacy helper ----------
        log_metrics_mlflow(
            val_metrics,
            prefix="val_",
            classes=classes,
            step=epoch,
            log_report_and_confusion_matrix_as_artifact=False,
        )

        # also log reconstruction / latent / per‑head losses
        extra_loss_keys = {
            k: v for k, v in val_metrics.items() if k.startswith("L")
        }
        _log_scalar_dict("val_", extra_loss_keys, step=epoch)

        # ------------------------------------------------------------------
        # 3) Scheduler step --------------------------------------------------
        if scheduler is not None:
            scheduler.step(val_loss)

        # ------------------------------------------------------------------
        # 4) Early stopping --------------------------------------------------
        val_loss_ES = val_metrics[ES_metric_key]  # used to be val_loss
        if not run_full_epochs:
            improved = val_loss_ES < best_val_loss
            if improved:
                if verbose:
                    improvement = (-(val_loss_ES - best_val_loss) / best_val_loss)*100
                    print(
                        f"    YEAH!! New best validation {ES_metric_key} ↓ "
                        f"{best_val_loss:.4f} → {val_loss_ES:.4f}  this is a {improvement:.1f}% improvement \n"
                    )
                best_val_loss = val_loss_ES
                best_AE_state = copy.deepcopy(model_AE.state_dict())
                if model_classifier is not None:
                    best_clf_state = copy.deepcopy(model_classifier.state_dict())
                epochs_no_improve = 0

                # optional MLflow model snapshot
                if log_model:
                    mlflow.pytorch.log_model(model_AE, "best_model_AE")
                    if model_classifier is not None:
                        mlflow.pytorch.log_model(
                            model_classifier, "best_model_classifier"
                        )

                # artifacts: save best classification report & CM
                mlflow.log_dict(
                    val_metrics["classification_report"],
                    "metrics/best_val_classification_report.json",
                )
                mlflow.log_dict(
                    val_metrics["confusion_matrix"],
                    "metrics/best_val_confusion_matrix.json",
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(
                        f"Early stopping – no improvement for {patience} epochs."
                    )
                    break

    # ----------------------------------------------------------------------
    # 5) Wrap‑up: restore best weights -------------------------------------
    if not run_full_epochs:
        model_AE.load_state_dict(best_AE_state)
        if model_classifier is not None and best_clf_state is not None:
            model_classifier.load_state_dict(best_clf_state)

    return model_AE, model_classifier



def train_loop_AE_v3(
    model_AE: torch.nn.Module,
    model_classifier: Optional[torch.nn.Module],
    train_dataloaders: Dict[str, torch.utils.data.DataLoader],
    val_loaders: Dict[str, torch.utils.data.DataLoader],
    *,
    # loss functions
    loss_fn_x: Optional[torch.nn.Module] = None,
    loss_fn_y: Optional[torch.nn.Module] = None,
    loss_fn_z: Optional[torch.nn.Module] = None,
    loss_fn_z_triplet_classes: Optional[torch.nn.Module] = None,
    # optimiser etc.
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cuda",
    # training config
    epochs: int = 1000,
    patience: int = 10,
    run_full_epochs: bool = False,
    lambda_x: float = 1.0,
    lambda_y: float = 1.0,
    lambda_z: float = 1.0,
    lambda_z_triplet_classes = 1.0,  
    transform=lambda x: x,
    classes: Optional[List[str]] = None,
    log_model: bool = False,
    verbose: bool = True,
    ES_metric_key = "Ly",  # which metric to use for early stopping
    append_BEST_VAL_as_last = False
):
    """
    Similar to v2 but added triplet loss. 
    """

    best_val_loss = float("inf")
    best_AE_state = copy.deepcopy(model_AE.state_dict())
    best_clf_state = (
        copy.deepcopy(model_classifier.state_dict())
        if model_classifier is not None
        else None
    )
    epochs_no_improve = 0
    print("Starting training")

    metrics_Tr, metrics_Val = [], []
    for epoch in range(epochs):
        # ------------------------------------------------------------------
        # 1) Train one epoch -------------------------------------------------
        train_metrics_dct = training_epoch_AE_v3(
            model_AE,
            optimizer,
            train_dataloaders,
            model_classifier=model_classifier,
            loss_fn_x=loss_fn_x,
            loss_fn_y=loss_fn_y,
            loss_fn_z=loss_fn_z,
            loss_fn_z_triplet_classes = loss_fn_z_triplet_classes,
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            lambda_z=lambda_z,
            lambda_z_triplet_classes = lambda_z_triplet_classes, 
            transform=transform,
            device=device,
        )
        metrics_Tr.append(train_metrics_dct)

        # log training scalar losses
        
        _log_scalar_dict("train_", train_metrics_dct, step=epoch)

        # ------------------------------------------------------------------
        # 2) Validation ------------------------------------------------------
        val_metrics_dct = val_epoch_AE_v3(
            model_AE,
            val_loaders,
            model_classifier=model_classifier,
            loss_fn_x=loss_fn_x,
            loss_fn_y=loss_fn_y,
            loss_fn_z=loss_fn_z,
            loss_fn_z_triplet_classes = loss_fn_z_triplet_classes,
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            lambda_z=lambda_z,
            lambda_z_triplet_classes = lambda_z_triplet_classes, 
            transform = lambda x: x, # No denoising in val. loop  #transform,
            device=device,
            classes=classes,
            return_all_predictions=False,
            calc_ICC3=0  # maybe add for HPS...
        )
        val_loss = val_metrics_dct["L"]
        metrics_Val.append(val_metrics_dct)

        # basic console printout -------------------------------------------
        if verbose:
            print(
                (
                    f"Epoch {epoch}/{epochs} | "
                    f"train L: {train_metrics_dct['L']:.4f} "
                    f"val L: {val_loss:.4f}"
                )
            )
        if verbose > 1:
            train_losses = "  Train:      " + " | ".join(
            f"{key}: {value:.3f}" for key, value in train_metrics_dct.items() if key != "L"
            )
            val_losses = "  Validation: " + " | ".join(
            f"{key}: {value:.3f}" for key, value in val_metrics_dct.items() if (
                key.startswith("L") and 
                key != "L")
            )
            print(train_losses)
            print(val_losses)

        if verbose > 2:
            # define the names of your component losses in order
            loss_keys = ["Lx", "Ly", "Lz", "Lz_triplet_classes"]
            # collect the corresponding lambda weights
            lambda_vals = [lambda_x, lambda_y, lambda_z, lambda_z_triplet_classes]

            # build and print the training-relative‐loss line
            train_rel = "     rel. loss contribution Tr.: " + " | ".join(
                f"{k}: {(w * train_metrics_dct[k] / train_metrics_dct['L']):.2f}"
                for k, w in zip(loss_keys, lambda_vals)
                if k in train_metrics_dct
            )
            print(train_rel)

            # build and print the validation-relative‐loss line
            val_rel = "     rel. loss contribution Val: " + " | ".join(
                f"{k}: {(w * val_metrics_dct[k] / val_metrics_dct['L']):.2f}"
                for k, w in zip(loss_keys, lambda_vals)
                if k in val_metrics_dct
            )
            print(val_rel)

        # log validation classification metrics with legacy helper ----------
        log_metrics_mlflow(
            val_metrics_dct,
            prefix="val_",
            classes=classes,
            step=epoch,
            log_report_and_confusion_matrix_as_artifact=False,
        )

        # also log reconstruction / latent / per‑head losses
        extra_loss_keys = {
            k: v for k, v in val_metrics_dct.items() if k.startswith("L")
        }
        _log_scalar_dict("val_", extra_loss_keys, step=epoch)

        # ------------------------------------------------------------------
        # 3) Scheduler step --------------------------------------------------
        if scheduler is not None:
            scheduler.step(val_loss)

        # ------------------------------------------------------------------
        # 4) Early stopping --------------------------------------------------
        val_loss_ES = val_metrics_dct[ES_metric_key]  # used to be val_loss
        if not run_full_epochs:
            improved = val_loss_ES < best_val_loss
            if improved:
                val_metrics_dct_BEST = val_metrics_dct.copy()
                train_metrics_dct_BEST = train_metrics_dct.copy()
                if verbose:
                    improvement = (-(val_loss_ES - best_val_loss) / best_val_loss)*100
                    print(
                        f"    YEAH!! New best validation {ES_metric_key} ↓ "
                        f"{best_val_loss:.4f} → {val_loss_ES:.4f}  this is a {improvement:.1f}% improvement \n"
                    )
                best_val_loss = val_loss_ES
                best_AE_state = copy.deepcopy(model_AE.state_dict())
                if model_classifier is not None:
                    best_clf_state = copy.deepcopy(model_classifier.state_dict())
                epochs_no_improve = 0

                # optional MLflow model snapshot
                if log_model:
                    mlflow.pytorch.log_model(model_AE, "best_model_AE")
                    if model_classifier is not None:
                        mlflow.pytorch.log_model(
                            model_classifier, "best_model_classifier"
                        )

                # artifacts: save best classification report & CM
                mlflow.log_dict(
                    val_metrics_dct["classification_report"],
                    "metrics/best_val_classification_report.json",
                )
                mlflow.log_dict(
                    val_metrics_dct["confusion_matrix"],
                    "metrics/best_val_confusion_matrix.json",
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(
                        f"Early stopping – no improvement for {patience} epochs."
                    )
                    break


    if append_BEST_VAL_as_last: 
        metrics_Tr.append(train_metrics_dct_BEST)
        metrics_Val.append(val_metrics_dct_BEST)



    # ----------------------------------------------------------------------
    # 5) Wrap‑up: restore best weights -------------------------------------
    if not run_full_epochs:
        model_AE.load_state_dict(best_AE_state)
        if model_classifier is not None and best_clf_state is not None:
            model_classifier.load_state_dict(best_clf_state)

    return model_AE, model_classifier, metrics_Tr, metrics_Val




# ---------------------------------------------------------------------------
# Validation epoch (AutoEncoder + multi‑head classifier)
# ---------------------------------------------------------------------------

def val_epoch_AE_v1(
    model_AE: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    *,
    model_classifier: Optional[torch.nn.Module] = None,
    loss_fn_x: Optional[torch.nn.Module] = None,
    loss_fn_y: Optional[torch.nn.Module] = None,
    loss_fn_z: Optional[torch.nn.Module] = None,
    lambda_x: float = 1.0,
    lambda_y: float = 1.0,
    lambda_z: float = 1.0,
    transform=lambda x: x,
    device: str = "cuda",
    classes: Optional[List[str]] = None,
    return_all_predictions: bool = False,
):
    """Validation loop comparable to the legacy *val_epoch* but aware of the
    multi‑headed classifier architecture.

    Returns
    -------
    metrics : dict
        Scalar losses (L, Lx, Ly, Lz, per‑head Ly_*), classical classification
        metrics, confusion matrix and sklearn classification report.
    outputs_all_samples : dict, optional
        Returned only when *return_all_predictions* is *True*.
    """

    # ------------------------------------------------------------------
    # 0) Book‑keeping & preparation
    # ------------------------------------------------------------------
    dummy = DummyReturnZeroLoss(device)
    loss_fn_x = dummy if loss_fn_x is None else loss_fn_x.to(device).eval()
    loss_fn_y = dummy if loss_fn_y is None else loss_fn_y.to(device).eval()
    loss_fn_z = dummy if loss_fn_z is None else loss_fn_z.to(device).eval()

    model_AE.eval().to(device)
    if model_classifier is not None:
        model_classifier.eval().to(device)
        running_head_loss: defaultdict[str, float] = defaultdict(float)
        head_counts: defaultdict[str, int] = defaultdict(int)

        # Max width across all classifier heads – used to build batch_logits
        max_out_dim = max(
            hinfo["out_dim"]
            for hinfo in model_classifier.classifier_head_infos.values()
        )
    else:
        running_head_loss = defaultdict(float)
        head_counts = defaultdict(int)
        max_out_dim = 1  # trivial – no classifier

    running_loss = dict(Lx=0.0, Ly=0.0, Lz=0.0, L=0.0)
    n_samples = 0

    # containers for classification statistics
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_logits: list[np.ndarray] = []
    extra_keys = [
        "file_name",
        "score_type",
        "JSN_or_ERO",
        "extremity",
        "patient_id",
    ]
    all_extras: Dict[str, list] = {k: [] for k in extra_keys}

    # ------------------------------------------------------------------
    # 1) Main validation loop
    # ------------------------------------------------------------------
    with torch.no_grad():
        for batch in dataloader:
            X = batch["img"].to(device)
            Y = batch["score"].to(device)
            score_types = batch["score_type"]  # list[str] – one per sample
            B = X.size(0)

            # ----------------------------------------------------------
            # 1.1 Autoencoder forward & reconstruction/latent losses
            # ----------------------------------------------------------
            X_pred, z = model_AE(transform(X))
            loss_x = loss_fn_x(X_pred, X)
            loss_z = loss_fn_z(z, z * 0)

            # ----------------------------------------------------------
            # 1.2 Classification path (if present)
            # ----------------------------------------------------------
            loss_y = torch.tensor(0.0, device=device)
            # Initialise with -inf so invalid columns never win arg‑max
            batch_logits = torch.full((B, max_out_dim), float("-inf"), device=device)
            batch_preds = torch.empty(B, dtype=torch.long, device=device)

            if model_classifier is not None:
                out_dict = model_classifier(z, score_types, return_dict=True)

                for head_name, (idx, logits) in out_dict.items():
                    cnt = idx.numel()
                    if cnt == 0:
                        continue

                    # ---- loss ------------------------------------------------
                    head_target = Y[idx]
                    head_loss = loss_fn_y(logits, head_target)
                    w = model_classifier.classifier_head_infos[head_name].get(
                        "loss_weight", 1.0
                    )
                    loss_y += head_loss * cnt * w  # Σ (loss_i * w)

                    # per‑head running sums (raw, un‑weighted)
                    running_head_loss[head_name] += head_loss.item() * cnt
                    head_counts[head_name] += cnt

                    # ---- store preds & logits ------------------------------
                    batch_logits[idx, : logits.shape[1]] = logits
                    batch_preds[idx] = logits.argmax(dim=1)

                # convert Σ(loss*count) ➜ mean per sample of *whole batch*
                loss_y = loss_y / B
            else:
                # no classifier – leave loss_y = 0, predictions dummy‑filled
                batch_logits.zero_()
                batch_preds.fill_(0)

            # ----------------------------------------------------------
            # 1.3 Total weighted loss & running sums
            # ----------------------------------------------------------
            loss_total = lambda_x * loss_x + lambda_y * loss_y + lambda_z * loss_z

            running_loss["Lx"] += loss_x.item() * B
            running_loss["Ly"] += loss_y.item() * B
            running_loss["Lz"] += loss_z.item() * B
            running_loss["L"] += loss_total.item() * B
            n_samples += B

            # ----------------------------------------------------------
            # 1.4 Collect per‑sample statistics for later metrics
            # ----------------------------------------------------------
            all_preds.extend(batch_preds.cpu().numpy())
            all_labels.extend(Y.cpu().numpy())
            all_logits.extend(batch_logits.cpu().numpy())
            for k in extra_keys:
                if k in batch:
                    all_extras[k].extend(batch[k])

    # ------------------------------------------------------------------
    # 2) Aggregate scalar losses
    # ------------------------------------------------------------------
    for k in running_loss:
        running_loss[k] /= n_samples if n_samples else 1.0

    if model_classifier is not None:
        for head_name in model_classifier.classifier_head_infos:
            cnt = head_counts[head_name]
            running_loss[f"Ly_{head_name}"] = (
                running_head_loss[head_name] / cnt if cnt else 0.0
            )

    # ------------------------------------------------------------------
    # 3) Classical classification statistics (global)
    # ------------------------------------------------------------------
    metrics = {
        **running_loss,
        "loss": running_loss["L"],  # compatibility with legacy logger
    }

    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    all_logits_np = np.array(all_logits)
    all_probs_np = F.softmax(torch.from_numpy(all_logits_np), dim=1).numpy()

    metrics.update(calculate_some_classification_metrics(all_preds_np, all_labels_np))

    # ---- top‑2 accuracy --------------------------------------------------
    top2 = np.argsort(all_probs_np, axis=1)[:, -2:]
    top2_correct = sum(label in top2[i] for i, label in enumerate(all_labels_np))
    metrics["top2_accuracy"] = top2_correct / len(all_labels_np)

    # ---- confusion matrix & classification report -----------------------
    cm = confusion_matrix(all_labels_np, all_preds_np)
    metrics["confusion_matrix"] = cm.tolist()

    if classes is not None:
        labels = list(range(len(classes)))
        report = classification_report(
            all_labels_np,
            all_preds_np,
            labels=labels,
            target_names=classes,
            output_dict=True,
            zero_division=0.0,
        )
    else:
        report = classification_report(
            all_labels_np, all_preds_np, output_dict=True, zero_division=0.0
        )
    metrics["classification_report"] = report

    # ------------------------------------------------------------------
    # 4) Optionally return per‑sample outputs
    # ------------------------------------------------------------------
    if return_all_predictions:
        outputs_all_samples = {
            "labels": all_labels_np,
            "preds": all_preds_np,
            "probs": all_probs_np,
            **all_extras,
        }
        return metrics, outputs_all_samples

    return metrics



def val_epoch_AE_v2(
    model_AE: torch.nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    *,
    model_classifier: Optional[torch.nn.Module] = None,
    loss_fn_x: Optional[torch.nn.Module] = None,
    loss_fn_y: Optional[torch.nn.Module] = None,
    loss_fn_z: Optional[torch.nn.Module] = None,
    lambda_x: float = 1.0,
    lambda_y: float = 1.0,
    lambda_z: float = 1.0,
    transform=lambda x: x,
    device: str = "cuda",
    classes: Optional[List[str]] = None,
    return_all_predictions: bool = False,
    calc_ICC3=0
):
    """
    Difference to v1: 
     dataloader -> dataloaders (dict)
     model(img) -> mode(img, score_type)

    Returns
    -------
    metrics : dict
        Scalar losses (L, Lx, Ly, Lz, per‑head Ly_*), classical classification
        metrics, confusion matrix and sklearn classification report.
    outputs_all_samples : dict, optional
        Returned only when *return_all_predictions* is *True*.
    """

    # ------------------------------------------------------------------
    # 0) Book‑keeping & preparation
    # ------------------------------------------------------------------
    dummy = DummyReturnZeroLoss(device)
    loss_fn_x = dummy if loss_fn_x is None else loss_fn_x.to(device).eval()
    loss_fn_y = dummy if loss_fn_y is None else loss_fn_y.to(device).eval()
    loss_fn_z = dummy if loss_fn_z is None else loss_fn_z.to(device).eval()

    model_AE.eval().to(device)
    if model_classifier is not None:
        model_classifier.eval().to(device)
        running_head_loss: defaultdict[str, float] = defaultdict(float)
        head_counts: defaultdict[str, int] = defaultdict(int)

        # Max width across all classifier heads – used to build batch_logits
        max_out_dim = max(
            hinfo["out_dim"]
            for hinfo in model_classifier.classifier_head_infos.values()
        )
    else:
        running_head_loss = defaultdict(float)
        head_counts = defaultdict(int)
        max_out_dim = 1  # trivial – no classifier

    running_loss = dict(Lx=0.0, Ly=0.0, Lz=0.0, L=0.0)
    n_samples = 0

    # containers for classification statistics
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_logits: list[np.ndarray] = []
    extra_keys = [
        "file_name",
        "score_type",
        "JSN_or_ERO",
        "extremity",
        "patient_id",
    ]
    all_extras: Dict[str, list] = {k: [] for k in extra_keys}

    # ------------------------------------------------------------------
    # 1) Main validation loop
    # ------------------------------------------------------------------
    with torch.no_grad():
        for dataloader_name, dataloader in dataloaders.items():
            for batch in dataloader:
                X = batch["img"].to(device)
                Y = batch["score"].to(device)
                score_types = batch["score_type"]  # list[str] – one per sample
                B = X.size(0)

                # ----------------------------------------------------------
                # 1.1 Autoencoder forward & reconstruction/latent losses
                # ----------------------------------------------------------
                X_pred, z = model_AE(transform(X), score_types)
                loss_x = loss_fn_x(X_pred, X)
                loss_z = loss_fn_z(z, z * 0)

                # ----------------------------------------------------------
                # 1.2 Classification path (if present)
                # ----------------------------------------------------------
                loss_y = torch.tensor(0.0, device=device)
                # Initialise with -inf so invalid columns never win arg‑max
                batch_logits = torch.full((B, max_out_dim), float("-inf"), device=device)
                batch_preds = torch.empty(B, dtype=torch.long, device=device)

                if model_classifier is not None:
                    out_dict = model_classifier(z, score_types, return_dict=True)

                    for head_name, (idx, logits) in out_dict.items():
                        cnt = idx.numel()
                        if cnt == 0:
                            continue

                        # ---- loss ------------------------------------------------
                        head_target = Y[idx]
                        head_loss = loss_fn_y(logits, head_target)
                        w = model_classifier.classifier_head_infos[head_name].get(
                            "loss_weight", 1.0
                        )
                        loss_y += head_loss * cnt * w  # Σ (loss_i * w)

                        # per‑head running sums (raw, un‑weighted)
                        running_head_loss[head_name] += head_loss.item() * cnt
                        head_counts[head_name] += cnt

                        # ---- store preds & logits ------------------------------
                        batch_logits[idx, : logits.shape[1]] = logits
                        batch_preds[idx] = logits.argmax(dim=1)

                    # convert Σ(loss*count) ➜ mean per sample of *whole batch*
                    loss_y = loss_y / B
                else:
                    # no classifier – leave loss_y = 0, predictions dummy‑filled
                    batch_logits.zero_()
                    batch_preds.fill_(0)

                # ----------------------------------------------------------
                # 1.3 Total weighted loss & running sums
                # ----------------------------------------------------------
                loss_total = lambda_x * loss_x + lambda_y * loss_y + lambda_z * loss_z

                running_loss["Lx"] += loss_x.item() * B
                running_loss["Ly"] += loss_y.item() * B
                running_loss["Lz"] += loss_z.item() * B
                running_loss["L"] += loss_total.item() * B
                n_samples += B

                # ----------------------------------------------------------
                # 1.4 Collect per‑sample statistics for later metrics
                # ----------------------------------------------------------
                all_preds.extend(batch_preds.cpu().numpy())
                all_labels.extend(Y.cpu().numpy())
                all_logits.extend(batch_logits.cpu().numpy())
                for k in extra_keys:
                    if k in batch:
                        all_extras[k].extend(batch[k])

    # ------------------------------------------------------------------
    # 2) Aggregate scalar losses
    # ------------------------------------------------------------------
    for k in running_loss:
        running_loss[k] /= n_samples if n_samples else 1.0

    if model_classifier is not None:
        for head_name in model_classifier.classifier_head_infos:
            cnt = head_counts[head_name]
            running_loss[f"Ly_{head_name}"] = (
                running_head_loss[head_name] / cnt if cnt else 0.0
            )

    # ------------------------------------------------------------------
    # 3) Classical classification statistics (global)
    # ------------------------------------------------------------------
    metrics = {
        **running_loss,
        "loss": running_loss["L"],  # compatibility with legacy logger
    }

    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    all_logits_np = np.array(all_logits) 
    all_probs_np = F.softmax(torch.from_numpy(all_logits_np), dim=1).numpy() 
    # TODO there are likely issues with this if number of classes is differnt 

    metrics.update(calculate_some_classification_metrics(all_preds_np, all_labels_np, calc_ICC3=calc_ICC3))

    # ---- top‑2 accuracy --------------------------------------------------
    top2 = np.argsort(all_probs_np, axis=1)[:, -2:]
    top2_correct = sum(label in top2[i] for i, label in enumerate(all_labels_np))
    metrics["top2_accuracy"] = top2_correct / len(all_labels_np)

    # ---- confusion matrix & classification report -----------------------
    cm = confusion_matrix(all_labels_np, all_preds_np)
    metrics["confusion_matrix"] = cm.tolist()

    if classes is not None:
        labels = list(range(len(classes)))
        report = classification_report(
            all_labels_np,
            all_preds_np,
            labels=labels,
            target_names=classes,
            output_dict=True,
            zero_division=0.0,
        )
    else:
        report = classification_report(
            all_labels_np, all_preds_np, output_dict=True, zero_division=0.0
        )
    metrics["classification_report"] = report

    # ------------------------------------------------------------------
    # 4) Optionally return per‑sample outputs
    # ------------------------------------------------------------------
    if return_all_predictions:
        outputs_all_samples = {
            "labels": all_labels_np,
            "preds": all_preds_np,
            "probs": all_probs_np,
            **all_extras,
        }
        return metrics, outputs_all_samples

    return metrics


def val_epoch_AE_v3(
    model_AE: torch.nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    *,
    model_classifier: Optional[torch.nn.Module] = None,
    loss_fn_x: Optional[torch.nn.Module] = None,
    loss_fn_y: Optional[torch.nn.Module] = None,
    loss_fn_z: Optional[torch.nn.Module] = None,
    loss_fn_z_triplet_classes: Optional[torch.nn.Module] = None,
    lambda_x: float = 1.0,
    lambda_y: float = 1.0,
    lambda_z: float = 1.0,
    lambda_z_triplet_classes: float = 1.0,
    transform=lambda x: x,
    device: str = "cuda",
    classes: Optional[List[str]] = None,
    return_all_predictions: bool = False,
    calc_ICC3=0
):
    """
    Difference to v1: 
     dataloader -> dataloaders (dict)
     model(img) -> mode(img, score_type)

    Returns
    -------
    metrics : dict
        Scalar losses (L, Lx, Ly, Lz, per‑head Ly_*), classical classification
        metrics, confusion matrix and sklearn classification report.
    outputs_all_samples : dict, optional
        Returned only when *return_all_predictions* is *True*.
    """

    # ------------------------------------------------------------------
    # 0) Book‑keeping & preparation
    # ------------------------------------------------------------------
    dummy = DummyReturnZeroLoss(device)
    loss_fn_x = dummy if loss_fn_x is None else loss_fn_x.to(device).eval()
    loss_fn_y = dummy if loss_fn_y is None else loss_fn_y.to(device).eval()
    loss_fn_z = dummy if loss_fn_z is None else loss_fn_z.to(device).eval()
    loss_fn_z_triplet_classes = dummy if loss_fn_z_triplet_classes is None else loss_fn_z_triplet_classes.to(device).eval()

    model_AE.eval().to(device)
    if model_classifier is not None:
        model_classifier.eval().to(device)
        running_head_loss: defaultdict[str, float] = defaultdict(float)
        head_counts: defaultdict[str, int] = defaultdict(int)

        # Max width across all classifier heads – used to build batch_logits
        max_out_dim = max(
            hinfo["out_dim"]
            for hinfo in model_classifier.classifier_head_infos.values()
        )
    else:
        running_head_loss = defaultdict(float)
        head_counts = defaultdict(int)
        max_out_dim = 1  # trivial – no classifier

    running_loss = dict(Lx=0.0, Ly=0.0, Lz=0.0, L=0.0, Lz_triplet_classes=0.0)
    n_samples = 0

    # containers for classification statistics
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_logits: list[np.ndarray] = []
    extra_keys = [
        "file_name",
        "score_type",
        "JSN_or_ERO",
        "extremity",
        "patient_id",
    ]
    all_extras: Dict[str, list] = {k: [] for k in extra_keys}

    # ------------------------------------------------------------------
    # 1) Main validation loop
    # ------------------------------------------------------------------
    with torch.no_grad():
        for dataloader_name, dataloader in dataloaders.items():
            for batch in dataloader:
                X = batch["img"].to(device)
                Y = batch["score"].to(device)
                score_types = batch["score_type"]  # list[str] – one per sample
                B = X.size(0)

                # ----------------------------------------------------------
                # 1.1 Autoencoder forward & reconstruction/latent losses
                # ----------------------------------------------------------
                X_pred, z = model_AE(transform(X), score_types)
                loss_x = loss_fn_x(X_pred, X)
                loss_z = loss_fn_z(z, z * 0)
                loss_z_triplet_classes = loss_fn_z_triplet_classes(labels=Y, embeddings=z)

                # ----------------------------------------------------------
                # 1.2 Classification path (if present)
                # ----------------------------------------------------------
                loss_y = torch.tensor(0.0, device=device)
                # Initialise with -inf so invalid columns never win arg‑max
                batch_logits = torch.full((B, max_out_dim), float("-inf"), device=device)
                batch_preds = torch.empty(B, dtype=torch.long, device=device)

                if model_classifier is not None:
                    out_dict = model_classifier(z, score_types, return_dict=True)

                    for head_name, (idx, logits) in out_dict.items():
                        cnt = idx.numel()
                        if cnt == 0:
                            continue

                        # ---- loss ------------------------------------------------
                        head_target = Y[idx]
                        head_loss = loss_fn_y(logits, head_target)
                        w = model_classifier.classifier_head_infos[head_name].get(
                            "loss_weight", 1.0
                        )
                        loss_y += head_loss * cnt * w  # Σ (loss_i * w)

                        # per‑head running sums (raw, un‑weighted)
                        running_head_loss[head_name] += head_loss.item() * cnt
                        head_counts[head_name] += cnt

                        # ---- store preds & logits ------------------------------
                        batch_logits[idx, : logits.shape[1]] = logits
                        batch_preds[idx] = logits.argmax(dim=1)

                    # convert Σ(loss*count) ➜ mean per sample of *whole batch*
                    loss_y = loss_y / B
                else:
                    # no classifier – leave loss_y = 0, predictions dummy‑filled
                    batch_logits.zero_()
                    batch_preds.fill_(0)

                # ----------------------------------------------------------
                # 1.3 Total weighted loss & running sums
                # ----------------------------------------------------------
                loss_total = lambda_x * loss_x + lambda_y * loss_y + lambda_z * loss_z

                running_loss["Lx"] += loss_x.item() * B
                running_loss["Ly"] += loss_y.item() * B
                running_loss["Lz"] += loss_z.item() * B
                running_loss["Lz_triplet_classes"] += loss_z_triplet_classes.item() * B
                running_loss["L"] += loss_total.item() * B
                n_samples += B

                # ----------------------------------------------------------
                # 1.4 Collect per‑sample statistics for later metrics
                # ----------------------------------------------------------
                all_preds.extend(batch_preds.cpu().numpy())
                all_labels.extend(Y.cpu().numpy())
                all_logits.extend(batch_logits.cpu().numpy())
                for k in extra_keys:
                    if k in batch:
                        all_extras[k].extend(batch[k])

    # ------------------------------------------------------------------
    # 2) Aggregate scalar losses
    # ------------------------------------------------------------------
    for k in running_loss:
        running_loss[k] /= n_samples if n_samples else 1.0

    if model_classifier is not None:
        for head_name in model_classifier.classifier_head_infos:
            cnt = head_counts[head_name]
            running_loss[f"Ly_{head_name}"] = (
                running_head_loss[head_name] / cnt if cnt else 0.0
            )

    # ------------------------------------------------------------------
    # 3) Classical classification statistics (global)
    # ------------------------------------------------------------------
    metrics = {
        **running_loss,
        "loss": running_loss["L"],  # compatibility with legacy logger
    }

    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    all_logits_np = np.array(all_logits) 
    all_probs_np = F.softmax(torch.from_numpy(all_logits_np), dim=1).numpy() 
    # TODO there are likely issues with this if number of classes is differnt 

    metrics.update(calculate_some_classification_metrics(all_preds_np, all_labels_np, calc_ICC3=calc_ICC3))

    # ---- top‑2 accuracy --------------------------------------------------
    top2 = np.argsort(all_probs_np, axis=1)[:, -2:]
    top2_correct = sum(label in top2[i] for i, label in enumerate(all_labels_np))
    metrics["top2_accuracy"] = top2_correct / len(all_labels_np)

    # ---- confusion matrix & classification report -----------------------
    cm = confusion_matrix(all_labels_np, all_preds_np)
    metrics["confusion_matrix"] = cm.tolist()

    if classes is not None:
        labels = list(range(len(classes)))
        report = classification_report(
            all_labels_np,
            all_preds_np,
            labels=labels,
            target_names=classes,
            output_dict=True,
            zero_division=0.0,
        )
    else:
        report = classification_report(
            all_labels_np, all_preds_np, output_dict=True, zero_division=0.0
        )
    metrics["classification_report"] = report

    # ------------------------------------------------------------------
    # 4) Optionally return per‑sample outputs
    # ------------------------------------------------------------------
    if return_all_predictions:
        outputs_all_samples = {
            "labels": all_labels_np,
            "preds": all_preds_np,
            "probs": all_probs_np,
            **all_extras,
        }
        return metrics, outputs_all_samples

    return metrics





# ---------------------------------------------------------------------------
# Training epoch (AutoEncoder + multi‑head classifier)
# ----------------------
def training_epoch_AE_v1(
    model_AE,
    optimizer,
    dataloader,
    model_classifier=None,
    *,
    loss_fn_x=None,
    loss_fn_y=None,
    loss_fn_z=None,
    lambda_x: float = 1.0,
    lambda_y: float = 1.0,
    lambda_z: float = 1.0,
    transform=lambda x: x,
    device="cuda",
    debugging: bool = False,
):
    # ----------------------------- set-up ---------------------------------- #
    model_AE.train().to(device)

    # fall-back losses
    dummy = DummyReturnZeroLoss(device)
    loss_fn_x = dummy if loss_fn_x is None else loss_fn_x.to(device).train()
    loss_fn_y = dummy if loss_fn_y is None else loss_fn_y.to(device).train()
    loss_fn_z = dummy if loss_fn_z is None else loss_fn_z.to(device).train()

    if model_classifier is not None:
        model_classifier.train().to(device)

        # running sums for each head
        running_head_loss = defaultdict(float)  # Σ loss*count
        head_counts = defaultdict(int)          # Σ count

    # running sums for global losses
    running_loss = dict(Lx=0.0, Ly=0.0, Lz=0.0, L=0.0)
    samples_tr = 0

    # ---------------------------- epoch loop ------------------------------- #
    for batch in dataloader:
        X      = batch["img"].to(device)
        y      = batch["score"].to(device)
        s_type = batch["score_type"]             # list[str]

        B = X.size(0)
        samples_tr += B
        optimizer.zero_grad()

        # -- autoencoder ---------------------------------------------------- #
        X_pred, z = model_AE(transform(X))       # ONE forward pass

        # -- classifier heads ---------------------------------------------- #
        loss_y = torch.tensor(0.0, device=device)
        if model_classifier is not None:
            out_dict = model_classifier(z, s_type, return_dict=True)

            # accumulate per-head losses
            for head_name, (idx, logits) in out_dict.items():
                cnt = idx.numel()                # how many samples for this head
                if cnt == 0:
                    continue                     # no contribution this batch

                head_target = y[idx]
                head_loss   = loss_fn_y(logits, head_target)   # mean over head-batch

                w = model_classifier.classifier_head_infos[head_name].get(
                    "loss_weight", 1.0
                )

                # weighted sum for global Ly
                loss_y += head_loss * cnt * w

                # running stats for this head (raw, un-weighted)
                running_head_loss[head_name] += head_loss.item() * cnt
                head_counts[head_name]       += cnt

            # convert Σ(loss*count) ➜ mean per sample of *whole batch*
            loss_y = loss_y / B

        # -- reconstruction & latent losses -------------------------------- #
        loss_x = loss_fn_x(X_pred, X)
        loss_z = loss_fn_z(z, z * 0)

        # -- total loss & optimisation ------------------------------------- #
        loss = lambda_x * loss_x + lambda_y * loss_y + lambda_z * loss_z
        loss.backward()
        optimizer.step()

        # -- running averages ---------------------------------------------- #
        running_loss["Lx"] += loss_x.item() * B
        running_loss["Ly"] += loss_y.item() * B
        running_loss["Lz"] += loss_z.item() * B
        running_loss["L"]  += loss.item()  * B

        if debugging:
            print("debugging --- break after 1 step")
            break

    # ------------------------------- wrap-up ------------------------------- #
    for k in running_loss:
        running_loss[k] /= samples_tr            # mean per sample

    # per-head means
    if model_classifier is not None:
        for head_name in model_classifier.classifier_head_infos:
            cnt = head_counts[head_name]
            running_loss[f"Ly_{head_name}"] = (
                running_head_loss[head_name] / cnt if cnt else 0.0
            )

    return running_loss




# ---------------------------------------------------------------------------
# Training epoch (AutoEncoder + multi‑head classifier)
# ----------------------
def training_epoch_AE_v2(
    model_AE,
    optimizer,
    dataloaders: dict,
    model_classifier=None,
    *,
    loss_fn_x=None,
    loss_fn_y=None,
    loss_fn_z=None,
    lambda_x: float = 1.0,
    lambda_y: float = 1.0,
    lambda_z: float = 1.0,
    transform=lambda x: x,
    device="cuda",
    debugging: bool = False
):
    """
    Difference to v1: 
     dataloader -> dataloaders (dict)
     model(img) -> mode(img, score_type)
    """
    
    
    # ----------------------------- set-up ---------------------------------- #
    model_AE.train().to(device)

    # fall-back losses
    dummy = DummyReturnZeroLoss(device)
    loss_fn_x = dummy if loss_fn_x is None else loss_fn_x.to(device).train()
    loss_fn_y = dummy if loss_fn_y is None else loss_fn_y.to(device).train()
    loss_fn_z = dummy if loss_fn_z is None else loss_fn_z.to(device).train()

    if model_classifier is not None:
        model_classifier.train().to(device)

        # running sums for each head
        running_head_loss = defaultdict(float)  # Σ loss*count
        head_counts = defaultdict(int)          # Σ count

    # running sums for global losses
    running_loss = dict(Lx=0.0, Ly=0.0, Lz=0.0, L=0.0)
    samples_tr = 0

    # ---------------------------- epoch loop ------------------------------- #
    for dataloader_name, dataloader in dataloaders.items():
        for batch in dataloader:
            X      = batch["img"].to(device)
            y      = batch["score"].to(device)
            s_type = batch["score_type"]             # list[str]

            B = X.size(0)
            samples_tr += B
            optimizer.zero_grad()

            # -- autoencoder ---------------------------------------------------- #
            X_pred, z = model_AE(transform(X), s_type)       # ONE forward pass

            # triplet loss on classes: 
            
            

            # -- classifier heads ---------------------------------------------- #
            loss_y = torch.tensor(0.0, device=device)
            if model_classifier is not None:
                out_dict = model_classifier(z, s_type, return_dict=True)

                # accumulate per-head losses
                for head_name, (idx, logits) in out_dict.items():
                    cnt = idx.numel()                # how many samples for this head
                    if cnt == 0:
                        continue                     # no contribution this batch

                    head_target = y[idx]
                    head_loss   = loss_fn_y(logits, head_target)   # mean over head-batch

                    w = model_classifier.classifier_head_infos[head_name].get(
                        "loss_weight", 1.0
                    )

                    # weighted sum for global Ly
                    loss_y += head_loss * cnt * w

                    # running stats for this head (raw, un-weighted)
                    running_head_loss[head_name] += head_loss.item() * cnt
                    head_counts[head_name]       += cnt

                # convert Σ(loss*count) ➜ mean per sample of *whole batch*
                loss_y = loss_y / B

            # -- reconstruction & latent losses -------------------------------- #
            loss_x = loss_fn_x(X_pred, X)
            loss_z = loss_fn_z(z, z * 0)

            # -- total loss & optimisation ------------------------------------- #
            loss = lambda_x * loss_x + lambda_y * loss_y + lambda_z * loss_z
            loss.backward()
            optimizer.step()

            # -- running averages ---------------------------------------------- #
            running_loss["Lx"] += loss_x.item() * B
            running_loss["Ly"] += loss_y.item() * B
            running_loss["Lz"] += loss_z.item() * B
            running_loss["L"]  += loss.item()  * B

            if debugging:
                print("debugging --- break after 1 step")
                break

    # ------------------------------- wrap-up ------------------------------- #
    for k in running_loss:
        running_loss[k] /= samples_tr            # mean per sample

    # per-head means
    if model_classifier is not None:
        for head_name in model_classifier.classifier_head_infos:
            cnt = head_counts[head_name]
            running_loss[f"Ly_{head_name}"] = (
                running_head_loss[head_name] / cnt if cnt else 0.0
            )

    return running_loss


def training_epoch_AE_v3(
    model_AE,
    optimizer,
    dataloaders: dict,
    model_classifier=None,
    *,
    loss_fn_x=None,
    loss_fn_y=None,
    loss_fn_z=None,
    loss_fn_z_triplet_classes = None,
    lambda_x: float = 1.0,
    lambda_y: float = 1.0,
    lambda_z: float = 1.0,
    lambda_z_triplet_classes: float = 0.0,
    transform=lambda x: x,
    device="cuda",
    debugging: bool = False
):
    """
    Difference to v2: 
     dataloader -> dataloaders (dict)
     model(img) -> mode(img, score_type)
    """
    
    
    # ----------------------------- set-up ---------------------------------- #
    model_AE.train().to(device)

    # fall-back losses
    dummy = DummyReturnZeroLoss(device)
    loss_fn_x = dummy if loss_fn_x is None else loss_fn_x.to(device).train()
    loss_fn_y = dummy if loss_fn_y is None else loss_fn_y.to(device).train()
    loss_fn_z = dummy if loss_fn_z is None else loss_fn_z.to(device).train()
    loss_fn_z_triplet_classes = dummy if loss_fn_z_triplet_classes is None else loss_fn_z_triplet_classes.to(device).train()

    if model_classifier is not None:
        model_classifier.train().to(device)

        # running sums for each head
        running_head_loss = defaultdict(float)  # Σ loss*count
        head_counts = defaultdict(int)          # Σ count

    # running sums for global losses
    running_loss = dict(Lx=0.0, Ly=0.0, Lz=0.0, L=0.0, Lz_triplet_classes=0.0)
    samples_tr = 0

    # ---------------------------- epoch loop ------------------------------- #
    for dataloader_name, dataloader in dataloaders.items():
        for batch in dataloader:
            X      = batch["img"].to(device)
            y      = batch["score"].to(device)
            s_type = batch["score_type"]             # list[str]

            B = X.size(0)
            samples_tr += B
            optimizer.zero_grad()

            # -- autoencoder ---------------------------------------------------- #
            X_pred, z = model_AE(transform(X), s_type)       # ONE forward pass

            # triplet loss on classes: 
            loss_z_triplet_classes = loss_fn_z_triplet_classes(labels=y, embeddings=z)            

            # -- classifier heads ---------------------------------------------- #
            loss_y = torch.tensor(0.0, device=device)
            if model_classifier is not None:
                out_dict = model_classifier(z, s_type, return_dict=True)

                # accumulate per-head losses
                for head_name, (idx, logits) in out_dict.items():
                    cnt = idx.numel()                # how many samples for this head
                    if cnt == 0:
                        continue                     # no contribution this batch

                    head_target = y[idx]
                    head_loss   = loss_fn_y(logits, head_target)   # mean over head-batch

                    w = model_classifier.classifier_head_infos[head_name].get(
                        "loss_weight", 1.0
                    )

                    # weighted sum for global Ly
                    loss_y += head_loss * cnt * w

                    # running stats for this head (raw, un-weighted)
                    running_head_loss[head_name] += head_loss.item() * cnt
                    head_counts[head_name]       += cnt

                # convert Σ(loss*count) ➜ mean per sample of *whole batch*
                loss_y = loss_y / B

            # -- reconstruction & latent losses -------------------------------- #
            loss_x = loss_fn_x(X_pred, X)
            loss_z = loss_fn_z(z, z * 0)

            # -- total loss & optimisation ------------------------------------- #
            loss = lambda_x * loss_x + lambda_y * loss_y + lambda_z * loss_z + lambda_z_triplet_classes * loss_z_triplet_classes
            loss.backward()
            optimizer.step()

            # -- running averages ---------------------------------------------- #
            running_loss["Lx"] += loss_x.item() * B
            running_loss["Ly"] += loss_y.item() * B
            running_loss["Lz"] += loss_z.item() * B
            running_loss["Lz_triplet_classes"] += loss_z_triplet_classes.item() * B
            running_loss["L"]  += loss.item()  * B

            if debugging:
                print("debugging --- break after 1 step")
                break

    # ------------------------------- wrap-up ------------------------------- #
    for k in running_loss:
        running_loss[k] /= samples_tr            # mean per sample

    # per-head means
    if model_classifier is not None:
        for head_name in model_classifier.classifier_head_infos:
            cnt = head_counts[head_name]
            running_loss[f"Ly_{head_name}"] = (
                running_head_loss[head_name] / cnt if cnt else 0.0
            )

    return running_loss


# ---------------------------------------------------------------------------
# Test‑set evaluation helper (AE v1)
# ---------------------------------------------------------------------------


def evaluate_and_log_testset_results_AE_v1(
    model_AE:               torch.nn.Module,
    model_classifier:       Optional[torch.nn.Module],
    dataloader:             torch.utils.data.DataLoader,
    *,
    loss_fn_x: Optional[torch.nn.Module] = None,
    loss_fn_y: Optional[torch.nn.Module] = None,
    loss_fn_z: Optional[torch.nn.Module] = None,
    device:    str  = "cuda",
    classes:   Optional[List[str]] = None,
    transform            = lambda x: x,
    prefix:     str  = "test_",
    lambda_x:   float = 1.0,
    lambda_y:   float = 1.0,
    lambda_z:   float = 1.0,
):
    """
    Run *val_epoch_AE_v1* on the test-set, then

    1. log global metrics (existing behaviour),
    2. build + log classification reports
       • by **score_type**
       • by **classifier head** (if available),
    3. log losses and artefacts (npz + recon plots).
    """

    # ------------------------------------------------------------------ #
    # 0) Forward pass through the whole test set
    # ------------------------------------------------------------------ #
    metrics, outputs_all = val_epoch_AE_v1(
        model_AE,
        dataloader,
        model_classifier=model_classifier,
        loss_fn_x=loss_fn_x,
        loss_fn_y=loss_fn_y,
        loss_fn_z=loss_fn_z,
        lambda_x=lambda_x,
        lambda_y=lambda_y,
        lambda_z=lambda_z,
        transform=transform,
        device=device,
        classes=classes,
        return_all_predictions=True,
    )

    # ------------------------------------------------------------------ #
    # 1) Log global metrics (loss, acc, …)
    # ------------------------------------------------------------------ #
    log_metrics_mlflow(
        metrics,
        prefix=prefix,
        classes=classes,
        step=None,
        log_report_and_confusion_matrix_as_artifact=True,
    )

    # ------------------------------------------------------------------ #
    # 2) Per-score-type & per-head classification reports
    # ------------------------------------------------------------------ #
    # --- extract predictions ------------------------------------------ #
    score_types = outputs_all.get("score_type")
    y_true      = outputs_all["labels"]
    y_pred      = outputs_all["preds"]

    if score_types is not None:
        score_types = np.asarray(score_types)

        # ---- per score_type ------------------------------------------- #
        by_st = {}
        for st in np.unique(score_types):
            idx = score_types == st
            by_st[st] = classification_report(
                y_true[idx], y_pred[idx],
                labels=list(range(len(classes))) if classes else None,
                target_names=classes if classes else None,
                output_dict=True, zero_division=0.0,
            )

        mlflow.log_dict(
            by_st, f"metrics/{prefix}classification_report_by_score_type.json"
        )

        # ---- per head (needs classifier) ------------------------------ #
        if model_classifier is not None:
            st2head = make_score_type_2_head_name_dct(
                model_classifier.classifier_head_infos
            )
            heads = np.vectorize(st2head.get)(score_types)

            by_head: Dict[str, Dict] = {}
            for h in np.unique(heads):
                idx = heads == h
                by_head[h] = classification_report(
                    y_true[idx], y_pred[idx],
                    labels=list(range(len(classes))) if classes else None,
                    target_names=classes if classes else None,
                    output_dict=True, zero_division=0.0,
                )

            mlflow.log_dict(
                by_head, f"metrics/{prefix}classification_report_by_head.json"
            )

    # ------------------------------------------------------------------ #
    # 3) Log reconstruction / latent / per-head losses
    # ------------------------------------------------------------------ #
    extra_loss = {k: v for k, v in metrics.items() if k.startswith("L")}
    _log_scalar_dict("val_", extra_loss, step=None)

    # ------------------------------------------------------------------ #
    # 4) Save raw preds + extras as compressed .npz
    # ------------------------------------------------------------------ #
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        np.savez_compressed(tmp.name, **outputs_all)
        mlflow.log_artifact(tmp.name, artifact_path=f"predictions/{prefix}")
        npz_path = tmp.name
    os.remove(npz_path)

    # ------------------------------------------------------------------ #
    # 5) (Optional) plot reconstructions
    # ------------------------------------------------------------------ #
    try:
        n_vis = min(10, len(dataloader.dataset))
        orig, noisy, recon = [], [], []
        model_AE.eval()

        with torch.no_grad():
            for batch in dataloader:
                X_cpu = batch["img"]          # [B,1,H,W]  (assumed 0-1 scaled)
                X     = X_cpu.to(device)
                X_t   = transform(X)
                X_rec, _ = model_AE(X_t)

                for i in range(X.size(0)):
                    orig. append(X_cpu[i,0].numpy())
                    noisy.append(X_t[i,0].cpu().numpy())
                    recon.append(X_rec[i,0].cpu().numpy())
                    if len(orig) >= n_vis:
                        break
                if len(orig) >= n_vis:
                    break

        fig, axes = plt.subplots(n_vis, 3, figsize=(6, 2 * n_vis))
        titles = ["original", "noisy", "recon"]
        for r in range(n_vis):
            for c, img in enumerate((orig[r], noisy[r], recon[r])):
                ax = axes[r, c] if n_vis > 1 else axes[c]
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
                if r == 0:
                    ax.set_title(titles[c])
        fig.tight_layout()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_fig:
            fig.savefig(tmp_fig.name, dpi=150)
            mlflow.log_artifact(tmp_fig.name, artifact_path=f"plots/{prefix}reconstructions")
            plt.close(fig)
            os.remove(tmp_fig.name)
    except Exception as e:
        print("Reconstruction plotting failed:", e)

    # ------------------------------------------------------------------ #
    return metrics, outputs_all





def _collect_examples(
    dl: torch.utils.data.DataLoader,
    model_AE: torch.nn.Module,
    transform: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    n_vis: int = 5,
):
    """Run one minibatch through *model_AE* and gather *n_vis* examples."""
    orig, noisy, recon = [], [], []
    model_AE.eval()

    with torch.no_grad():
        for batch in dl:  # noqa: B007 (batch is used indirectly)
            X_cpu = batch["img"]  # [B, 1, H, W] – assumed 0‑1 scaled
            X = X_cpu.to(device, non_blocking=True)
            score_types = batch["score_type"]
            X_t = transform(X)
            X_rec, _ = model_AE(X_t, score_types)

            for i in range(X.size(0)):
                orig.append(X_cpu[i, 0].numpy())
                noisy.append(X_t[i, 0].cpu().numpy())
                recon.append(X_rec[i, 0].cpu().numpy())
                if len(orig) >= n_vis:
                    return orig, noisy, recon
    return orig, noisy, recon  # fallback when dataset < n_vis


def _make_grid(orig, noisy, recon):
    """Return a Matplotlib figure with rows=[orig,noisy,recon]."""
    n_vis = len(orig)
    fig, axes = plt.subplots(n_vis, 3, figsize=(6, 2 * n_vis))
    titles = ["original", "noisy", "reconstruction"]

    for r in range(n_vis):
        for c, img in enumerate((orig[r], noisy[r], recon[r])):
            ax = axes[r, c] if n_vis > 1 else axes[c]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if r == 0:
                ax.set_title(titles[c])
    fig.tight_layout()
    return fig


def plot_reconstructions(
    dataloaders: Dict[str, Dict[str, torch.utils.data.DataLoader]],
    model_AE: torch.nn.Module,
    transform: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    *,
    prefix: str = "",
    n_vis_max: int = 5,
) -> None:

    for name, dl in dataloaders.items():
        try:
            n_vis = min(n_vis_max, len(dl.dataset))
            orig, noisy, recon = _collect_examples(
                dl, model_AE, transform, device, n_vis
            )

            if not orig:
                print(f"[WARN] {name}: no data available; skipping.")
                continue

            fig = _make_grid(orig, noisy, recon)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_fig:
                fig.savefig(tmp_fig.name, dpi=150)
                artifact_path = f"plots/{prefix}_{name}_reconstructions"
                mlflow.log_artifact(tmp_fig.name, artifact_path=artifact_path)
            plt.close(fig)
            os.remove(tmp_fig.name)

        except Exception as exc:  # noqa: BLE001 (broad ≥ log + carry on)
            print(f"Reconstruction plotting failed for {name}: {exc}")


def evaluate_and_log_testset_results_AE_v2(
    model_AE:               torch.nn.Module,
    model_classifier:       Optional[torch.nn.Module],
    dataloaders:            Dict[str, torch.utils.data.DataLoader],
    *,
    loss_fn_x: Optional[torch.nn.Module] = None,
    loss_fn_y: Optional[torch.nn.Module] = None,
    loss_fn_z: Optional[torch.nn.Module] = None,
    device:    str  = "cuda",
    classes:   Optional[List[str]] = None,
    transform            = lambda x: x,
    prefix:     str  = "test_",
    lambda_x:   float = 1.0,
    lambda_y:   float = 1.0,
    lambda_z:   float = 1.0,
    skip_metrics_logging: bool = False
):
    """
    Run *val_epoch_AE_v1* on the test-set, then

    1. log global metrics (existing behaviour),
    2. build + log classification reports
       • by **score_type**
       • by **classifier head** (if available),
    3. log losses and artefacts (npz + recon plots).
    """

    # ------------------------------------------------------------------ #
    # 0) Forward pass through the whole test set
    # ------------------------------------------------------------------ #
    metrics, outputs_all = val_epoch_AE_v2(
        model_AE,
        dataloaders,
        model_classifier=model_classifier,
        loss_fn_x=loss_fn_x,
        loss_fn_y=loss_fn_y,
        loss_fn_z=loss_fn_z,
        lambda_x=lambda_x,
        lambda_y=lambda_y,
        lambda_z=lambda_z,
        transform=transform,
        device=device,
        classes=classes,
        return_all_predictions=True,
        calc_ICC3=3
    )

    # ------------------------------------------------------------------ #
    # 1) Log global metrics (loss, acc, …)
    # ------------------------------------------------------------------ #
    if not skip_metrics_logging:
        log_metrics_mlflow(
            metrics,
            prefix=prefix,
            classes=classes,
            step=None,
            log_report_and_confusion_matrix_as_artifact=True,
        )

    # ------------------------------------------------------------------ #
    # 2) Per-score-type & per-head classification reports
    # ------------------------------------------------------------------ #
    # --- extract predictions ------------------------------------------ #
    score_types = outputs_all.get("score_type")
    y_true      = outputs_all["labels"]
    y_pred      = outputs_all["preds"]

    if score_types is not None:
        score_types = np.asarray(score_types)

        # ---- per score_type ------------------------------------------- #
        by_st = {}
        for st in np.unique(score_types):
            idx = score_types == st
            by_st[st] = classification_report(
                y_true[idx], y_pred[idx],
                labels=list(range(len(classes))) if classes else None,
                target_names=classes if classes else None,
                output_dict=True, zero_division=0.0,
            )

        mlflow.log_dict(
            by_st, f"metrics/{prefix}classification_report_by_score_type.json"
        )

        # ---- per head (needs classifier) ------------------------------ #
        if model_classifier is not None:
            st2head = make_score_type_2_head_name_dct(
                model_classifier.classifier_head_infos
            )
            heads = np.vectorize(st2head.get)(score_types)

            by_head: Dict[str, Dict] = {}
            for h in np.unique(heads):
                idx = heads == h
                by_head[h] = classification_report(
                    y_true[idx], y_pred[idx],
                    labels=list(range(len(classes))) if classes else None,
                    target_names=classes if classes else None,
                    output_dict=True, zero_division=0.0,
                )
            mlflow.log_dict(
                by_head, f"metrics/{prefix}classification_report_by_head.json"
            )

            # Extract and log macro-average metrics per head
            for head_name, report in by_head.items():
                if "macro avg" in report:
                    macro_metrics = report["macro avg"]
                    mlflow.log_metric(f"{prefix}{head_name}_macro_precision", macro_metrics["precision"])
                    mlflow.log_metric(f"{prefix}{head_name}_macro_recall", macro_metrics["recall"])
                    mlflow.log_metric(f"{prefix}{head_name}_macro_f1", macro_metrics["f1-score"])


    # ------------------------------------------------------------------ #
    # 3) Log reconstruction / latent / per-head losses
    # ------------------------------------------------------------------ #
    if not skip_metrics_logging:
        extra_loss = {k: v for k, v in metrics.items() if k.startswith("L")}
        _log_scalar_dict(prefix, extra_loss, step=None)

    # ------------------------------------------------------------------ #
    # 4) Save raw preds + extras as compressed .npz
    # ------------------------------------------------------------------ #
    out_fn = os.path.join(tempfile.gettempdir(), f"scores.npz")
    np.savez_compressed(out_fn, **outputs_all)
    mlflow.log_artifact(out_fn, artifact_path=f"predictions/{prefix}")
    os.remove(out_fn)
    # with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
    #     np.savez_compressed(tmp.name, **outputs_all)
    #     mlflow.log_artifact(tmp.name, artifact_path=f"predictions/{prefix}")
    #     npz_path = tmp.name
    # os.remove(npz_path)

    # ------------------------------------------------------------------ #
    # 5) (Optional) plot reconstructions
    # ------------------------------------------------------------------ #
    plot_reconstructions(
        dataloaders,
        model_AE,
        transform,
        device,
        prefix = prefix,
        n_vis_max = 6,
    )
    # ------------------------------------------------------------------ #
    return metrics, outputs_all
