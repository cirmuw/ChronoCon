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

from ra_utils.training.scores_SHS.scores_SHS_training_lib_AE_v1 import (
    plot_reconstructions,
    plot_reconstructions_by_type,
    log_scalar_dict
)

import ra_utils.loss.online_mining_delta_loss
import ra_utils.networks.score_estimator
import random
import datetime

import os, tempfile, time, uuid, datetime
import mlflow
import torch

def _retry(fun, tries=3, delay=0.5, backoff=2.0):
    last = None
    for i in range(tries):
        try:
            return fun()
        except Exception as e:
            last = e
            if i == tries - 1:
                raise
            time.sleep(delay)
            delay *= backoff

def _log_single_state_dict_atomic(model, nice_basename, artifact_subdir="checkpoints"):
    """
    Saves `model.state_dict()` to a unique temp dir using `nice_basename`
    and logs it to MLflow, then cleans up the temp dir automatically.
    """
    with tempfile.TemporaryDirectory(prefix=f"ckpt_{os.getpid()}_") as tmpdir:
        tmp_path = os.path.join(tmpdir, nice_basename)
        # Save
        torch.save(model.state_dict(), tmp_path)
        os.sync() if hasattr(os, "sync") else None  # best-effort flush on Linux
        # Log with retry (helps with flaky filesystems)
        _retry(lambda: mlflow.log_artifact(tmp_path, artifact_path=artifact_subdir))

def _log_text_atomic(text, nice_basename="checkpoint_info.yaml", artifact_subdir="checkpoints"):
    with tempfile.TemporaryDirectory(prefix=f"ckptmeta_{os.getpid()}_") as tmpdir:
        tmp_path = os.path.join(tmpdir, nice_basename)
        with open(tmp_path, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        _retry(lambda: mlflow.log_artifact(tmp_path, artifact_path=artifact_subdir))



def train_loop_AE_v4(
    model_AE: torch.nn.Module,
    model_classifier: Optional[torch.nn.Module],
    train_dataloaders: Dict[str, torch.utils.data.DataLoader],
    val_loaders: Dict[str, torch.utils.data.DataLoader],
    loss_fn_dict: dict, 
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cuda",
    # training config
    epochs: int = 1000,
    patience: int = 10,
    run_full_epochs: bool = False,
    transform=lambda x: x,
    classes: Optional[List[str]] = None,
    log_model_full: bool = False,
    log_model_state_dct: bool = False,
    verbose: bool = True,
    ES_metric_key = "L",  # which metric to use for early stopping
    append_BEST_VAL_as_last: bool = False,
    task_type_y: Literal["classification","regression"]="classification"
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
        train_metrics_dct = training_epoch_AE_v4(
            model_AE,
            optimizer,
            train_dataloaders,
            model_classifier=model_classifier,
            loss_fn_dict=loss_fn_dict,
            transform=transform,
            device=device,
            task_type_y=task_type_y
        )
        metrics_Tr.append(train_metrics_dct)

        # log training scalar losses
        log_scalar_dict("train_", train_metrics_dct, step=epoch)

        # ------------------------------------------------------------------
        # 2) Validation ------------------------------------------------------
        val_metrics_dct = val_epoch_AE_v4(
            model_AE,
            val_loaders,
            model_classifier=model_classifier,
            loss_fn_dict=loss_fn_dict,
            transform=lambda x: x,  # No denoising in val. loop
            device=device,
            classes=classes,
            return_all_predictions=False,
            calc_ICC3=1,
            task_type_y=task_type_y
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
                f"{key}: {value:.3f}" for key, value in train_metrics_dct.items()
                if ((key != "L") and (value != 0.0))
            )
            val_losses = "  Validation: " + " | ".join(
                f"{key}: {value:.3f}" for key, value in val_metrics_dct.items()
                if (key.startswith("L") and (value != 0.0) and key != "L")
            )
            print(train_losses)
            print(val_losses)

        if verbose > 2:
            keys = list(loss_fn_dict.keys()) 
            loss_keys = [f"L{k}"  for k in keys]
            lambda_vals = [loss_fn_dict[k]["lambda"] for k in keys]

            # build and print the training-relative‐loss line
            train_rel = "     rel. loss contribution Tr.: " + " | ".join(
                f"{k[1:]}: {(w * train_metrics_dct[k] / train_metrics_dct['L']):.2f}"
                for k, w in zip(loss_keys, lambda_vals)
                if ((k in train_metrics_dct)  and (lambda_vals != 0.0))
            )
            print(train_rel)

            # build and print the validation-relative‐loss line
            val_rel = "     rel. loss contribution Val: " + " | ".join(
                f"{k[1:]}: {(w * val_metrics_dct[k] / val_metrics_dct['L']):.2f}"
                for k, w in zip(loss_keys, lambda_vals)
                if ((k in val_metrics_dct)  and (lambda_vals != 0.0))
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

        # also log reconstruction / latent / per-head losses
        extra_loss_keys = {k: v for k, v in val_metrics_dct.items() if k.startswith("L")}
        log_scalar_dict("val_", extra_loss_keys, step=epoch)

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

                # optional MLflow model snapshot (full models)
                if log_model_full:
                    mlflow.pytorch.log_model(model_AE, "best_model_AE")
                    if model_classifier is not None:
                        mlflow.pytorch.log_model(
                            model_classifier, "best_model_classifier"
                        )

                # ------- race-safe, non-accumulating state_dict logging -------
                if log_model_state_dct:
                    # Use fixed basenames so each improvement overwrites previous artifact
                    _log_single_state_dict_atomic(model_AE, "model_AE_state_dict.pt")
                    if model_classifier is not None:
                        _log_single_state_dict_atomic(model_classifier, "model_classifier_state_dict.pt")

                    # Sidecar metadata (also overwrite same name)
                    info_txt = (
                        f"best_epoch: {epoch}\n"
                        f"val_loss: {val_loss_ES:.6f}\n"
                        f"saved_at: {datetime.datetime.now().isoformat()}\n"
                    )
                    _log_text_atomic(info_txt, nice_basename="checkpoint_info.yaml")

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
                    print(f"Early stopping – no improvement for {patience} epochs.")
                    break

    if append_BEST_VAL_as_last: 
        metrics_Tr.append(train_metrics_dct_BEST)
        metrics_Val.append(val_metrics_dct_BEST)

    # ----------------------------------------------------------------------
    # 5) Wrap-up: restore best weights -------------------------------------
    if not run_full_epochs:
        model_AE.load_state_dict(best_AE_state)
        if model_classifier is not None and best_clf_state is not None:
            model_classifier.load_state_dict(best_clf_state)

    return model_AE, model_classifier, metrics_Tr, metrics_Val



######################################################################################
######################################################################################

def val_epoch_AE_v4(
    model_AE: torch.nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    loss_fn_dict: dict,
    model_classifier: Optional[torch.nn.Module] = None,
    transform=lambda x: x,
    device: str = "cuda",
    classes: Optional[List[str]] = None,
    return_all_predictions: bool = False,
    calc_ICC3=0,
    task_type_y: Literal["classification", "regression", "classification_regression_mix"] = "classification",
):
    """
    Difference to v1:
     dataloader -> dataloaders (dict)
     model(img) -> mode(img, score_type)

    Returns
    -------
    metrics : dict
        Scalar losses (L, Lx, Ly, Lz, per-head Ly_*), classical classification
        metrics, confusion matrix and sklearn classification report.
    outputs_all_samples : dict, optional
        Returned only when *return_all_predictions* is *True*.
    """

    # ------------------------------------------------------------------
    # 0) Book-keeping & preparation
    # ------------------------------------------------------------------
    loss_fn_x = loss_fn_dict["x"]["function"];  lambda_x = loss_fn_dict["x"]["lambda"]
    loss_fn_y = loss_fn_dict["y"]["function"];  lambda_y = loss_fn_dict["y"]["lambda"]
    loss_fn_z = loss_fn_dict["z"]["function"];  lambda_z = loss_fn_dict["z"]["lambda"]

    loss_fn_z_triplet_classes = loss_fn_dict["z_triplet_classes"]["function"];  
    lambda_z_triplet_classes  = loss_fn_dict["z_triplet_classes"]["lambda"]
    lambda_z_triplet_classes_val_multiplier  = loss_fn_dict["z_triplet_classes"]["options"].get("validation_multiplier", 1.0)

    loss_fn_z_triplet_WST_score = loss_fn_dict["z_triplet_WST_score"]["function"];  
    lambda_z_triplet_WST_score  = loss_fn_dict["z_triplet_WST_score"]["lambda"]
    lambda_z_triplet_WST_score_val_multiplier  = loss_fn_dict["z_triplet_WST_score"]["options"].get("validation_multiplier", 1.0)

    loss_fn_z_triplet_WST_time = loss_fn_dict["z_triplet_WST_time"]["function"];  
    lambda_z_triplet_WST_time  = loss_fn_dict["z_triplet_WST_time"]["lambda"]
    lambda_z_triplet_WST_time_val_multiplier  = loss_fn_dict["z_triplet_WST_time"]["options"].get("validation_multiplier", 1.0)


    loss_fn_z_CR = loss_fn_dict["z_score_consistency_regularizer"]["function"];  
    lambda_z_CR  = loss_fn_dict["z_score_consistency_regularizer"]["lambda"]
    lambda_z_CR_val_multiplier  = loss_fn_dict["z_score_consistency_regularizer"]["options"].get("validation_multiplier", 1.0)

    loss_fn_y_reg_extra = loss_fn_dict["y_reg_extra"]["function"];  
    lambda_y_reg_extra = loss_fn_dict["y_reg_extra"]["lambda"]

    loss_fn_y_delta = loss_fn_dict["y_delta"]["function"];  
    lambda_y_delta  = loss_fn_dict["y_delta"]["lambda"]

    loss_terms_ = ["x", "y", "z", 
                  # "z_triplet_classes", "z_triplet_WST_score", "z_triplet_WST_time", "z_score_consistency_regularizer", 
                  "y_reg_extra", "y_delta"]
    loss_terms_contrastive_ = ["z_triplet_classes", "z_triplet_WST_score", "z_triplet_WST_time", "z_score_consistency_regularizer"]

    assert set(loss_fn_dict.keys()) == set(loss_terms_ + loss_terms_contrastive_), \
     f"Expected loss functions not found in loss_fn_dict {loss_fn_dict.keys()} or found more"

    model_AE.eval().to(device)
    if model_classifier is not None:
        model_classifier.eval().to(device)
        running_head_loss: defaultdict[str, float] = defaultdict(float)
        head_counts: defaultdict[str, int] = defaultdict(int)

        # Max width across all classifier heads – used to build batch_logits
        max_out_dim = max(
            hinfo["out_dim"] for hinfo in model_classifier.classifier_head_infos.values()
        )
    else:
        running_head_loss = defaultdict(float)
        head_counts = defaultdict(int)
        max_out_dim = 1  # trivial – no classifier


    # running sums for global losses
    running_loss = dict(L=0.0)
    for k in loss_terms_:
        running_loss[f"L{k}"] = 0.0

    # All 
    contr_dct = dict() # for the counts pos, ... of the triplet losses 

    n_samples = 0

    # containers for classification statistics
    all_preds = []
    all_preds_float = []
    all_labels = []
    # collect logits for classification AND mix (in mix: first col is regression, rest are class logits)
    all_logits: list[np.ndarray] | None = [] if task_type_y in ("classification", "classification_regression_mix") else None

    extra_keys = ["file_name", "score_type", "JSN_or_ERO", "extremity", "patient_id"]
    all_extras: Dict[str, list] = {k: [] for k in extra_keys}

    # ------------------------------------------------------------------
    # 1) Main validation loop
    # ------------------------------------------------------------------
    with torch.no_grad():
        for dataloader_name, dataloader in dataloaders.items():
            for batch in dataloader:
                X = batch["img"].to(device)
                X_pos = batch.get("img_pos", None)
                if X_pos is not None:
                    X_pos = X_pos.to(device)  # positive part for triplet loss
                y = batch["score"].to(device)
                s_type = batch["score_type"]  # list[str] – one per sample
                instance_label = np.array(batch["patient_scoretype_key"])
                B = X.size(0)

                # Times and date labels
                years_np = datestr_to_years_since_2000(batch["date_str"])
                days_np = (years_np*365.2425).astype(int)
                days_t = torch.from_numpy(days_np).to(device)
                years_t = torch.from_numpy(years_np).to(device)
                date_str_np = np.array(batch["date_str"])

                # ----------------------------------------------------------
                # 1.1 Autoencoder forward & reconstruction/latent losses
                # ----------------------------------------------------------
                X_pred, z = model_AE(transform(X), s_type)

                if X_pos is not None:
                    _, z_positive = model_AE(X_pos, score_types=s_type)
                else:
                    z_positive = z  # Fall back (Note that in this case lambda should be 0!)

                loss_x = loss_fn_x(X_pred, X)
                loss_z = loss_fn_z(z, z * 0)

                # triplet/class losses
                loss_z_triplet_classes = loss_fn_z_triplet_classes(
                    labels=y, embeddings=z, ids=instance_label, margin_scores=y
                )
                loss_z_triplet_classes, fraction_positive_triplets__classes, number_of_valid_triplets__classes = loss_z_triplet_classes

                loss_z_triplet_WST_scores = loss_fn_z_triplet_WST_score(
                    labels=y, embeddings=z, embeddings_self_transform=z_positive,
                    ids=instance_label, margin_scores=y
                )
                loss_z_triplet_WST_scores, fraction_positive_triplets__WST_classes, number_of_valid_triplets__WST_classes = loss_z_triplet_WST_scores
            

                # with self-transform on TIME
                loss_z_triplet_WST_time = loss_fn_z_triplet_WST_time(labels=days_t,
                                                                    embeddings=z, 
                                                                    embeddings_self_transform = z_positive,
                                                                    ids=instance_label,  # pat001_L_PIPII
                                                                    margin_scores=years_t
                                                                )
                loss_z_triplet_WST_time, fraction_positive_triplets__WST_time, number_of_valid_triplets__WST_time = loss_z_triplet_WST_time


                loss_z_SCR, number_of_valid_contributions_SCR = loss_fn_z_CR(scores=y, embeddings=z, ids=instance_label)

                # ----------------------------------------------------------
                # 1.2 Classification path (if present)
                # ----------------------------------------------------------
                loss_y = torch.tensor(0.0, device=device)
                loss_y_reg_extra = torch.tensor(0.0, device=device)
                loss_y_delta = torch.tensor(0.0, device=device)

                batch_preds_float = torch.empty(B, dtype=torch.float32, device=device)

                if task_type_y == "classification":
                    batch_logits = torch.full((B, max_out_dim), float("-inf"), device=device)
                    batch_preds = torch.empty(B, dtype=torch.long, device=device)
                elif (task_type_y == "regression") or (task_type_y == "classification_regression_mix"):
                    batch_preds = torch.empty(B, dtype=torch.float32, device=device)
                    if task_type_y == "classification_regression_mix":
                        # 1 for regression + max_out_dim for class logits
                        # batch_logits = torch.full((B, 1 + max_out_dim), float("-inf"), device=device)
                        batch_logits = torch.full((B, max_out_dim), float("-inf"), device=device)
                else:
                    raise NotImplementedError(f"{task_type_y = }")

                if model_classifier is not None:
                    out_dict = model_classifier(z, s_type, return_dict=True)

                    for head_name, (idx, logits) in out_dict.items():
                        cnt = idx.numel()
                        if cnt == 0:
                            continue

                        # ---- loss ------------------------------------------------
                        head_target = y[idx]
                        head_instance_label = instance_label[idx.cpu().numpy()]

                        if task_type_y == "regression":
                            loss_fn_y_input = logits.squeeze(-1)
                            score_estimation = loss_fn_y_input
                            head_target = head_target.float().to(device)

                        if task_type_y == "classification":
                            loss_fn_y_input = logits
                            score_estimation = ra_utils.networks.score_estimator.estimate_scalar_score_from_logits(
                                logits, mode="expectation_value"
                            )

                        if task_type_y == "classification_regression_mix":
                            loss_fn_y_input = logits
                            score_estimation_1 = loss_fn_y_input[..., 0]  # regression value
                            score_estimation_2 = ra_utils.networks.score_estimator.estimate_scalar_score_from_logits(
                                logits[..., 1:], mode="expectation_value"
                            )
                            score_estimation = loss_fn_y.mse_weight * score_estimation_1 + (1 - loss_fn_y.mse_weight) * score_estimation_2

                        head_loss = loss_fn_y(loss_fn_y_input, head_target)
                        head_loss_y_extra = loss_fn_y_reg_extra(score_estimation, head_target.float().to(device))
                        head_loss_y_delta = loss_fn_y_delta(score_estimation, head_target.float().to(device), head_instance_label)

                        w = model_classifier.classifier_head_infos[head_name].get("loss_weight", 1.0)
                        loss_y += head_loss * cnt * w
                        loss_y_reg_extra += head_loss_y_extra * cnt * w
                        loss_y_delta += head_loss_y_delta * cnt * w

                        # per-head running sums (raw, un-weighted)
                        running_head_loss[head_name] += head_loss.item() * cnt
                        head_counts[head_name] += cnt

                        # ---- store preds & logits ------------------------------
                        if task_type_y == "classification":
                            batch_logits[idx, : logits.shape[1]] = logits
                            batch_preds[idx] = logits.argmax(dim=1)
                        elif task_type_y == "regression":
                            batch_preds[idx] = logits.squeeze(-1)  # these are not logits
                        elif task_type_y == "classification_regression_mix":
                            # save both regression (col0) and class logits (col1..)
                            batch_logits[idx, : logits.shape[1]] = logits
                            batch_preds[idx] = score_estimation
                        else:
                            raise NotImplementedError(f"{task_type_y = }")

                        batch_preds_float[idx] = score_estimation

                    # convert Σ(loss*count) ➜ mean per sample of *whole batch*
                    loss_y = loss_y / B
                    loss_y_reg_extra = loss_y_reg_extra / B
                    loss_y_delta = loss_y_delta / B

                else:
                    # no classifier – leave loss_y = 0, predictions dummy-filled
                    if task_type_y == "classification":
                        batch_logits = torch.full((B, max_out_dim), float("-inf"), device=device)
                        batch_logits.zero_()
                        batch_preds.fill_(0)
                    elif task_type_y == "regression":
                        batch_preds.fill_(0)
                    elif task_type_y == "classification_regression_mix":
                        batch_logits = torch.full((B, max_out_dim), float("-inf"), device=device)
                        batch_logits.zero_()
                        batch_preds.fill_(0)
                    else:
                        raise NotImplementedError(f"{task_type_y = }")
                    batch_preds_float.fill_(0.0)

                # ----------------------------------------------------------
                # 1.3 Total weighted loss & running sums
                # ----------------------------------------------------------
                loss_total = (
                    lambda_x * loss_x
                    + lambda_y * loss_y
                    + lambda_z * loss_z
                    # + lambda_z_triplet_classes * loss_z_triplet_classes * lambda_z_triplet_classes_val_multiplier
                    # + lambda_z_CR * loss_z_SCR * lambda_z_CR_val_multiplier
                    # + lambda_z_triplet_WST_score * loss_z_triplet_WST_scores * lambda_z_triplet_WST_score_val_multiplier
                    # + lambda_z_triplet_WST_time * loss_z_triplet_WST_time * lambda_z_triplet_WST_time_val_multiplier
                    + lambda_y_delta * loss_y_delta
                    + lambda_y_reg_extra * loss_y_reg_extra
                )

                running_loss["Lx"] += loss_x.item() * B
                running_loss["Ly"] += loss_y.item() * B
                running_loss["Lz"] += loss_z.item() * B
                running_loss["Ly_delta"] += loss_y_delta.item() * B
                running_loss["Ly_reg_extra"] += loss_y_reg_extra.item() * B
                running_loss["L"] += loss_total.item() * B
                n_samples += B

                # running_loss["Lz_triplet_classes"] += loss_z_triplet_classes.item() * B
                # running_loss["Lz_triplet_WST_score"] += loss_z_triplet_WST_scores.item() * B
                # running_loss["Lz_triplet_WST_time"] += loss_z_triplet_WST_time.item() * B
                # running_loss["Lz_score_consistency_regularizer"] += loss_z_SCR.item() * B

                # Add fraction of positive triplets and number of valid triplets to running loss
                N_pos = number_of_valid_triplets__classes.item() * fraction_positive_triplets__classes.item()
                l = loss_z_triplet_classes.item()
                contr_dct["Lz_TriCls"] = contr_dct.get("Lz_TriCls", 0.0) + (l * N_pos)
                contr_dct["Lz_TriCls_numPosTrip"]   = contr_dct.get("Lz_TriCls_numPosTrip", 0.0) + N_pos
                contr_dct["Lz_TriCls_fracPosTrip"]  = contr_dct.get("Lz_TriCls_fracPosTrip", 0.0) + fraction_positive_triplets__classes.item() * B
                contr_dct["Lz_TriCls_numValidTrip"] = contr_dct.get("Lz_TriCls_numValidTrip", 0.0) + number_of_valid_triplets__classes.item() * B

                N_pos = number_of_valid_triplets__classes.item() * fraction_positive_triplets__WST_classes.item()
                l = loss_z_triplet_WST_scores.item()
                contr_dct["Lz_TriClsWST"] = contr_dct.get("Lz_TriClsWST", 0.0) + (l * N_pos)
                contr_dct["Lz_TriClsWST_numPosTrip"]   = contr_dct.get("Lz_TriClsWST_numPosTrip", 0.0) + N_pos            
                contr_dct["Lz_TriClsWST_fracPosTrip"]  = contr_dct.get("Lz_TriClsWST_fracPosTrip", 0.0) + fraction_positive_triplets__WST_classes.item() * B
                contr_dct["Lz_TriClsWST_numValidTrip"] = contr_dct.get("Lz_TriClsWST_numValidTrip", 0.0) + number_of_valid_triplets__WST_classes.item() * B


                N_pos = number_of_valid_triplets__WST_time.item() * fraction_positive_triplets__WST_time.item()
                l = loss_z_triplet_WST_time.item()
                contr_dct["Lz_TriTimeWST"] = contr_dct.get("Lz_TriTimeWST", 0.0) + (l * N_pos)
                contr_dct["Lz_TriTimeWST_numPosTrip"]   = contr_dct.get("Lz_TriTimeWST_numPosTrip", 0.0) + N_pos
                contr_dct["Lz_TriTimeWST_fracPosTrip"]  = contr_dct.get("Lz_TriTimeWST_fracPosTrip", 0.0) + fraction_positive_triplets__WST_time.item() * B
                contr_dct["Lz_TriTimeWST_numValidTrip"] = contr_dct.get("Lz_TriTimeWST_numValidTrip", 0.0) + number_of_valid_triplets__WST_time.item() * B                    

                N_pos = number_of_valid_contributions_SCR.item()
                l = loss_z_SCR.item()
                contr_dct["Lz_SCR"]   = contr_dct.get("Lz_SCR", 0.0) + (N_pos * l) 
                contr_dct["Lz_SCR_numPos"]   = contr_dct.get("Lz_SCR_numPos", 0.0) + N_pos        


                # ----------------------------------------------------------
                # 1.4 Collect per-sample statistics for later metrics
                # ----------------------------------------------------------
                all_preds.extend(batch_preds.cpu().numpy())
                all_preds_float.extend(batch_preds_float.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

                if (model_classifier is not None) and (task_type_y in ("classification", "classification_regression_mix")):
                    all_logits.extend(batch_logits.cpu().numpy())

                for k in extra_keys:
                    if k in batch:
                        all_extras[k].extend(batch[k])

    # ------------------------------------------------------------------
    # 2) Aggregate scalar losses
    # ------------------------------------------------------------------
    for k in running_loss:
        running_loss[k] /= n_samples if n_samples else 1.0

    # Mean per sample for contrastive losses: 
    contr_dct["Lz_TriCls"] /= (contr_dct["Lz_TriCls_numPosTrip"] + 1.0e-10)
    contr_dct["Lz_TriClsWST"] /= (contr_dct["Lz_TriClsWST_numPosTrip"]  + 1.0e-10)
    contr_dct["Lz_TriTimeWST"] /= (contr_dct["Lz_TriTimeWST_numPosTrip"] + 1.0e-10)
    contr_dct["Lz_SCR"] /= (contr_dct["Lz_SCR_numPos"]  + 1.0e-10)



    # The purpose behind the ..._val_multipliers is that the validation uses a different sampler than the training. 
    # So these can not really be compared that well and might need to be turned of 
    running_loss["L"] += (
              contr_dct["Lz_TriCls"]     * lambda_z_triplet_classes   * lambda_z_triplet_classes_val_multiplier
            + contr_dct["Lz_TriClsWST"]  * lambda_z_triplet_WST_score * lambda_z_triplet_WST_score_val_multiplier
            + contr_dct["Lz_TriTimeWST"] * lambda_z_triplet_WST_time  * lambda_z_triplet_WST_time_val_multiplier        
            + contr_dct["Lz_SCR"]        * lambda_z_CR                * lambda_z_CR_val_multiplier
    )

    running_loss = {**running_loss, **contr_dct}

    if model_classifier is not None:
        for head_name in model_classifier.classifier_head_infos:
            cnt = head_counts[head_name]
            running_loss[f"Ly_{head_name}"] = (running_head_loss[head_name] / cnt if cnt else 0.0)

    # ------------------------------------------------------------------
    # 3) Classical classification statistics (global)
    # ------------------------------------------------------------------
    metrics = {**running_loss, "loss": running_loss["L"]}

    all_preds_np = np.array(all_preds)
    all_preds_np_float = np.array(all_preds_float)
    all_labels_np = np.array(all_labels)

    if task_type_y == "classification":
        all_logits_np = np.array(all_logits)  # (N, C)
        all_preds_np_classes = all_preds_np
        all_probs_np = F.softmax(torch.from_numpy(all_logits_np), dim=1).numpy()

    elif task_type_y == "classification_regression_mix":
        all_logits_np = np.array(all_logits)  # (N, 1 + C)  [col0=regression, col1..=class logits]
        class_logits_np = all_logits_np[:, 1:]  # drop regression column for probs
        all_probs_np = F.softmax(torch.from_numpy(class_logits_np), dim=1).numpy()

        # round regression/class-mix predictions for classification metrics
        all_preds_np_classes = np.round(all_preds_np)
        all_preds_np_classes = np.clip(all_preds_np_classes, 0, np.inf)
        if classes is not None:
            all_preds_np_classes = np.clip(all_preds_np_classes, 0, len(classes) - 1)

    else:  # pure regression
        all_logits_np = None
        all_preds_np_classes = np.round(all_preds_np)
        all_preds_np_classes = np.clip(all_preds_np_classes, 0, np.inf)
        if classes is not None:
            all_preds_np_classes = np.clip(all_preds_np_classes, 0, len(classes) - 1)

    all_preds_np_classes = all_preds_np_classes.astype(np.int64)
    all_labels_np = all_labels_np.astype(np.int64)

    metrics_ = calculate_some_classification_metrics(
        all_preds_np_classes, all_labels_np, calc_ICC3=calc_ICC3,
    )
    metrics.update(metrics_)

    if (task_type_y == "regression") or (task_type_y == "classification_regression_mix"):
        # regression-style metrics on the raw (non-rounded) predictions
        regression_metrics = calculate_some_classification_metrics(
            all_preds_np, all_labels_np, calc_ICC3=calc_ICC3,
            add_classification_metrics=False, add_spearman=True, add_kappa=False,
        )
        metrics.update(regression_metrics)

    if task_type_y == "classification":
        # ---- top-2 accuracy --------------------------------------------------
        top2 = np.argsort(all_probs_np, axis=1)[:, -2:]
        top2_correct = sum(label in top2[i] for i, label in enumerate(all_labels_np))
        metrics["top2_accuracy"] = top2_correct / len(all_labels_np)

    # ---- confusion matrix & classification report -----------------------
    cm = confusion_matrix(all_labels_np, all_preds_np_classes)
    metrics["confusion_matrix"] = cm.tolist()

    if classes is not None:
        labels = list(range(len(classes)))
        report = classification_report(
            all_labels_np, all_preds_np_classes, labels=labels, target_names=classes,
            output_dict=True, zero_division=0.0,
        )
    else:
        report = classification_report(
            all_labels_np, all_preds_np_classes, output_dict=True, zero_division=0.0
        )
    metrics["classification_report"] = report

    # ------------------------------------------------------------------
    # 4) Optionally return per-sample outputs
    # ------------------------------------------------------------------
    if return_all_predictions:
        outputs_all_samples = {
            "labels": all_labels_np,
            "preds": all_preds_np_classes, 
            "preds_float": all_preds_np_float,
            **all_extras,
        }
        if task_type_y in ("classification", "classification_regression_mix"):
            outputs_all_samples["logits"] = all_logits_np  # in mix: includes regression in col0
            outputs_all_samples["probs"] = all_probs_np    # in mix: probs from class logits only
        return metrics, outputs_all_samples

    return metrics


# def val_epoch_AE_v4(
#     model_AE: torch.nn.Module,
#     dataloaders: Dict[str, torch.utils.data.DataLoader],
#     loss_fn_dict: dict, 
#     model_classifier: Optional[torch.nn.Module] = None,
#     transform=lambda x: x,
#     device: str = "cuda",
#     classes: Optional[List[str]] = None,
#     return_all_predictions: bool = False,
#     calc_ICC3=0,
#     task_type_y: Literal["classification","regression", "classification_regression_mix"]="classification"
# ):
#     """
#     Difference to v1: 
#      dataloader -> dataloaders (dict)
#      model(img) -> mode(img, score_type)

#     Returns
#     -------
#     metrics : dict
#         Scalar losses (L, Lx, Ly, Lz, per‑head Ly_*), classical classification
#         metrics, confusion matrix and sklearn classification report.
#     outputs_all_samples : dict, optional
#         Returned only when *return_all_predictions* is *True*.
#     """

#     # ------------------------------------------------------------------
#     # 0) Book‑keeping & preparation
#     # ------------------------------------------------------------------
#     loss_fn_x = loss_fn_dict["x"]["function"];  lambda_x = loss_fn_dict["x"]["lambda"]
#     loss_fn_y = loss_fn_dict["y"]["function"];  lambda_y = loss_fn_dict["y"]["lambda"]
#     loss_fn_z = loss_fn_dict["z"]["function"];  lambda_z = loss_fn_dict["z"]["lambda"]
    
#     loss_fn_z_triplet_classes = loss_fn_dict["z_triplet_classes"]["function"];  
#     lambda_z_triplet_classes  = loss_fn_dict["z_triplet_classes"]["lambda"]

#     loss_fn_z_triplet_WST_score = loss_fn_dict["z_triplet_WST_score"]["function"];  
#     #lambda_z_triplet_WST_score  = loss_fn_dict["z_triplet_WST_score"]["lambda"]
#     lambda_z_triplet_WST_score = 0.0 # Since dataloader for val does not contain self transforms I simply skip... 

#     loss_fn_z_CR = loss_fn_dict["z_score_consistency_regularizer"]["function"];  
#     lambda_z_CR  = loss_fn_dict["z_score_consistency_regularizer"]["lambda"]

#     loss_fn_y_reg_extra = loss_fn_dict["y_reg_extra"]["function"];  
#     lambda_y_reg_extra = loss_fn_dict["y_reg_extra"]["lambda"]

#     loss_fn_y_delta = loss_fn_dict["y_delta"]["function"];  
#     lambda_y_delta  = loss_fn_dict["y_delta"]["lambda"]

#     loss_terms_ = ["x", "y", "z", 
#                   "z_triplet_classes", "z_triplet_WST_score", "z_score_consistency_regularizer", 
#                   "y_reg_extra", "y_delta"]
#     assert set(loss_fn_dict.keys()) == set(loss_terms_), \
#      f"Expected loss functions not found in loss_fn_dict {loss_fn_dict.keys()} or found more"
    

#     model_AE.eval().to(device)
#     if model_classifier is not None:
#         model_classifier.eval().to(device)
#         running_head_loss: defaultdict[str, float] = defaultdict(float)
#         head_counts: defaultdict[str, int] = defaultdict(int)

#         # Max width across all classifier heads – used to build batch_logits
#         max_out_dim = max(
#             hinfo["out_dim"]
#             for hinfo in model_classifier.classifier_head_infos.values()
#         )
#     else:
#         running_head_loss = defaultdict(float)
#         head_counts = defaultdict(int)
#         max_out_dim = 1  # trivial – no classifier

#     # running_loss = dict(Lx=0.0, Ly=0.0, Lz=0.0, L=0.0, Lz_triplet_classes=0.0, Ly_delta=0.0, Ly_reg_extra=0.0)
#     loss_dct_keys = list(loss_fn_dict.keys())
#     running_loss = dict(L=0.0)
#     for k in loss_dct_keys:
#         running_loss[f"L{k}"] = 0.0

#     n_samples = 0

#     # containers for classification statistics
#     all_preds = []
#     all_preds_float = []
#     all_labels = []
#     all_logits: list[np.ndarray] | None = [] if task_type_y in ("classification", "classification_regression_mix") else None

#     extra_keys = [
#         "file_name",
#         "score_type",
#         "JSN_or_ERO",
#         "extremity",
#         "patient_id",
#     ]
#     all_extras: Dict[str, list] = {k: [] for k in extra_keys}

#     # ------------------------------------------------------------------
#     # 1) Main validation loop
#     # ------------------------------------------------------------------
#     with torch.no_grad():
#         for dataloader_name, dataloader in dataloaders.items():
#             for batch in dataloader:
#                 X = batch["img"].to(device)
#                 X_pos  = batch.get("img_pos", None)
#                 if X_pos is not None: 
#                     X_pos  = X_pos.to(device) # positive part for triplet loss
#                 y = batch["score"].to(device)
#                 s_type = batch["score_type"]  # list[str] – one per sample
#                 instance_label = np.array(batch["patient_scoretype_key"])
#                 B = X.size(0)

#                 # ----------------------------------------------------------
#                 # 1.1 Autoencoder forward & reconstruction/latent losses
#                 # ----------------------------------------------------------
#                 X_pred, z = model_AE(transform(X), s_type)

#                 if X_pos is not None: 
#                     _, z_positive  = model_AE(X_pos, score_types=s_type)   # Could add transform(X_pos)
#                 else: 
#                     z_positive = z  # Fall back (Note that in this case lambda should be 0!)
#                     # Note that triplet loss with self-transform does not really work without transforms (in the val loop...)
#                     # assert lambda_z_triplet_WST_score == 0.0

#                 loss_x = loss_fn_x(X_pred, X)
#                 loss_z = loss_fn_z(z, z * 0)

#                 # triplet loss on classes: 
#                 loss_z_triplet_classes = ( 
#                     loss_fn_z_triplet_classes(labels=y, embeddings=z, 
#                     ids=instance_label,  # Use None for ablation study!
#                     margin_scores=y # score dependent margin
#                     ))
#                 loss_z_triplet_classes, fraction_positive_triplets__classes, number_of_valid_triplets__classes = loss_z_triplet_classes

#                 loss_z_triplet_WST_scores = loss_fn_z_triplet_WST_score(labels=y, 
#                                                                                 embeddings=z, 
#                                                                                 embeddings_self_transform = z_positive,
#                                                                                 ids=instance_label,  # Use None for ablation study!
#                                                                                 margin_scores=y # score dependent margin
#                                                                 )
#                 loss_z_triplet_WST_scores, fraction_positive_triplets__WST_classes, number_of_valid_triplets__WST_classes = loss_z_triplet_WST_scores
                
#                 loss_z_CRL = loss_fn_z_CR(scores=y, embeddings=z, ids=instance_label)


#                 # ----------------------------------------------------------
#                 # 1.2 Classification path (if present)
#                 # ----------------------------------------------------------
#                 loss_y = torch.tensor(0.0, device=device)
#                 loss_y_reg_extra = torch.tensor(0.0, device=device)
#                 loss_y_delta = torch.tensor(0.0, device=device)

#                 batch_preds_float = torch.empty(B, dtype=torch.float32, device=device)   

#                 if task_type_y == "classification":
#                     batch_logits = torch.full((B, max_out_dim), float("-inf"), device=device)
#                     batch_preds = torch.empty(B, dtype=torch.long, device=device)
#                 elif (task_type_y == "regression") or (task_type_y == "classification_regression_mix"):
#                     batch_preds = torch.empty(B, dtype=torch.float32, device=device)                
#                 else: 
#                     raise NotImplementedError(f"{task_type_y = }")

#                 if model_classifier is not None:
#                     out_dict = model_classifier(z, s_type, return_dict=True)

#                     for head_name, (idx, logits) in out_dict.items():
#                         cnt = idx.numel()
#                         if cnt == 0:
#                             continue

#                         # ---- loss ------------------------------------------------
#                         head_target = y[idx]
#                         head_instance_label = instance_label[idx.cpu().numpy()]
#                         if task_type_y == "regression":
#                             loss_fn_y_input = logits.squeeze(-1) #logits[:,0]
#                             score_estimation = loss_fn_y_input                              
#                             head_target = head_target.float().to(device)
#                         if task_type_y == "classification":
#                             loss_fn_y_input = logits
#                             score_estimation = ra_utils.networks.score_estimator.estimate_scalar_score_from_logits(logits, mode="expectation_value")                              
#                         if task_type_y == "classification_regression_mix": 
#                             #raise NotImplementedError("task_type_y == 'classification_regression_mix'")
#                             loss_fn_y_input = logits
#                             score_estimation_1 = loss_fn_y_input[..., 0]  #  These are not logits! First element is assumed to be the reg. value
#                             score_estimation_2 = ra_utils.networks.score_estimator.estimate_scalar_score_from_logits(logits[..., 1:], mode="expectation_value")
#                             score_estimation = loss_fn_y.mse_weight * score_estimation_1 + (1 - loss_fn_y.mse_weight) * score_estimation_2
#                             # head_target_float = head_target.float().to(device)


#                         head_loss   = loss_fn_y(loss_fn_y_input, head_target)   # mean over head-batch
#                         head_loss_y_extra = loss_fn_y_reg_extra(score_estimation, head_target.float().to(device))
#                         head_loss_y_delta = loss_fn_y_delta(score_estimation, head_target.float().to(device), head_instance_label)
                        
                        
#                         w = model_classifier.classifier_head_infos[head_name].get(
#                             "loss_weight", 1.0
#                         )
#                         loss_y += head_loss * cnt * w  # Σ (loss_i * w)
#                         loss_y_reg_extra += head_loss_y_extra * cnt * w
#                         loss_y_delta += head_loss_y_delta * cnt * w


#                         # per‑head running sums (raw, un‑weighted)
#                         running_head_loss[head_name] += head_loss.item() * cnt
#                         head_counts[head_name] += cnt

#                         # ---- store preds & logits ------------------------------
#                         if task_type_y == "classification":
#                             batch_logits[idx, : logits.shape[1]] = logits
#                             batch_preds[idx] = logits.argmax(dim=1)
#                         elif task_type_y == "regression":
#                             batch_preds[idx] = logits.squeeze(-1) # these are not logits
#                         elif task_type_y == "classification_regression_mix":
#                             batch_preds[idx] = score_estimation  # Should I save integer preds instead?
#                         else: 
#                             raise NotImplementedError(f"{task_type_y = }")
#                         batch_preds_float[idx] = score_estimation

#                     # convert Σ(loss*count) ➜ mean per sample of *whole batch*
#                     loss_y = loss_y / B
#                     loss_y_reg_extra = loss_y_reg_extra / B
#                     loss_y_delta = loss_y_delta / B
#                 else:
#                     # no classifier – leave loss_y = 0, predictions dummy‑filled
#                     if task_type_y == "classification":
#                         batch_logits.zero_()
#                         batch_preds.fill_(0)
#                     elif task_type_y == "regression":
#                         batch_preds.fill_(0)
#                     else: 
#                         raise NotImplementedError(f"{task_type_y = }")
#                     batch_preds_float.fill_(0.0)

#                 # ----------------------------------------------------------
#                 # 1.3 Total weighted loss & running sums
#                 # ----------------------------------------------------------
#                 loss_total = (lambda_x * loss_x + 
#                               lambda_y * loss_y + 
#                               lambda_z * loss_z + 
#                               lambda_z_triplet_classes * loss_z_triplet_classes + 
#                               lambda_z_CR * loss_z_CRL + 
#                               lambda_z_triplet_WST_score * loss_z_triplet_WST_scores +                               
#                               lambda_y_delta * loss_y_delta + 
#                               lambda_y_reg_extra * loss_y_reg_extra
#                              )                               

#                 running_loss["Lx"] += loss_x.item() * B
#                 running_loss["Ly"] += loss_y.item() * B
#                 running_loss["Lz"] += loss_z.item() * B
#                 running_loss["Lz_triplet_classes"] += loss_z_triplet_classes.item() * B
#                 running_loss["Lz_triplet_WST_score"] += loss_z_triplet_WST_scores.item() * B
#                 running_loss["Lz_score_consistency_regularizer"] += loss_z_CRL.item() * B
#                 running_loss["Ly_delta"] += loss_y_delta.item() * B
#                 running_loss["Ly_reg_extra"] += loss_y_reg_extra.item() * B

#                 running_loss["L"] += loss_total.item() * B
#                 n_samples += B

#                 # ----------------------------------------------------------
#                 # 1.4 Collect per‑sample statistics for later metrics
#                 # ----------------------------------------------------------
#                 all_preds.extend(batch_preds.cpu().numpy())
#                 all_preds_float.extend(batch_preds_float.cpu().numpy())
#                 all_labels.extend(y.cpu().numpy())                        
#                 if task_type_y == "classification":
#                     all_logits.extend(batch_logits.cpu().numpy())
#                 if task_type_y == "classification_regression_mix":
#                     all_logits.extend(batch_logits.cpu().numpy())  # I could also just save probs... 

#                 for k in extra_keys:
#                     if k in batch:
#                         all_extras[k].extend(batch[k])

#     # ------------------------------------------------------------------
#     # 2) Aggregate scalar losses
#     # ------------------------------------------------------------------
#     for k in running_loss:
#         running_loss[k] /= n_samples if n_samples else 1.0

#     if model_classifier is not None:
#         for head_name in model_classifier.classifier_head_infos:
#             cnt = head_counts[head_name]
#             running_loss[f"Ly_{head_name}"] = (
#                 running_head_loss[head_name] / cnt if cnt else 0.0
#             )

#     # ------------------------------------------------------------------
#     # 3) Classical classification statistics (global)
#     # ------------------------------------------------------------------
#     metrics = {
#         **running_loss,
#         "loss": running_loss["L"],  # compatibility with legacy logger
#     }

#     all_preds_np = np.array(all_preds)
#     all_preds_np_float = np.array(all_preds_float)
#     all_labels_np = np.array(all_labels)
     

#     if task_type_y == "classification":
#         all_logits_np = np.array(all_logits)
#         all_preds_np_classes = all_preds_np
#         all_probs_np = F.softmax(torch.from_numpy(all_logits_np), dim=1).numpy()

#     elif task_type_y == "classification_regression_mix":
#         all_logits_np = np.array(all_logits)      # shape (N, 1+C)
#         class_logits_np = all_logits_np[:, 1:]    # drop regression column
#         all_probs_np = F.softmax(torch.from_numpy(class_logits_np), dim=1).numpy()

#         # round regression/class-mix predictions for classification metrics
#         all_preds_np_classes = np.round(all_preds_np)
#         all_preds_np_classes = np.clip(all_preds_np_classes, 0, np.inf)
#         if classes is not None:
#             all_preds_np_classes = np.clip(all_preds_np_classes, 0, len(classes)-1)

#     else:  # pure regression
#         all_logits_np = None
#         all_preds_np_classes = np.round(all_preds_np)
#         all_preds_np_classes = np.clip(all_preds_np_classes, 0, np.inf)
#         if classes is not None:
#             all_preds_np_classes = np.clip(all_preds_np_classes, 0, len(classes)-1)

#     all_preds_np_classes = all_preds_np_classes.astype(np.int64)
#     all_labels_np        = all_labels_np.astype(np.int64)

    
#     metrics_ = calculate_some_classification_metrics(all_preds_np_classes, 
#                                                      all_labels_np, 
#                                                      calc_ICC3=calc_ICC3, 
#                                                      # add_classification_metrics = (task_type_y=="classification")
#                                                      )
#     metrics.update(metrics_)

    
#     if (task_type_y == "regression") or (task_type_y == "classification_regression_mix") : # to calculate RMSE, ... on non-rounded parts
#         regression_metrics = calculate_some_classification_metrics(
#             all_preds_np,           # raw scores
#             all_labels_np,
#             calc_ICC3=calc_ICC3,
#             add_classification_metrics=False,   # no acc / bal-acc here
#             add_spearman=True,
#             add_kappa=False,
#         )
#         metrics.update(regression_metrics)


#     if task_type_y == "classification":
#         # ---- top‑2 accuracy --------------------------------------------------
#         top2 = np.argsort(all_probs_np, axis=1)[:, -2:]
#         top2_correct = sum(label in top2[i] for i, label in enumerate(all_labels_np))
#         metrics["top2_accuracy"] = top2_correct / len(all_labels_np)


#     # ---- confusion matrix & classification report -----------------------
#     cm = confusion_matrix(all_labels_np, all_preds_np_classes)
#     metrics["confusion_matrix"] = cm.tolist()



#     if classes is not None:
#         labels = list(range(len(classes)))
#         report = classification_report(
#             all_labels_np,
#             all_preds_np_classes,
#             labels=labels,
#             target_names=classes,
#             output_dict=True,
#             zero_division=0.0,
#         )
#     else:
#         report = classification_report(
#             all_labels_np, all_preds_np_classes, output_dict=True, zero_division=0.0
#         )
#     metrics["classification_report"] = report

#     # ------------------------------------------------------------------
#     # 4) Optionally return per‑sample outputs
#     # ------------------------------------------------------------------
#     if return_all_predictions:
#         outputs_all_samples = {
#             "labels": all_labels_np,
#             "preds": all_preds_np,
#             "preds_float": all_preds_np_float,
#             **all_extras,
#         }
#         if task_type_y == "classification":
#             outputs_all_samples["probs"] = all_probs_np
#         return metrics, outputs_all_samples

#     return metrics






######################################################################################
######################################################################################


from ra_utils.utils.utils import datestr_to_years_since_2000


def training_epoch_AE_v4(
    model_AE,
    optimizer,
    dataloaders: dict,
    loss_fn_dict: dict,
    model_classifier=None,
    transform=lambda x: x,
    device="cuda",
    debugging: bool = False, 
    task_type_y: Literal["classification","regression", "classification_regression_mix"]="classification"
):
    """
    Difference to v3: 
     - loss functions are now passed as a dictionary
     - triplet loss with self-transform  
    """
    verbose = True
    
    # ----------------------------- set-up ---------------------------------- #
    model_AE.train().to(device)
    loss_fn_x = loss_fn_dict["x"]["function"];  lambda_x = loss_fn_dict["x"]["lambda"]
    loss_fn_y = loss_fn_dict["y"]["function"];  lambda_y = loss_fn_dict["y"]["lambda"]
    loss_fn_z = loss_fn_dict["z"]["function"];  lambda_z = loss_fn_dict["z"]["lambda"]
    
    loss_fn_z_triplet_classes = loss_fn_dict["z_triplet_classes"]["function"];  
    lambda_z_triplet_classes  = loss_fn_dict["z_triplet_classes"]["lambda"]

    loss_fn_z_triplet_WST_score = loss_fn_dict["z_triplet_WST_score"]["function"];  
    lambda_z_triplet_WST_score  = loss_fn_dict["z_triplet_WST_score"]["lambda"]

    loss_fn_z_triplet_WST_time = loss_fn_dict["z_triplet_WST_time"]["function"];  
    lambda_z_triplet_WST_time  = loss_fn_dict["z_triplet_WST_time"]["lambda"]

    loss_fn_z_CR = loss_fn_dict["z_score_consistency_regularizer"]["function"];  
    lambda_z_CR  = loss_fn_dict["z_score_consistency_regularizer"]["lambda"]

    loss_fn_y_reg_extra = loss_fn_dict["y_reg_extra"]["function"];  
    lambda_y_reg_extra = loss_fn_dict["y_reg_extra"]["lambda"]

    loss_fn_y_delta = loss_fn_dict["y_delta"]["function"];  
    lambda_y_delta  = loss_fn_dict["y_delta"]["lambda"]


    loss_terms_ = ["x", "y", "z", 
                  # "z_triplet_classes", "z_triplet_WST_score", "z_triplet_WST_time", "z_score_consistency_regularizer", 
                  "y_reg_extra", "y_delta"]
    loss_terms_contrastive_ = ["z_triplet_classes", "z_triplet_WST_score", "z_triplet_WST_time", "z_score_consistency_regularizer"]

    # Checks
    assert set(loss_fn_dict.keys()) == set(loss_terms_ + loss_terms_contrastive_), \
     f"Expected loss functions not found in loss_fn_dict {loss_fn_dict.keys()} or found more"

    if model_classifier is not None:
        model_classifier.train().to(device)
        # running sums for each head
        running_head_loss = defaultdict(float)  # Σ loss*count
        head_counts = defaultdict(int)          # Σ count

    # running sums for global losses
    running_loss = dict(L=0.0)
    for k in loss_terms_:
        running_loss[f"L{k}"] = 0.0

    # All 
    contr_dct = dict() # for the counts pos, ... of the triplet losses 

    samples_tr = 0

    # ---------------------------- epoch loop ------------------------------- #
    # Randomize dataloader order for better training
    dataloader_items = list(dataloaders.items())
    random.shuffle(dataloader_items)
    for i_dataloader, (dataloader_name, dataloader) in enumerate(dataloader_items):
        for i_batch, batch in enumerate(dataloader):
            X      = batch["img"].to(device)
            X_pos  = batch.get("img_pos", None)
            if X_pos is not None: 
                X_pos  = X_pos.to(device) # positive part for triplet loss
            y      = batch["score"].to(device)
            s_type = batch["score_type"]             # list[str]
            instance_label = np.array(batch["patient_scoretype_key"])


            years_np = datestr_to_years_since_2000(batch["date_str"])
            days_np = (years_np*365.2425).astype(int)
            days_t = torch.from_numpy(days_np).to(device)
            years_t = torch.from_numpy(years_np).to(device)


            B = X.size(0)
            samples_tr += B
            optimizer.zero_grad()

            # -- (denoising-)autoencoder ---------------------------------------------------- #
            X_pred, z = model_AE(transform(X), s_type)       # ONE forward pass
            if X_pos is not None: 
                _, z_positive  = model_AE(X_pos, score_types=s_type)   # Could add transform(X_pos) 
                                                             # as well but should not matter. 
                                                             # Usually denoising is already trained before
            else: 
                z_positive = z  # Fall back (Note that in this case lambda should be 0!)
                assert lambda_z_triplet_WST_score == 0.0


            # triplet loss on classes: 
            loss_z_triplet_classes = loss_fn_z_triplet_classes(labels=y, embeddings=z, 
                                          ids=instance_label,  # Use None for ablation study!
                                          margin_scores=y) # score dependent margin
            loss_z_triplet_classes, fraction_positive_triplets__classes, number_of_valid_triplets__classes = loss_z_triplet_classes


            # with self-transform
            loss_z_triplet_WST_scores = loss_fn_z_triplet_WST_score(labels=y, embeddings=z, 
                                                                    embeddings_self_transform = z_positive,
                                                                    ids=instance_label,  # Use None for ablation study!
                                                                    margin_scores=y # score dependent margin
                                                               )
            loss_z_triplet_WST_scores, fraction_positive_triplets__WST_classes, number_of_valid_triplets__WST_classes = loss_z_triplet_WST_scores

            # with self-transform on TIME
            loss_z_triplet_WST_time = loss_fn_z_triplet_WST_time(labels=days_t,
                                                                  embeddings=z, 
                                                                  embeddings_self_transform = z_positive,
                                                                  ids=instance_label,  # pat001_L_PIPII
                                                                  margin_scores=years_t
                                                               )
            loss_z_triplet_WST_time, fraction_positive_triplets__WST_time, number_of_valid_triplets__WST_time = loss_z_triplet_WST_time


            # if i_batch == 0 and  i_dataloader==0 and verbose: 
            #     print(f"              first batch first DL:: fraction_positive_triplets__triplet_classes = {fraction_positive_triplets__classes.item()}")
            #     print(f"              first batch first DL:: fraction_positive_triplets__WST_classes     = {fraction_positive_triplets__WST_classes.item() }")        
            
            loss_z_SCR, number_of_valid_contributions_SCR = loss_fn_z_CR(scores=y, embeddings=z, ids=instance_label)

            # -- classifier heads ---------------------------------------------- #
            loss_y = torch.tensor(0.0, device=device)
            loss_y_delta = torch.tensor(0.0, device=device)
            loss_y_reg_extra = torch.tensor(0.0, device=device)
            if model_classifier is not None:
                out_dict = model_classifier(z, s_type, return_dict=True)

                # accumulate per-head losses
                for head_name, (idx, logits) in out_dict.items():
                    cnt = idx.numel()                # how many samples for this head
                    if cnt == 0:
                        continue                     # no contribution this batch

                    head_target = y[idx]
                    head_instance_label = instance_label[idx.cpu().numpy()]
                    if task_type_y == "regression":
                        loss_fn_y_input = logits.squeeze(-1)
                        score_estimation = loss_fn_y_input
                        head_target = head_target.float().to(device)
                    if task_type_y == "classification":
                        loss_fn_y_input = logits
                        score_estimation = ra_utils.networks.score_estimator.estimate_scalar_score_from_logits(logits, mode="expectation_value")
                    if task_type_y == "classification_regression_mix": 
                        #raise NotImplementedError("task_type_y == 'classification_regression_mix'")
                        loss_fn_y_input = logits
                        score_estimation_1 = loss_fn_y_input[..., 0]  #  These are not logits! First element is assumed to be the reg. value
                        score_estimation_2 = ra_utils.networks.score_estimator.estimate_scalar_score_from_logits(logits[..., 1:], mode="expectation_value")
                        score_estimation = loss_fn_y.mse_weight * score_estimation_1 + (1 - loss_fn_y.mse_weight) * score_estimation_2
                        # head_target_float = head_target.float().to(device)


                    head_loss   = loss_fn_y(loss_fn_y_input, head_target)   # mean over head-batch
                    head_loss_y_extra = loss_fn_y_reg_extra(score_estimation, head_target.float().to(device))
                    head_loss_y_delta = loss_fn_y_delta(score_estimation, head_target.float().to(device), head_instance_label)


                    w = model_classifier.classifier_head_infos[head_name].get(
                        "loss_weight", 1.0
                    )

                    # weighted sum for global Ly
                    loss_y += head_loss * cnt * w
                    loss_y_reg_extra += head_loss_y_extra * cnt * w
                    loss_y_delta += head_loss_y_delta * cnt * w
                    

                    # running stats for this head (raw, un-weighted)
                    running_head_loss[head_name] += head_loss.item() * cnt
                    head_counts[head_name]       += cnt

                # convert Σ(loss*count) ➜ mean per sample of *whole batch*
                loss_y = loss_y / B
                loss_y_reg_extra = loss_y_reg_extra / B
                loss_y_delta = loss_y_delta / B

            # -- reconstruction & latent losses -------------------------------- #
            loss_x = loss_fn_x(X_pred, X)
            loss_z = loss_fn_z(z, z * 0)

            # -- total loss & optimisation ------------------------------------- #
            loss = (lambda_x * loss_x + 
                    lambda_y * loss_y + 
                    lambda_z * loss_z + 
                    lambda_z_triplet_classes * loss_z_triplet_classes +
                    lambda_z_CR * loss_z_SCR + 
                    lambda_z_triplet_WST_score * loss_z_triplet_WST_scores + 
                    lambda_z_triplet_WST_time * loss_z_triplet_WST_time + 
                    lambda_y_delta * loss_y_delta + 
                    lambda_y_reg_extra * loss_y_reg_extra
                    )
            loss.backward()
            optimizer.step()

            # -- running averages ---------------------------------------------- #
            running_loss["Lx"] += loss_x.item() * B
            running_loss["Ly"] += loss_y.item() * B
            running_loss["Lz"] += loss_z.item() * B
            running_loss["Ly_delta"] += loss_y_delta.item() * B
            running_loss["Ly_reg_extra"] += loss_y_reg_extra.item() * B

            # running_loss["Lz_triplet_classes"] += loss_z_triplet_classes.item() * B
            # running_loss["Lz_triplet_WST_score"] += loss_z_triplet_WST_scores.item() * B
            # running_loss["Lz_triplet_WST_time"] += loss_z_triplet_WST_time.item() * B            
            # running_loss["Lz_score_consistency_regularizer"] += loss_z_SCR.item() * B

            # Add fraction of positive triplets and number of valid triplets to running loss
            N_pos = number_of_valid_triplets__classes.item() * fraction_positive_triplets__classes.item()
            l = loss_z_triplet_classes.item()
            contr_dct["Lz_TriCls"] = contr_dct.get("Lz_TriCls", 0.0) + (l * N_pos)
            contr_dct["Lz_TriCls_numPosTrip"]   = contr_dct.get("Lz_TriCls_numPosTrip", 0.0) + N_pos
            contr_dct["Lz_TriCls_fracPosTrip"]  = contr_dct.get("Lz_TriCls_fracPosTrip", 0.0) + fraction_positive_triplets__classes.item()
            contr_dct["Lz_TriCls_numValidTrip"] = contr_dct.get("Lz_TriCls_numValidTrip", 0.0) + number_of_valid_triplets__classes.item()

            N_pos = number_of_valid_triplets__classes.item() * fraction_positive_triplets__WST_classes.item()
            l = loss_z_triplet_WST_scores.item()
            contr_dct["Lz_TriClsWST"] = contr_dct.get("Lz_TriClsWST", 0.0) + (l * N_pos)
            contr_dct["Lz_TriClsWST_numPosTrip"]   = contr_dct.get("Lz_TriClsWST_numPosTrip", 0.0) + N_pos            
            contr_dct["Lz_TriClsWST_fracPosTrip"]  = contr_dct.get("Lz_TriClsWST_fracPosTrip", 0.0) + fraction_positive_triplets__WST_classes.item()
            contr_dct["Lz_TriClsWST_numValidTrip"] = contr_dct.get("Lz_TriClsWST_numValidTrip", 0.0) + number_of_valid_triplets__WST_classes.item()


            N_pos = number_of_valid_triplets__WST_time.item() * fraction_positive_triplets__WST_time.item()
            l = loss_z_triplet_WST_time.item()
            contr_dct["Lz_TriTimeWST"] = contr_dct.get("Lz_TriTimeWST", 0.0) + (l * N_pos)
            contr_dct["Lz_TriTimeWST_numPosTrip"]   = contr_dct.get("Lz_TriTimeWST_numPosTrip", 0.0) + N_pos
            contr_dct["Lz_TriTimeWST_fracPosTrip"]  = contr_dct.get("Lz_TriTimeWST_fracPosTrip", 0.0) + fraction_positive_triplets__WST_time.item()
            contr_dct["Lz_TriTimeWST_numValidTrip"] = contr_dct.get("Lz_TriTimeWST_numValidTrip", 0.0) + number_of_valid_triplets__WST_time.item()                 

            N_pos = number_of_valid_contributions_SCR.item()
            l = loss_z_SCR.item()
            contr_dct["Lz_SCR"]   = contr_dct.get("Lz_SCR", 0.0) + (N_pos * l) 
            contr_dct["Lz_SCR_numPos"]   = contr_dct.get("Lz_SCR_numPos", 0.0) + N_pos        

            
            running_loss["L"]  += loss.item()  * B

            if debugging:
                print("debugging --- break after 1 step")
                break

    # ------------------------------- wrap-up ------------------------------- #
    for k in running_loss:
        running_loss[k] /= samples_tr            # mean per sample

    # Mean per sample for contrastive losses: 
    contr_dct["Lz_TriCls"] /= (contr_dct["Lz_TriCls_numPosTrip"] + 1.0e-10)
    contr_dct["Lz_TriClsWST"] /= (contr_dct["Lz_TriClsWST_numPosTrip"]  + 1.0e-10)
    contr_dct["Lz_TriTimeWST"] /= (contr_dct["Lz_TriTimeWST_numPosTrip"] + 1.0e-10)
    contr_dct["Lz_SCR"] /= (contr_dct["Lz_SCR_numPos"]  + 1.0e-10)

    running_loss = {**running_loss, **contr_dct}

    # per-head means
    if model_classifier is not None:
        for head_name in model_classifier.classifier_head_infos:
            cnt = head_counts[head_name]
            running_loss[f"Ly_{head_name}"] = (
                running_head_loss[head_name] / cnt if cnt else 0.0
            )

    return running_loss






def evaluate_and_log_testset_results_AE_v4(
    model_AE:               torch.nn.Module,
    model_classifier:       Optional[torch.nn.Module],
    dataloaders:            Dict[str, torch.utils.data.DataLoader],
    loss_fn_dict: dict, 
    device:    str  = "cuda",
    classes:   Optional[List[str]] = None,
    transform            = lambda x: x,
    prefix:     str  = "test_",
    skip_metrics_logging: bool = False,
    task_type_y: Literal["classification", "regression", "classification_regression_mix"] = "classification",
):


    # ------------------------------------------------------------ 0. Forward
    metrics, outputs_all = val_epoch_AE_v4(
        model_AE,
        dataloaders,
        model_classifier=model_classifier,
        loss_fn_dict = loss_fn_dict,
        transform=transform,
        device=device,
        classes=classes,
        return_all_predictions=True,
        calc_ICC3=3,
        task_type_y=task_type_y,
    )

    # --------------------------------------------------- 1. Global metric log
    if not skip_metrics_logging:
        log_metrics_mlflow(
            metrics,
            prefix=prefix,
            classes=classes,
            step=None,
            log_report_and_confusion_matrix_as_artifact=True,
        )

    # --------------------------------------- 2. Per-score-type / per-head CRs
    score_types = outputs_all.get("score_type")
    y_true = outputs_all["labels"]
    y_pred = outputs_all["preds"]

    # Build integer class labels for classification metrics
    if task_type_y in ("regression", "classification_regression_mix"):
        y_pred_for_cls = np.round(y_pred)
        y_pred_for_cls = np.clip(y_pred_for_cls, 0, np.inf)
        if classes is not None:
            y_pred_for_cls = np.clip(y_pred_for_cls, 0, len(classes) - 1)
        y_pred_for_cls = y_pred_for_cls.astype(np.int64)
        y_true_for_cls = y_true.astype(np.int64)
    else:
        # pure classification: already class indices
        y_pred_for_cls = y_pred
        y_true_for_cls = y_true

    # # round & clip when we came from regression
    # if task_type_y == "regression":
    #     y_pred = np.round(y_pred)
    #     y_pred = np.clip(y_pred, 0, np.inf)
    #     if classes is not None:
    #         y_pred = np.clip(y_pred, 0, len(classes) - 1)
    #     y_pred = y_pred.astype(np.int64)
    #     y_true = y_true.astype(np.int64)

    if score_types is not None:
        score_types = np.asarray(score_types)

        # def _class_rep(idx):
        #     return classification_report(
        #         y_true[idx],
        #         y_pred[idx],
        #         labels=list(range(len(classes))) if classes else None,
        #         target_names=classes if classes else None,
        #         output_dict=True,
        #         zero_division=0.0,
        #     )

        def _class_rep(idx):
            return classification_report(
                y_true_for_cls[idx],
                y_pred_for_cls[idx],
                labels=list(range(len(classes))) if classes else None,
                target_names=classes if classes else None,
                output_dict=True,
                zero_division=0.0,
            )

        # per score-type
        by_st = {st: _class_rep(score_types == st) for st in np.unique(score_types)}
        mlflow.log_dict(by_st, f"metrics/{prefix}classification_report_by_score_type.json")

        # per head
        if model_classifier is not None:
            st2head = make_score_type_2_head_name_dct(model_classifier.classifier_head_infos)
            heads = np.vectorize(st2head.get)(score_types)
            by_head = {h: _class_rep(heads == h) for h in np.unique(heads)}
            mlflow.log_dict(by_head, f"metrics/{prefix}classification_report_by_head.json")

            # macro averages
            for h, rep in by_head.items():
                if "macro avg" in rep:
                    macro = rep["macro avg"]
                    mlflow.log_metric(f"{prefix}{h}_macro_precision", macro["precision"])
                    mlflow.log_metric(f"{prefix}{h}_macro_recall",    macro["recall"])
                    mlflow.log_metric(f"{prefix}{h}_macro_f1",        macro["f1-score"])

    # ----------------------------------------- 3. Extra losses (incl. triplet)
    if not skip_metrics_logging:
        extra_loss = {k: v for k, v in metrics.items() if k.startswith("L")}
        log_scalar_dict(prefix, extra_loss, step=None)  # Lz_triplet_classes now included

    # -------------------------------------------------- 4. Save raw predictions
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        np.savez_compressed(tmp.name, **outputs_all)
        mlflow.log_artifact(tmp.name, artifact_path=f"predictions/{prefix}")
        tmp_path = tmp.name
    os.remove(tmp_path)

    # -------------------------------------------------- 5. Plot reconstructions
    plot_reconstructions_by_type(
        dataloaders,
        model_AE,
        transform,
        device,
        prefix=prefix,
        n_vis_max=6,
    )

    return metrics, outputs_all
