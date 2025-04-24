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

from ra_utils.training.scores_SHS.scores_SHS_training_lib import (
    calculate_some_classification_metrics,
    log_metrics_mlflow
)

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
            transform=transform,
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
            f"{key}: {value:.2e}" for key, value in train_loss_dct.items() if key != "L"
            )
            val_losses = "  Validation: " + " | ".join(
            f"{key}: {value:.2e}" for key, value in val_metrics.items() if key.startswith("L") and key != "L"
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
        if not run_full_epochs:
            improved = val_loss < best_val_loss
            if improved:
                if verbose:
                    print(
                        "  New best validation loss ↓ "
                        f"{best_val_loss:.4f} → {val_loss:.4f}"
                    )
                best_val_loss = val_loss
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


# ---------------------------------------------------------------------------
# multi‑head classifier model and helper functions
# ---------------------------------------------------------------------------


def make_score_type_2_head_name_dct(classifier_head_infos: dict):
    out = {}
    used_score_types = []
    for k,v in classifier_head_infos.items():
        assert set(v["score_types"]) & set(used_score_types) == set(), f"duplicate score type in dct {v} | {used_score_types}"
        for vv in v["score_types"]:
            out[vv] = k
    return out


classifier_head_infos = {
    "PIP_2-5_EP": {
        "out_dim": 6,
        "score_types":   ['PIPIIEP', 'PIPIIIEP', 'PIPIVEP', 'PIPVEP'],
        "loss_weight": 1.0 
    },
    "PIP_1_EP": {
        "out_dim": 6,
        "score_types":   ['IPIEP'],
        "loss_weight": 1.0 
    }
}


class ClassifierHeads(nn.Module):
    def __init__(self,
                 classifier_head_infos = classifier_head_infos,  
                 latent_dim: int = 480,
                 hidden_dim: int = 280,
                 dropout_p = None):
        super(ClassifierHeads, self).__init__()
        self.classifier_head_infos = classifier_head_infos.copy()
        
        # input checks on classifier_head_infos -> No overlaps in score_types
        self.score_type_2_head_name = make_score_type_2_head_name_dct(classifier_head_infos)
        self.heads = nn.ModuleDict({k: self.__make_mlp(latent_dim, v["out_dim"], 
                                                       hidden_dim = hidden_dim, 
                                                       dropout_p=dropout_p) 
                                    for k,v in classifier_head_infos.items()})
        
    def forward(self, z: torch.Tensor, score_types: List[str]) -> torch.Tensor:
        if len(z) != len(score_types):
            raise ValueError(
                f"z (length {len(z)}) and score_types (length {len(score_types)}) "
                "must match."
            )

        active_heads = [self.score_type_2_head_name[s] for s in score_types]

        # Fast path: whole batch uses the same head
        if len(set(active_heads)) == 1:
            return self.heads[active_heads[0]](z)

        # Slow path: split the batch per head (avoid calling each head B times)
        outs: list[torch.Tensor | None] = [None] * len(active_heads)
        for head_name in set(active_heads):
            idx = [i for i, h in enumerate(active_heads) if h == head_name]
            head_out = self.heads[head_name](z[idx])            # run once
            for k, i in enumerate(idx):                         # restore order
                outs[i] = head_out[k : k + 1]
        return torch.cat(outs, dim=0)

    def forward(
        self,
        z: torch.Tensor,            # [B, latent_dim]
        score_types: List[str],
        *,
        return_dict: bool = False,  # <-- NEW
    ) -> (
        Dict[str, tuple[torch.Tensor, torch.Tensor]]
        | torch.Tensor
    ):
        """
        Parameters
        ----------
        z            : latent vectors, shape [B, latent_dim]
        score_types  : length-B list with one entry per sample
        return_dict  : if True  → always return a dict
                       if False → keep the old behaviour
        Returns
        -------
        * return_dict = True
            {head_name: (idx, logits)}  where
                idx    : 1-D LongTensor with the sample positions that
                         belong to this head, length n_h
                logits : Tensor [n_h, out_dim_h]
        * return_dict = False
            Same tensor output as before (fast path when all samples share
            one head, or padded tensor+mask when heads differ).
        """
        if len(z) != len(score_types):
            raise ValueError(
                f"z has {len(z)} rows but score_types has {len(score_types)} items."
            )

        active_heads = [self.score_type_2_head_name[s] for s in score_types]

        # ------------------------------------------------------------------ #
        # ── 1. Return the new dict format, used when return_dict = True ─────
        # ------------------------------------------------------------------ #
        if return_dict:
            out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            for head_name, head in self.heads.items():        # ← iterate over *all* heads
                # positions in the batch that belong to this head
                idx = torch.as_tensor(
                    [i for i, h in enumerate(active_heads) if h == head_name],
                    dtype=torch.long,
                    device=z.device,
                )

                if idx.numel():                               # ≥1 example for this head
                    logits = head(z[idx])                     # [n_h, out_dim_h]
                else:                                         # no sample ➜ return empties
                    out_dim = head[-1].out_features
                    logits = torch.empty((0, out_dim), device=z.device, dtype=z.dtype)  
                out[head_name] = (idx, logits)                # always present

            return out

        # ------------------------------------------------------------------ #
        # ── 2. Legacy tensor output  (unchanged) ────────────────────────────
        # ------------------------------------------------------------------ #
        if len(set(active_heads)) == 1:          # fast path
            return self.heads[active_heads[0]](z)

        # different out_dim per head → pad to the largest width
        max_dim = max(self.heads[h][-1].out_features for h in set(active_heads))
        logits = z.new_zeros(len(z), max_dim)
        mask   = torch.zeros_like(logits, dtype=torch.bool)

        for head in set(active_heads):
            idx   = [i for i, h in enumerate(active_heads) if h == head]
            out_h = self.heads[head](z[idx])          # [n_h, out_dim_h]
            logits[idx, : out_h.shape[1]] = out_h
            mask[idx, : out_h.shape[1]]   = True      # valid columns

        return logits, mask




    def __make_mlp(self, in_dim, out_dim, hidden_dim=256, dropout_p=None):
        layers = [nn.Linear(in_dim, hidden_dim),
              #nn.BatchNorm1d(hidden_dim),
              nn.LayerNorm(hidden_dim),
              nn.ReLU(inplace=True)]
        if dropout_p is not None:
            layers.append(nn.Dropout(p=dropout_p))
        layers.append(nn.Linear(hidden_dim, out_dim))
        return nn.Sequential(layers)
        
        
class DummyReturnZeroLoss(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
    def forward(self, *args, **kwargs):
        return torch.tensor(0.0, device=self.device)
    



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
# Test‑set evaluation helper (AE v1)
# ---------------------------------------------------------------------------


def evaluate_and_log_testset_results_AE_v1(
    model_AE: torch.nn.Module,
    model_classifier: Optional[torch.nn.Module],
    dataloader: torch.utils.data.DataLoader,
    *,
    loss_fn_x: Optional[torch.nn.Module] = None,
    loss_fn_y: Optional[torch.nn.Module] = None,
    loss_fn_z: Optional[torch.nn.Module] = None,
    device: str = "cuda",
    classes: Optional[List[str]] = None,
    transform=lambda x: x,
    prefix: str = "test_",
    lambda_x: float = 1.0,
    lambda_y: float = 1.0,
    lambda_z: float = 1.0,
):
    """Run *val_epoch_AE_v1* on the test‑set, log results & return metrics."""

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

    log_metrics_mlflow(
        metrics,
        prefix=prefix,
        classes=classes,
        step=None,
        log_report_and_confusion_matrix_as_artifact=True,
    )

    # save raw predictions
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        np.savez_compressed(tmp.name, **outputs_all)
        npz_path = tmp.name
    mlflow.log_artifact(npz_path, artifact_path=f"predictions/{prefix}")
    os.remove(npz_path)

    # --------------------------------------------------------------
    #  Plot & log reconstructions (original → noisy → reconstructed)
    # --------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt

        n_vis = min(10, len(dataloader.dataset))  # at most 10 samples
        collected = 0
        orig, noisy, recon = [], [], []

        model_AE.eval()
        with torch.no_grad():
            for batch in dataloader:
                X_cpu = batch["img"]  # [B,1,H,W]
                X = X_cpu.to(device)
                X_t = transform(X)
                X_pred, _ = model_AE(X_t)

                for i in range(X.size(0)):
                    orig.append(X_cpu[i].squeeze().numpy())
                    noisy.append(X_t[i].cpu().squeeze().numpy())
                    recon.append(X_pred[i].cpu().squeeze().numpy())
                    collected += 1
                    if collected >= n_vis:
                        break
                if collected >= n_vis:
                    break

        # build figure --------------------------------------------------
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
        print("Failed to log reconstruction plots:", e)

    return metrics, outputs_all
