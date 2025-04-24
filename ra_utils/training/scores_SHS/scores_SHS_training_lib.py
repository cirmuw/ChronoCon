import os
import copy
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
import mlflow
import mlflow.pytorch

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
)

from ra_utils.networks.architecture import (
    model_interface_forward
)


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
        output_dct = model_interface_forward(
            model, batch, device, options=interface_option)
        loss = criterion(output_dct["class_logits"], Y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)



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
            output_dct = model_interface_forward(
                model, batch, device, options=interface_option)
            logits = output_dct["class_logits"]
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
    interface_option="image only"
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


