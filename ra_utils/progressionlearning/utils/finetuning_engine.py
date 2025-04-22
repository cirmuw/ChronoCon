
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from utils.plotting import plot_metric
import os

def train_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device = 'cuda',
        timepoints: list = ['T0', 'T1', 'T2', 'T3'],
        ):
    losses_epoch= []
    model.train()
    for batch_data in loader:
        x = [batch_data[t][0].float().to(device) for t in timepoints]
        y = batch_data['pcr'].float().to(device).unsqueeze(1)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses_epoch.append(loss.item())

    return {'loss' : sum(losses_epoch)/len(losses_epoch)}

def eval_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device = 'cuda',
        timepoints: list = ['T0', 'T1', 'T2', 'T3'],
        ):
    losses_epoch = []
    auc_scores = [] 
    prau_scores = []
    model.eval()
    with torch.no_grad():
        for batch_data in loader:
            x = [batch_data[t].float().to(device) for t in timepoints]
            y = batch_data['pcr'].float().to(device).unsqueeze(1)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            losses_epoch.append(loss.item())
            y_pred_prob = torch.sigmoid(y_pred)
            auc = roc_auc_score(y.cpu().numpy(), y_pred_prob.cpu().numpy())
            auc_scores.append(auc)
            prau = average_precision_score(y.cpu().numpy(), y_pred_prob.cpu().numpy())
            prau_scores.append(prau)

    return {
        'loss': sum(losses_epoch)/len(losses_epoch),
        'auc': sum(auc_scores)/len(auc_scores),
        'auc_std': np.std(auc_scores),
        'prauc': sum(prau_scores)/len(prau_scores),
        'prauc_std': np.std(prau_scores),
    }



def do_final_evaluation(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    savepath: str,
    device: torch.device = 'cuda',
    timepoints: list = ['T0', 'T1', 'T2', 'T3'],   
):
    losses_epoch = []
    auc_scores = []
    prau_scores = []
    model.eval()
    os.makedirs(savepath, exist_ok=True)
    with torch.no_grad():
        for batch_data in loader:
            x = [batch_data[t].float().to(device) for t in timepoints]
            y = batch_data['pcr'].float().to(device).unsqueeze(1)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            losses_epoch.append(loss.item())
            y_pred_prob = torch.sigmoid(y_pred)
            
            # Calculate AUROC
            auc = roc_auc_score(y.cpu().numpy(), y_pred_prob.cpu().numpy())
            auc_scores.append(auc)
            
            # Calculate AUPR
            prau = average_precision_score(y.cpu().numpy(), y_pred_prob.cpu().numpy())
            prau_scores.append(prau)

    # Calculate mean and standard deviation
    auc_mean, auc_std = np.mean(auc_scores), np.std(auc_scores)
    prau_mean, prau_std = np.mean(prau_scores), np.std(prau_scores)

    # Plot results
    plot_metric("AUROC", auc_scores, auc_mean, auc_std, savepath)
    plot_metric("AUPR", prau_scores, prau_mean, prau_std, savepath)

    print(
        f'loss: {np.mean(losses_epoch)} \
        auc: {auc_mean} \
        auc_std: {auc_std} \
        prauc: {prau_mean} \
        prauc_std: {prau_std}'
    )