import time
import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import Affine, Rand2DElastic, RandGaussianNoise

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(data_train_x, data_train_y, model, optimizer, n_classes, classes, augment, aug_params, regression, weighted_kappa, lam, ordinal, 
          return_loss = False):

    model.train()
    total_loss = 0
    optimizer.zero_grad()


    for i in range(len(classes)):
        if classes[i] != i:
            data_train_y[data_train_y == classes[i]] = i

    """
    if augment == True:
        data_train_x = preprocess(data = data_train_x,
                                    aug_params = aug_params)
    """
    pred = model(data_train_x.to(device))

    if regression == True:
        pred = torch.reshape(pred, (pred.shape[0],))
        data_train_y = data_train_y.float()
        loss_all = F.mse_loss(pred, data_train_y.to(device), reduction="none")
    elif ordinal == True:
        modified_target = torch.zeros_like(pred)
        for i, target in enumerate(data_train_y):
            modified_target[i, 0:target+1] = 1
        loss_all = torch.abs(pred-modified_target).sum(axis=1)
        #loss_all = F.l1_loss(pred, modified_target.to(device), reduction="none").sum(axis=1)
    else:
        loss_all = F.cross_entropy(pred, data_train_y.to(device), reduction="none")
        if weighted_kappa:
            w = torch.abs(torch.argmax(pred, dim=1) - data_train_y.to(device)) + 1
            loss_all = loss_all*(w**lam)
            if False:
                print("kappa: ", weighted_kappa)
                print("lambda: ", lam)
                print("dist: ", w)
                print("dist lam: ", w**lam)
    if False:
        print("Pred: ", pred[:5])
        #print("True: ", modified_target[:5])
        print("True: ", data_train_y)
        print("loss_all: ", loss_all[:5])
    loss = torch.mean(loss_all)
    loss.backward()
    optimizer.step()

    mean_values, count_values = get_individual_loss(unique_classes = classes,
                                                    classes = data_train_y,
                                                    values = loss_all)

    #return loss/data_train_x.shape[0]
    if return_loss:
        return loss.item(), mean_values, count_values
    else:
        return mean_values, count_values



def validate(data_val_x, data_val_y, model, n_classes, classes, regression, weighted_kappa, lam, ordinal, 
             return_loss = False):
    model.eval()
    total_loss = 0


    for i in range(len(classes)):
        if classes[i] != i:
            data_val_y[data_val_y == classes[i]] = i

    pred = model(data_val_x.to(device))

    if regression == True:
        pred = torch.reshape(pred, (pred.shape[0],))
        data_val_y = data_val_y.float()
        loss_all = F.mse_loss(pred, data_val_y.to(device), reduction="none")
    elif ordinal == True:
        modified_target = torch.zeros_like(pred)
        for i, target in enumerate(data_val_y):
            modified_target[i, 0:target+1] = 1
        loss_all = F.mse_loss(pred, modified_target.to(device), reduction="none").sum(axis=1)
    else:
        loss_all = F.cross_entropy(pred, data_val_y.to(device), reduction="none")
        if weighted_kappa:
            w = torch.abs(torch.argmax(pred, dim=1) - data_val_y.to(device)) + 1
            loss_all = loss_all*(w**lam)
    loss = torch.mean(loss_all)

    mean_values, count_values = get_individual_loss(unique_classes = classes,
                                                    classes = data_val_y,
                                                    values = loss_all)
    if return_loss:
        return loss.item(), mean_values, count_values
    else:
        return mean_values, count_values



def test(data_val_x, model):
    model.eval()
    with torch.no_grad():
        pred = model(data_val_x.to(device))

    #return torch.argmax(input=pred, dim=1)
    return pred


def get_individual_loss(unique_classes, classes, values):
    mean_values = np.array([0.]*len(unique_classes))
    count_values = np.array([0.]*len(unique_classes))

    for i in range(len(unique_classes)):
        c = unique_classes[i]
        ind = np.where(classes == c)[0]
        if ind.size != 0:
            count_values[i] = len(values[ind])
            mean_values[i] = torch.mean(values[ind])
        else:
            count_values[i] = 0
            mean_values[i] = 0

    return mean_values, count_values


def get_individual_loss_v2(unique_classes, classes, values, device="cpu"):
    # Ensure unique_classes is a torch tensor (on same device)

    if isinstance(unique_classes, (np.ndarray, list)):
        classes = torch.tensor(classes, device=torch.device(device))
    if isinstance(values, (np.ndarray, list)):
        values = torch.tensor(values, device=torch.device(device))
    if isinstance(classes, (np.ndarray, list)):
        classes = torch.tensor(classes, device=torch.device(device))

    mean_values = torch.zeros(len(unique_classes), dtype=torch.float32)
    count_values = torch.zeros(len(unique_classes), dtype=torch.float32)

    for i, c in enumerate(unique_classes):
        mask = (classes == c)
        if mask.any():
            count_values[i] = mask.sum()
            mean_values[i] = values[mask].mean()
        else:
            count_values[i] = 0
            mean_values[i] = 0

    return mean_values, count_values