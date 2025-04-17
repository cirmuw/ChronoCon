import torch
import torch.nn as nn
import torch.functional as F


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
