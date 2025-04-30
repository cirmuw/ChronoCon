import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence
from torch import Tensor


def get_score_loss_function(cfg: dict):
    name = cfg.get("name", "CrossEntropyLoss")
    params = cfg.get("params", {})
    print(f"loss for score: {name}; \n    params = {params}")
    
    if name == "CrossEntropyLoss":
        loss = nn.CrossEntropyLoss(**params)
    elif name == "FocalLoss":
        loss = FocalLoss(**params)
    elif name == "PaulsOrdinalLoss":
        loss = PaulsOrdinalLoss(**params)
    elif name == "PaulsOrdinalLossFocal":
        loss = PaulsOrdinalLossFocal(**params)        
        
    else: 
        raise NotImplementedError(f"{name=}")
    return loss
        
        




#------------------------------------------------------------------------------------#
def focal_loss_simple(outputs, targets, gamma=0, alpha=None):
    ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none') # important to add reduction='none' to keep per-batch-item loss
    pt = torch.exp(-ce_loss)
    if alpha == None: 
        loss = ((1-pt)**gamma * ce_loss).mean() # mean over the batch
    else: 
        loss = (alpha * (1-pt)**gamma * ce_loss).mean() # mean over the batch
    return loss


# # https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py

class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


def focal_loss(alpha: Optional[Sequence] = None,
               gamma: float = 0.,
               reduction: str = 'mean',
               ignore_index: int = -100,
               device='cpu',
               dtype=torch.float32) -> FocalLoss:
    """Factory function for FocalLoss.

    Args:
        alpha (Sequence, optional): Weights for each class. Will be converted
            to a Tensor if not None. Defaults to None.
        gamma (float, optional): A constant, as described in the paper.
            Defaults to 0.
        reduction (str, optional): 'mean', 'sum' or 'none'.
            Defaults to 'mean'.
        ignore_index (int, optional): class label to ignore.
            Defaults to -100.
        device (str, optional): Device to move alpha to. Defaults to 'cpu'.
        dtype (torch.dtype, optional): dtype to cast alpha to.
            Defaults to torch.float32.

    Returns:
        A FocalLoss object
    """
    if alpha is not None:
        if not isinstance(alpha, Tensor):
            alpha = torch.tensor(alpha)
        alpha = alpha.to(device=device, dtype=dtype)

    fl = FocalLoss(
        alpha=alpha,
        gamma=gamma,
        reduction=reduction,
        ignore_index=ignore_index)
    return fl


#
#-------------------------------------------------------#

class PaulsOrdinalLoss(nn.Module):
    def __init__(self, lam=0):
        super(PaulsOrdinalLoss, self).__init__()
        self.lam = lam

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.cross_entropy(pred, target, reduction="none")
        if self.lam > 1.0e-8:
            w = torch.abs(torch.argmax(pred, dim=1) - target) + 1
            loss = loss*(w ** self.lam)
            loss = torch.mean(loss)
        return loss


class PaulsOrdinalLossFocal(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    but with Pauls modification 
      CE *=  (1 - |argmax(..) - argmax(..)|**lam(bda= )
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 lam: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
            lam: 
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        if alpha is not None and not isinstance(alpha, Tensor):
            alpha = torch.tensor(alpha)
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce
        
        # Pauls modification
        if self.lam > 1.0E-8:
            w = torch.abs(torch.argmax(log_p, dim=1) - y) + 1
            loss = loss*(w ** self.lam)
            
            

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

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
