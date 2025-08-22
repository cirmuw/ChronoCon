
__all__ = ['batch_hard_triplet_loss', 'batch_all_score_differences_loss']

# Cell
import torch
import torch.nn.functional as F
from typing import Callable, Literal, Union
import torch.nn as nn
import numpy as np

from ra_utils.loss.loss_utils import (
    get_instance_mask_2D
)



def batch_all_score_differences_loss(scores_pred: torch.tensor, scores_true: torch.tensor, labels: np.ndarray, 
                                     loss_fn = F.huber_loss):
    """
    Computes the MSE of pairwise score differences, but only between items
    with the same label (excluding self-pairs).


    Computes a loss between score differences. 
    E.g.
    `scores_pred = (p1, p2)`
    -> Δ_pred = [[0,     p1-p2],    [[Δp_11, Δp_12],
                 [p2-p1, 0    ]]  =  [Δp_21, Δp_22] 
    `scores_true = (t1, t2)`
    -> Δ_true = [[0,     t1-t2],    [[Δt_11, Δt_12],
                 [t2-t1, 0    ]]  =  [Δt_21, Δt_22]     
    `labels = ("F_L_SID1_652_IP", "F_L_SID1_652_IP")
    The expected input corresponts to predicted and true scores a time-series of images. 
    The difference over time should be predicted correctly. 
    Thus a contribution to the loss should only occure if all labels match meaning that
    - Extremity, side, ROI, patient_id and score_type are all the same. 
    - (Thus only the time is different)


    Parameters
    ----------
    scores_pred : torch.Tensor, shape (N,)
        Predicted scores.
    scores_true : torch.Tensor, shape (N,)
        Ground-truth scores.
    labels : np.ndarray, shape (N,)
        1-D array of labels; only pairs with equal labels contribute.
    loss_fn : callable
        A torch loss that accepts (input, target, reduction='none').

    Returns
    -------
    loss : torch.Tensor, scalar
        The average loss over all valid (i ≠ j, labels[i] == labels[j]) pairs.

    >>> import numpy as np, torch
    >>> # simple case: two elements, same label
    >>> scores_t = torch.tensor([1.0, 3.0])
    >>> scores_p = torch.tensor([1.0, 2.0])
    >>> labels = np.array(["x", "x"])
    >>> # Δ_true = [[0, -2], [2, 0]]; Δ_pred = [[0, -1], [1, 0]]
    >>> # element-wise MSE on off-diagonals: ((-1+2)**2 + (1-2)**2) / 2 = (1 + 1)/2 = 1.0
    >>> batch_all_score_differences_loss(scores_p, scores_t, labels).item()
    1.0

    >>> # if no matching labels, loss should be zero
    >>> labels = np.array([1, 2, 3])
    >>> scores_t = torch.randn(3)
    >>> scores_p = torch.randn(3)
    >>> batch_all_score_differences_loss(scores_p, scores_t, labels).item() == 0.0
    True
    """
    
    # basic shape checks
    assert isinstance(scores_pred, torch.Tensor) and isinstance(scores_true, torch.Tensor), \
        "scores_pred and scores_true must be torch.Tensor"
    assert scores_pred.ndim == 1 and scores_true.ndim == 1, \
        "scores_pred and scores_true must be 1-D tensors"
    assert scores_pred.shape == scores_true.shape, \
        "scores_pred and scores_true must have the same shape"
    assert isinstance(labels, np.ndarray), \
        "labels must be a numpy array"
    assert labels.ndim == 1 and labels.shape[0] == scores_pred.shape[0], \
        "labels must be 1-D and match the length of scores"    
    
    device = scores_pred.device

    # differences between scores (any vs any)  (N,N)
    scores_delta_true = (scores_true.unsqueeze(0) - scores_true.unsqueeze(1)).float()
    scores_delta_pred = (scores_pred.unsqueeze(0) - scores_pred.unsqueeze(1)).float()

    score_metric_cont = loss_fn(scores_delta_true, scores_delta_pred, reduction='none')
    mask = get_instance_mask_2D(labels, device = device)
    num_valid_duplets = mask.sum()
    if num_valid_duplets >= 1: 
        loss = (mask.float() * score_metric_cont).sum() / num_valid_duplets
    else: 
        loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    return loss


