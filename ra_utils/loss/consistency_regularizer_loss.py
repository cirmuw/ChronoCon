
__all__ = ['consistency_regularizer_loss', 'ConsistencyRegularizerLoss']

# Cell
import torch
import torch.nn.functional as F
from typing import Callable, Literal, Union
import torch.nn as nn
import numpy as np

from ra_utils.loss.loss_utils import (
    get_instance_mask_2D, 
    pairwise_distances
)


def consistency_regularizer_loss(
        embeddings: torch.Tensor, scores: torch.tensor, ids: np.ndarray, 
        metric_embeddings: Union[Literal["euclidean", "cosine", "tanh_euclidean"], Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = "tanh_euclidean", 
        metric_scores: Union[Literal["abs", "squared"], Callable[[torch.Tensor], torch.Tensor]] = "abs",
        deltino = 1.0e-1, 
        squared_embeddings_metric = False
        ):
    """
    Calculates `1/N  sum_{i !=j}  metric_embeddings(embeddings_i, embeddings_j) /  (metric_scores(scores_i - scores_j) +  deltino) * delta(ids_i, ids_j)`

    ids: np.ndarray  e.g. ["patient001_L_PIPII_JSN", "patient002_L_PIPII_JSN", "patient001_L_PIPII_ERO", ... ]
    scores: torch.Tensor  e.g. [0, 0, 2, 3, 1]
    """

    device = scores.device
    assert deltino > 0.0, "deltino should be > 0.0 to avoid division by zero"

    scores_diff = (scores.unsqueeze(0) - scores.unsqueeze(1)).float()
    if metric_scores == "abs": 
        scores_diff = torch.abs(scores_diff)
    elif metric_scores == "squared":
        scores_diff = torch.square(scores_diff)
    elif callable(metric_scores):
        scores_diff = metric_scores(scores_diff)
    scores_bracket = (scores_diff + deltino) ** -1


    emb_distances = pairwise_distances(embeddings, metric=metric_embeddings, squared=squared_embeddings_metric)
    mask_ids = get_instance_mask_2D(ids, device = device)  # but diag terms can still contribute. 
    indices_equal = torch.eye(scores.size(0), device=scores.device).bool()
    indices_not_equal = ~indices_equal
    mask = mask_ids & indices_not_equal

    N = mask.sum()  # num_valid_duplets * 2

    if N >= 1: 
        loss = (mask.float() * scores_bracket * emb_distances).sum() / N
    else: 
        loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    return loss


class ConsistencyRegularizerLoss(nn.Module):
    def __init__(
        self,
        metric_embeddings: Union[Literal["euclidean", "cosine", "tanh_euclidean"], Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = "tanh_euclidean",
        metric_scores: Union[Literal["abs", "squared"], Callable[[torch.Tensor], torch.Tensor]] = "abs",
        deltino: float = 1.0e-1,
        squared_embeddings_metric: bool = True
    ):
        super().__init__()
        self.metric_embeddings = metric_embeddings
        self.metric_scores = metric_scores
        self.deltino = deltino
        self.squared_embeddings_metric = squared_embeddings_metric

    def forward(self, embeddings: torch.Tensor, scores: torch.Tensor, ids: np.ndarray):
        """
        Calculates the consistency regularizer loss.

        Args:
            embeddings: torch.Tensor of shape (batch_size, embedding_dim)
            scores: torch.Tensor of shape (batch_size,)
            ids: np.ndarray of identifiers for samples

        Returns:
            torch.Tensor: consistency regularizer loss value
        """
        return consistency_regularizer_loss(
            embeddings=embeddings,
            scores=scores,
            ids=ids,
            metric_embeddings=self.metric_embeddings,
            metric_scores=self.metric_scores,
            deltino=self.deltino,
            squared_embeddings_metric=self.squared_embeddings_metric
        )