#
# CW: Original version taken from: https://github.com/NegatioN/OnlineMiningTripletLoss

__all__ = ['batch_all_triplet_loss_with_scores_and_ids']


import torch
import torch.nn.functional as F
from typing import Callable, Literal, Union
import torch.nn as nn
import numpy as np

from ra_utils.loss.loss_utils import (
    pairwise_distances,
    pairwise_distances_two_tensors,
    get_anchor_positive_triplet_mask,
    get_anchor_negative_triplet_mask,
    get_instance_mask_2D,
    get_instance_mask_3D,
    # get_triplet_mask,    
    get_triplet_mask_WST, 
    distance_two_tensors
)


def get_triplet_mask_time_leq(time):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - time[i] <= time[j]  <= time[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(time.size(0), device=time.device).bool()
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k


    time_leq_2d = time.unsqueeze(1) <= time.unsqueeze(0)
    # dt_2d = time.unsqueeze(1) - time.unsqueeze(0)
    # time_leq_2d = (dt_2d  >= 0)
    ti_leq_tj = time_leq_2d.unsqueeze(2)
    tj_leq_tk = time_leq_2d.unsqueeze(0)
    ti_leq_tj_leq_tk = ti_leq_tj & tj_leq_tk

    valid_labels = ti_leq_tj_leq_tk

    return valid_labels & distinct_indices


def get_triplet_mask_scores_leq(scores):

    # Check that i, j and k are distinct
    indices_equal = torch.eye(scores.size(0), device=scores.device).bool()
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k


    scores_leq_2d = scores.unsqueeze(1) <= scores.unsqueeze(0)
    si_leq_sj = scores_leq_2d.unsqueeze(2)
    sj_leq_sk = scores_leq_2d.unsqueeze(0)
    si_leq_sj_leq_sk = si_leq_sj & sj_leq_sk

    valid_labels = si_leq_sj_leq_sk

    return valid_labels & distinct_indices


def get_triplet_mask_scores_geq(scores):
    # Check that i, j and k are distinct
    indices_equal = torch.eye(scores.size(0), device=scores.device).bool()
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k


    scores_geq_2d = scores.unsqueeze(1) >= scores.unsqueeze(0)
    si_geq_sj = scores_geq_2d.unsqueeze(2)
    sj_geq_sk = scores_geq_2d.unsqueeze(0)
    si_geq_sj_geq_sk = si_geq_sj & sj_geq_sk

    valid_labels = si_geq_sj_geq_sk

    return valid_labels & distinct_indices




def batch_all_triplet_loss_MDP_and_ids(
    time,   # Labels for triplet term This is the time!!
    embeddings,
    ids: np.ndarray = None,
    margin_offset=0.0,
    metric: Union[Literal["euclidean", "cosine", "tanh_euclidean"], Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = "euclidean",
    squared: bool = False, 
    scores = None,
    score_direction: Literal["increase", "decrease", "ignore"] = "ignore"
    ):

    assert margin_offset >= 0.0, "margin_offset should be >= 0.0"

    if score_direction in ["increase", "decrease"]:
        assert scores is not None, f" {score_direction = } but {scores = }"


    # Get the pairwise distance matrix
    pairwise_dist = pairwise_distances(
        embeddings, metric=metric, squared=squared)

    anchor_positive_dist = pairwise_dist.unsqueeze(2)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)

    # Id must be same (e.g. same Patient; same side, same ROI)
    if ids is None:
        delta_id = torch.ones(1, dtype=torch.bool)
    else:
        # (N,N,N) tensor that ensures that ids match (e.g. loss only on same patient, ...) but not self
        delta_id = get_instance_mask_3D(ids=ids, device=embeddings.device)  # (1,N,N)

    # Create a 3d tensor with distance between scores as entry. (Can be time, JSN score, ...)
    margin = margin_offset

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask_triplet_time = get_triplet_mask_time_leq(time)
    if score_direction == "increase":
        mask_scores = get_triplet_mask_scores_leq(scores)
        mask = (mask_triplet_time & delta_id) & mask_scores
    elif score_direction == "decrease":
        mask_scores = get_triplet_mask_scores_geq(scores)
        mask = (mask_triplet_time & delta_id) & mask_scores
    elif score_direction == "ignore":
        mask = mask_triplet_time & delta_id
    else: 
        raise ValueError(f"{score_direction = }")

    triplet_loss = mask.float() * triplet_loss

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = F.relu(triplet_loss)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = triplet_loss[triplet_loss > 1e-16]
    num_positive_triplets = valid_triplets.size(0)
    num_valid_triplets = mask.sum()

    fraction_positive_triplets = num_positive_triplets / \
        (num_valid_triplets.float() + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets, num_valid_triplets



class OnlineBatchAllTripletLossMDPForward(nn.Module):
    def __init__(self,
                margin_offset=0.0,
                metric: Union[Literal["euclidean", "cosine", "tanh_euclidean"], Callable[[
                    torch.Tensor, torch.Tensor], torch.Tensor]] = "euclidean",
                squared: bool = False,
                return_fraction_positive_triplets=True, 
                score_direction: Literal["increase", "decrease", "ignore"] = "ignore"
                ):
        super().__init__()
        self.margin_offset = margin_offset
        self.metric = metric
        self.squared = squared
        self.return_fraction_positive_triplets = return_fraction_positive_triplets
        self.score_direction = score_direction

    def forward(self, time, embeddings, ids, scores=None):
        loss = batch_all_triplet_loss_MDP_and_ids(time=time, 
                                         embeddings=embeddings, 
                                         ids=ids,
                                         scores = scores, 
                                         margin_offset=self.margin_offset,
                                         metric=self.metric, 
                                         squared=self.squared, 
                                         score_direction=self.score_direction)
        if self.return_fraction_positive_triplets:
            return loss
        else: 
            return loss[0]
        
    def __repr__(self):
        return f"OnlineBatchAllTripletLossMDPForward(margin_offset={self.margin_offset}, " \
               f"metric={self.metric}, squared={self.squared}, " \
               f"return_fraction_positive_triplets={self.return_fraction_positive_triplets}, "\
               f"score_direction={self.score_direction})"

# TODO Possible extension:  Maybe also implement as Hard or Semi-Hard triplet loss
