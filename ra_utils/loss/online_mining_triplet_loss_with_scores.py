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
    get_triplet_mask,    
    get_triplet_mask_WST, 
    distance_two_tensors
)




def batch_all_triplet_loss_with_scores_and_ids(
    labels,   # Labels for triplet term
    embeddings,
    ids: np.ndarray = None,
    margin_offset=1.0,
    margin_scores=None,
    margin_scale: float = 0.0,
    metric: Union[Literal["euclidean", "cosine", "tanh_euclidean"], Callable[[
        torch.Tensor, torch.Tensor], torch.Tensor]] = "euclidean",
    squared: bool = False
):

    assert margin_offset >= 0.0, "margin_offset should be >= 0.0"
    assert margin_scale >= 0.0, "margin_scale should be >= 0.0"

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
    # Note that in this general way it can happen that the margin score of the positive pair is different than the anchor...
    # In such cases it is not advisable to use it  -> None
    margin = margin_offset
    if margin_scores is not None:
        s_p = margin_scores.unsqueeze(0).unsqueeze(2)  # -> (1, N, 1)
        s_n = margin_scores.unsqueeze(0).unsqueeze(0)  # -> (1, 1, N)
        margin += (torch.abs(s_p - s_n)) * \
            margin_scale + margin   # (a,p,n) tensor

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask_triplet = get_triplet_mask(labels)
    mask = mask_triplet & delta_id
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


class OnlineBatchAllTripletLossWithScores(nn.Module):
    def __init__(self,
                 margin_offset=1.0,
                 margin_scale: float = 0.0,
                 metric: Union[Literal["euclidean", "cosine", "tanh_euclidean"], Callable[[
                     torch.Tensor, torch.Tensor], torch.Tensor]] = "euclidean",
                 squared: bool = False,
                 return_fraction_positive_triplets=True
                 ):
        super().__init__()
        self.margin_offset = margin_offset
        self.margin_scale = margin_scale
        self.metric = metric
        self.squared = squared
        self.return_fraction_positive_triplets = return_fraction_positive_triplets

    def forward(self, labels, embeddings, ids=None, margin_scores=None):
        loss = batch_all_triplet_loss_with_scores_and_ids(labels, 
                                                          embeddings, 
                                                          ids=ids,
                                                          margin_offset=self.margin_offset,
                                                          margin_scores=margin_scores, 
                                                          margin_scale=self.margin_scale, 
                                                          metric=self.metric, 
                                                          squared=self.squared)
        if self.return_fraction_positive_triplets:
            return loss
        else:
            return loss[0]
        
    def __repr__(self):
        return f"OnlineBatchAllTripletLossWithScores(margin_offset={self.margin_offset}, " \
               f"margin_scale={self.margin_scale}, metric={self.metric}, squared={self.squared}, " \
               f"return_fraction_positive_triplets={self.return_fraction_positive_triplets})"

class OnlineBatchAllTripletLossWithScoresNoID(nn.Module):
    def __init__(self,
                 margin_offset=1.0,
                 margin_scale: float = 0.0,
                 metric: Union[Literal["euclidean", "cosine", "tanh_euclidean"], Callable[[
                     torch.Tensor, torch.Tensor], torch.Tensor]] = "euclidean",
                 squared: bool = False,
                 return_fraction_positive_triplets=True
                 ):
        super().__init__()
        self.margin_offset = margin_offset
        self.margin_scale = margin_scale
        self.metric = metric
        self.squared = squared
        self.return_fraction_positive_triplets = return_fraction_positive_triplets

    def forward(self, labels, embeddings, ids=None, margin_scores=None):
        loss = batch_all_triplet_loss_with_scores_and_ids(labels, 
                                                          embeddings, 
                                                          ids=None,   # No matter what user gives as Id use None!!
                                                          margin_offset=self.margin_offset,
                                                          margin_scores=margin_scores, 
                                                          margin_scale=self.margin_scale, 
                                                          metric=self.metric, 
                                                          squared=self.squared)
        if self.return_fraction_positive_triplets:
            return loss
        else:
            return loss[0]
        
    def __repr__(self):
        return f"OnlineBatchAllTripletLossWithScoresNoID(margin_offset={self.margin_offset}, " \
               f"margin_scale={self.margin_scale}, metric={self.metric}, squared={self.squared}, " \
               f"return_fraction_positive_triplets={self.return_fraction_positive_triplets})"





# Not the most efficient implementation... 
# E.g. Almost all of the computed positive distances are not used... 
# But for testing I will keep it as is!
def batch_all_triplet_loss_WST(
        labels : torch.Tensor, 
        embeddings: torch.Tensor, 
        embeddings_self_transform: torch.Tensor, 
        ids: np.ndarray = None,
        margin_offset=1.0,
        margin_scores=None,
        margin_scale: float = 0.0,
        metric: Union[Literal["euclidean", "cosine", "tanh_euclidean"], Callable[[
            torch.Tensor, torch.Tensor], torch.Tensor]] = "euclidean",
        squared: bool = False
):


    assert margin_offset >= 0.0, "margin_offset should be >= 0.0"
    assert margin_scale >= 0.0, "margin_scale should be >= 0.0"

 
    pairwise_dist = pairwise_distances(embeddings, metric=metric, squared=squared)
    anchor_negative_dist = pairwise_dist # (N,N)


    dist_self = distance_two_tensors(embeddings, embeddings_self_transform, metric=metric, squared=squared) # dim (N)
    anchor_positive_dist = dist_self.unsqueeze(1) # (N,1)

    # Id must be same (e.g. same Patient; same side, same ROI)
    if ids is None:
        delta_id = torch.ones(1, dtype=torch.bool)
    else:
        delta_id = get_instance_mask_2D(ids=ids, device=embeddings.device)  # (N,N)


    # marin tensor (e.g. times)
    margin = margin_offset
    if margin_scores is not None:
        s_a = margin_scores.unsqueeze(1)  # -> (N, 1)
        s_n = margin_scores.unsqueeze(0)  # -> (1, N)
        margin += (torch.abs(s_a - s_n)) * \
            margin_scale + margin   

    # Compute a 2D tensor of size (batch_size, batch_size)
    # triplet_loss[i, i, k] will contain the triplet loss of anchor=i, positive=i, negative=k
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin


    mask_triplet = get_triplet_mask_WST(labels) # (N,N)
    mask = mask_triplet & delta_id
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





class OnlineBatchAllTripletLossWithScoresAndSelfTransform(nn.Module):
    def __init__(self,
                margin_offset=1.0,
                margin_scale: float = 0.0,
                metric: Union[Literal["euclidean", "cosine", "tanh_euclidean"], Callable[[
                    torch.Tensor, torch.Tensor], torch.Tensor]] = "euclidean",
                squared: bool = False,
                return_fraction_positive_triplets=True
                ):
        super().__init__()
        self.margin_offset = margin_offset
        self.margin_scale = margin_scale
        self.metric = metric
        self.squared = squared
        self.return_fraction_positive_triplets = return_fraction_positive_triplets

    def forward(self, labels, embeddings, embeddings_self_transform, ids=None, margin_scores=None):
        loss = batch_all_triplet_loss_WST(labels, 
                                         embeddings, 
                                         embeddings_self_transform, 
                                         ids=ids,
                                         margin_offset=self.margin_offset,
                                         margin_scores=margin_scores,
                                         margin_scale=self.margin_scale, 
                                         metric=self.metric, 
                                         squared=self.squared)
        if self.return_fraction_positive_triplets:
            return loss
        else: 
            return loss[0]
        
    def __repr__(self):
        return f"OnlineBatchAllTripletLossWithScoresAndSelfTransform(margin_offset={self.margin_offset}, " \
               f"margin_scale={self.margin_scale}, metric={self.metric}, squared={self.squared}, " \
               f"return_fraction_positive_triplets={self.return_fraction_positive_triplets})"

# TODO Possible extension:  Maybe also implement as Hard or Semi-Hard triplet loss
