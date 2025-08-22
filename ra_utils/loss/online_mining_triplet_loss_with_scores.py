#
# CW: Original version taken from: https://github.com/NegatioN/OnlineMiningTripletLoss

__all__ = ['batch_hard_triplet_loss', 'batch_all_triplet_loss']

# Cell
import torch
import torch.nn.functional as F
from typing import Callable, Literal, Union
import torch.nn as nn

def __pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    dot_product = torch.matmul(embeddings, embeddings.t())

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = torch.diag(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances[distances < 0] = 0

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16

        distances = (1.0 -mask) * torch.sqrt(distances)

    return distances



def _pairwise_distances(
        embeddings: torch.Tensor,
        metric: Union[Literal["euclidean", "cosine"], Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = "euclidean",
        *,
        squared: bool = False,
) -> torch.Tensor:
    """
    Compute the pairwise distance matrix between all rows in `embeddings`
    for the given `metric`.

    Parameters
    ----------
    embeddings : Tensor, shape (N, D)
        Batch of row‑wise embeddings.
    metric : {"euclidean", "cosine"} or callable
        • "euclidean" – standard L2 (vectorised, fastest)  
        • "cosine"    – 1 ‑ cosine‑similarity (vectorised)  
        • any callable f(x, y) returning a scalar distance tensor.  
          It should broadcast over leading dims (will be called with
          `x[..., D]`, `y[..., D]`). If it does not, a for‑loop fallback is
          used.
    squared : bool, default False
        When metric == "euclidean", return ‖x − y‖² instead of ‖x − y‖.

    Returns
    -------
    Tensor, shape (N, N)
        Pairwise distance matrix.
    """
    if metric == "euclidean":
        # ‖a-b‖² = ‖a‖² + ‖b‖² − 2⟨a,b⟩  (same as your original code)
        dot = embeddings @ embeddings.t()
        sq_norm = torch.diag(dot)
        dist = sq_norm.unsqueeze(0) - 2 * dot + sq_norm.unsqueeze(1)
        dist.clamp_min_(0)                        # numeric safety

        if not squared:
            # avoid ∂/∂x sqrt(x) blowing up at 0 on the diagonal
            mask = dist.eq(0)
            dist = torch.sqrt(dist + mask * 1e-16)
        return dist

    elif metric == "cosine":
        # Expand to (N,1,D) and (1,N,D) then compute cosine sim along D
        sim = F.cosine_similarity(
            embeddings.unsqueeze(1),              # (N,1,D)
            embeddings.unsqueeze(0),              # (1,N,D)
            dim=-1,
            eps=1e-8
        )                                         # → (N,N)
        return 1.0 - sim                          # distance = 1 - similarity

    # ---------------- generic callable path ---------------- #
    # Try to apply the user‑supplied function in a vectorised way
    if callable(metric):
        try:
            # (N,1,D) vs (1,N,D) broadcasting — works if metric
            # accepts tensors of shape (N,1,D) and (1,N,D)
            return metric(
                embeddings.unsqueeze(1),          # broadcast
                embeddings.unsqueeze(0)
            )
        except Exception:
            # Fall back on explicit pairwise loop (O(N²) Python loops)
            pass

    # Slow fallback: compute every pair in Python; keeps grad tracking
    N = embeddings.size(0)
    out = torch.empty(N, N, device=embeddings.device, dtype=embeddings.dtype)
    for i in range(N):
        for j in range(i, N):
            d = metric(embeddings[i], embeddings[j])
            out[i, j] = out[j, i] = d
    return out




def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k


    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    valid_labels = ~i_equal_k & i_equal_j

    return valid_labels & distinct_indices


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
    indices_not_equal = ~indices_equal

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    return labels_equal & indices_not_equal


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)

    return ~(labels.unsqueeze(0) == labels.unsqueeze(1))


# Cell
def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = mask_anchor_positive * pairwise_dist

    # shape (batch_size, 1)
    hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    tl = hardest_positive_dist - hardest_negative_dist + margin
    tl = F.relu(tl)
    triplet_loss = tl.mean()

    return triplet_loss


def batch_hard_triplet_loss_NOT_TESTED(
        labels: torch.Tensor,
        embeddings: torch.Tensor,
        margin: float,
        *,
        label_margin_scale: float = 0.0,
        metric: Union[Literal["euclidean", "cosine"],
                      Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = "euclidean",
        squared: bool = False,
):
    """
    Hard‑triplet loss with optional label‑aware margins and arbitrary
    distance metrics (euclidean, cosine or a custom callable).

    Parameters
    ----------
    labels : (N,) LongTensor
    embeddings : (N, D) Tensor
    margin : float
        Base margin used when |label(a)‑label(n)| == 1.
    label_margin_scale : float, default 0.0
        Extra margin per unit of label distance beyond 1.
        margin_apn = margin + (|l_a − l_n| − 1) * label_margin_scale
    metric : {"euclidean", "cosine"} or callable, default "euclidean"
    squared : bool, default False
        Only affects the euclidean metric.

    Returns
    -------
    triplet_loss : scalar tensor
        Mean loss over the batch.
    """
    assert label_margin_scale >= 0.0, "`label_margin_scale` must be ≥ 0"

    # ------------------------------------------------------------------
    # 1. Distance matrix for the requested metric
    # ------------------------------------------------------------------
    pairwise_dist = _pairwise_distances(
        embeddings, metric=metric, squared=squared
    )

    # ------------------------------------------------------------------
    # 2. Hardest positive for every anchor
    # ------------------------------------------------------------------
    mask_pos = _get_anchor_positive_triplet_mask(labels).float()   # (N,N)
    anchor_positive_dist = mask_pos * pairwise_dist                # invalid → 0
    hardest_positive_dist, _ = anchor_positive_dist.max(dim=1, keepdim=True)  # (N,1)

    # ------------------------------------------------------------------
    # 3. Hardest negative for every anchor
    # ------------------------------------------------------------------
    mask_neg = _get_anchor_negative_triplet_mask(labels).float()   # (N,N)
    max_in_row, _ = pairwise_dist.max(dim=1, keepdim=True)         # to bump invalids
    anchor_negative_dist = pairwise_dist + max_in_row * (1.0 - mask_neg)
    hardest_negative_dist, hard_neg_idx = anchor_negative_dist.min(dim=1, keepdim=True)  # (N,1)

    # ------------------------------------------------------------------
    # 4. Dynamic margin that depends on label distance
    # ------------------------------------------------------------------
    anchor_labels = labels.view(-1, 1)                             # (N,1)
    neg_labels   = labels[hard_neg_idx.squeeze(1)].view(-1, 1)     # (N,1)

    margin_apn = (
        (anchor_labels.sub(neg_labels).abs() - 1) * label_margin_scale
        + margin
    ).type_as(hardest_negative_dist)                               # (N,1)

    # ------------------------------------------------------------------
    # 5. Loss
    # ------------------------------------------------------------------
    tl = hardest_positive_dist - hardest_negative_dist + margin_apn
    triplet_loss = F.relu(tl).mean()
    return triplet_loss




# Cell
def batch_all_triplet_loss(labels, embeddings, margin, 
                           label_margin_scale: float =0.0,
                           metric: Union[Literal["euclidean", "cosine"], Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = "euclidean",
                           squared: bool = False, 
                           scores = None    # TODO!!!!
                           ):
    """Build the triplet loss over a batch of embeddings.

    For label_margin_scale = 0.0 we get same as in original package

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    assert label_margin_scale >= 0.0, "label_margin_scale should be >= 0.0"

    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, metric=metric, squared=squared)

    anchor_positive_dist = pairwise_dist.unsqueeze(2)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)

    # Create a 3d tensor with distance between labels (=scores) as entry. 
    ## Note that label of a = label of p (enforced later) and label of n != label of a
    l_a = labels.unsqueeze(1).unsqueeze(2)  
    l_n = labels.unsqueeze(0).unsqueeze(0)
    margin_apn = (torch.abs(l_a - l_n) - 1) * label_margin_scale + margin



    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin_apn



    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    triplet_loss = mask.float() * triplet_loss

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = F.relu(triplet_loss)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = triplet_loss[triplet_loss > 1e-16]
    num_positive_triplets = valid_triplets.size(0)
    num_valid_triplets = mask.sum()

    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets



class OnlineBatchAllTripletLoss(nn.Module):
    """
    Drop‑in replacement for your previous class that supports
    alternative distance metrics (euclidean, cosine, or a custom
    callable) and an option to keep the distances squared.
    """
    def __init__(
            self,
            margin: float = 1.0,
            *,
            label_margin_scale: float = 0.0,
            metric: Union[Literal["euclidean", "cosine"],
                          Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = "euclidean",
            squared: bool = False,
    ):
        super().__init__()
        self.margin = margin
        self.label_margin_scale = label_margin_scale
        self.metric = metric
        self.squared = squared

    def forward(self, labels: torch.Tensor, embeddings: torch.Tensor):
        # Return both loss and fraction of positive triplets.
        loss, fraction_pos = batch_all_triplet_loss(
            labels,
            embeddings,
            margin=self.margin,
            label_margin_scale=self.label_margin_scale,
            metric=self.metric,
            squared=self.squared,
        )
        return loss #, fraction_pos


class OnlineBatchHardTripletLoss_NOT_TESTED(nn.Module):
    """
    Hard‑triplet loss module supporting custom metrics and label‑aware
    margins (mirrors OnlineBatchAllTripletLoss).
    """
    def __init__(
            self,
            margin: float = 1.0,
            *,
            label_margin_scale: float = 0.0,
            metric: Union[Literal["euclidean", "cosine"],
                          Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = "euclidean",
            squared: bool = False,
    ):
        super().__init__()
        self.margin = margin
        self.label_margin_scale = label_margin_scale
        self.metric = metric
        self.squared = squared

    def forward(self, labels: torch.Tensor, embeddings: torch.Tensor):
        return batch_hard_triplet_loss(
            labels,
            embeddings,
            margin=self.margin,
            label_margin_scale=self.label_margin_scale,
            metric=self.metric,
            squared=self.squared,
        )





#### CW: Additional functions for triplet loss with classes
class OnlineBatchHardTripletLoss(nn.Module):
    def __init__(self, margin = 1.0):
        super().__init__()
        self.margin = margin
    def forward(self, labels, embeddings):
        loss = batch_hard_triplet_loss(labels, embeddings, margin=self.margin)
        return loss


