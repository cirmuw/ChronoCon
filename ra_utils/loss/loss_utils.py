#
# CW: Original version taken from: https://github.com/NegatioN/OnlineMiningTripletLoss

__all__ = ['pairwise_distances', 'pairwise_distances_two_tensors',  'get_anchor_positive_triplet_mask', 'get_anchor_negative_triplet_mask', 'get_instance_mask_2D']

import torch
import torch.nn.functional as F
from typing import Callable, Literal, Union
import torch.nn as nn
import numpy as np




def pairwise_distances(
        embeddings: torch.Tensor,
        metric: Union[Literal["euclidean", "cosine", "tanh_euclidean"], Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = "euclidean",
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
    
    if metric == "tanh_euclidean":
        # compute tanh(‖a-b‖²)
        dot = embeddings @ embeddings.t()
        sq_norm = torch.diag(dot)
        dist = sq_norm.unsqueeze(0) - 2 * dot + sq_norm.unsqueeze(1)
        dist.clamp_min_(0)                        # numeric safety

        if not squared:
            # avoid ∂/∂x sqrt(x) blowing up at 0 on the diagonal
            mask = dist.eq(0)
            dist = torch.sqrt(dist + mask * 1e-16)
        return torch.tanh(dist)

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


def pairwise_distances_two_tensors(
        embeddingsA: torch.Tensor,
        embeddingsB: torch.Tensor,
        metric: Union[Literal["euclidean", "cosine", "tanh_euclidean"], Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = "euclidean",
        *,
        squared: bool = False,
) -> torch.Tensor:

    assert embeddingsA.ndim == 2 and embeddingsB.ndim == 2, "Both inputs must be 2-D tensors"
    assert embeddingsA.size(1) == embeddingsB.size(1), "Incompatible tensor dimensions"

    if metric == "euclidean":
        # ‖a-b‖² = ‖a‖² + ‖b‖² − 2⟨a,b⟩  (same as your original code)
        dot = embeddingsA @ embeddingsB.t()
        sq_norm = torch.diag(dot)
        dist = sq_norm.unsqueeze(0) - 2 * dot + sq_norm.unsqueeze(1)
        dist.clamp_min_(0)                        # numeric safety

        if not squared:
            # avoid ∂/∂x sqrt(x) blowing up at 0 on the diagonal
            mask = dist.eq(0)
            dist = torch.sqrt(dist + mask * 1e-16)
        return dist
    
    if metric == "tanh_euclidean":
        # compute tanh(‖a-b‖²)
        dot = embeddingsA @ embeddingsB.t()
        sq_norm = torch.diag(dot)
        dist = sq_norm.unsqueeze(0) - 2 * dot + sq_norm.unsqueeze(1)
        dist.clamp_min_(0)                        # numeric safety

        if not squared:
            # avoid ∂/∂x sqrt(x) blowing up at 0 on the diagonal
            mask = dist.eq(0)
            dist = torch.sqrt(dist + mask * 1e-16)
        return torch.tanh(dist)

    elif metric == "cosine":
        # Expand to (N,1,D) and (1,N,D) then compute cosine sim along D
        sim = F.cosine_similarity(
            embeddingsA.unsqueeze(1),              # (N,1,D)
            embeddingsB.unsqueeze(0),              # (1,N,D)
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
                embeddingsA.unsqueeze(1),          # broadcast
                embeddingsB.unsqueeze(0)
            )
        except Exception:
            # Fall back on explicit pairwise loop (O(N²) Python loops)
            pass

    # Slow fallback: compute every pair in Python; keeps grad tracking
    N = embeddingsA.size(0)
    out = torch.empty(N, N, device=embeddingsA.device, dtype=embeddingsA.dtype)
    for i in range(N):
        for j in range(i, N):
            d = metric(embeddingsA[i], embeddingsB[j])
            out[i, j] = out[j, i] = d
    return out





def get_triplet_mask(labels):
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


def get_triplet_mask_WST(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    Since tensor p is selftransform of a labels match automatically when i = j
    A triplet (i, j, k) is valid if:
        - i = j but k is distinct
        - labels[i] != labels[k]
    Args:
        labels: `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
    indices_not_equal = ~indices_equal

    idx_i_equal_j = indices_equal.unsqueeze(2)

    # idx_i_not_equal_j = indices_not_equal.unsqueeze(2)
    idx_i_not_equal_k = indices_not_equal.unsqueeze(1)
    # idx_j_not_equal_k = indices_not_equal.unsqueeze(0)

    valid_self_transform_index = idx_i_equal_j & idx_i_not_equal_k

    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    # i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)
    valid_labels = ~i_equal_k

    return valid_labels & valid_self_transform_index


def get_anchor_positive_triplet_mask(labels):
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


def get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)

    return ~(labels.unsqueeze(0) == labels.unsqueeze(1))



def get_instance_mask_2D(ids: np.array, device="cuda"):
    """
    Build an (N, N) boolean mask that is True iff two entries have the same label but different index.

    Parameters
    ----------
    score_instance_labels : np.ndarray, shape (N,)
        1-D array of instance labels.
    device : str or torch.device
        Where to place the output tensor.

    Returns
    -------
    mask : torch.BoolTensor, shape (N, N)

    >>> labels = np.array([1, 2, 1])
    >>> get_instance_mask(labels, device="cpu").tolist()
    [[False, False, True], [False, False, False], [True, False, False]]
    >>> # everything equal but self-comparisons are masked out
    >>> labels = np.array(["a", "a", "b"])
    >>> get_instance_mask(labels, device="cpu").tolist()
    [[False, True, False], [True, False, False], [False, False, False]]
    """
    assert isinstance(ids, np.ndarray), \
        "Input must be a numpy array"
    assert ids.ndim == 1, \
        f"Expected 1-D array, got ndim={ids.ndim}"
    
    
    assert isinstance(ids, np.ndarray), "Input type is expected to be numpy array"
    
    #score_instance_labels = np.array(score_instance_labels)
    labels_equal = ids[:, np.newaxis] == ids[np.newaxis, :]
    labels_equal = torch.tensor(labels_equal, device=device, dtype=torch.bool)
    indices_equal = torch.eye(ids.shape[0], device=device).bool()
    indices_not_equal = ~indices_equal
    
    return labels_equal & indices_not_equal


def get_instance_mask_3D(ids: np.array, device="cuda"):
    """
    Build an (N, N, N) boolean mask that is True iff three entries have the same label but different index.

    Parameters
    ----------
    score_instance_labels : np.ndarray, shape (N,)
        1-D array of instance labels.
    device : str or torch.device
        Where to place the output tensor.

    Returns
    -------
    mask : torch.BoolTensor, shape (N, N, N)
    """
    assert isinstance(ids, np.ndarray), \
        "Input must be a numpy array"
    assert ids.ndim == 1, \
        f"Expected 1-D array, got ndim={ids.ndim}"
    
    
    assert isinstance(ids, np.ndarray), "Input type is expected to be numpy array"

    label_equal = (ids[:, np.newaxis] == ids[np.newaxis, :] )
    i_equal_j = label_equal[:, :, np.newaxis]
    i_equal_k = label_equal[:, np.newaxis, :]
    valid_labels = i_equal_k & i_equal_j    

    labels_equal = torch.tensor(valid_labels, device=device, dtype=torch.bool)
    indices_equal = torch.eye(ids.shape[0], device=device).bool()
    indices_not_equal = ~indices_equal
    
    return labels_equal & indices_not_equal

