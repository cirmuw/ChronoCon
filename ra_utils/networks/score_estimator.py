import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


# class ScalarScoreEstimatorFromLogits(nn.Module):
#     def __init__(self, mode: Literal["argmax", "expectation_value", "learnable"]):
#         super().__init__()
#         self.mode = mode 
        
#         if 
        


def estimate_scalar_score_from_logits(
    logits: torch.Tensor,
    mode: Literal["argmax", "expectation_value"]
) -> torch.Tensor:
    """
    Given a [batch_size, num_classes] logits tensor, returns either:
      - the argmax class index per batch element, or
      - the expectation value over class indices.

    Examples:
        >>> import torch
        >>> # argmax mode
        >>> logits = torch.tensor([[0.1, 2.0, -1.0], [5.0, 1.0, 1.0]])
        >>> estimate_scalar_score_from_logits(logits, "argmax")
        tensor([1, 0])

        >>> # expectation_value mode with uniform logits → mean of [0,1,2] == 1.0
        >>> uniform_logits = torch.zeros(1, 3)
        >>> estimate_scalar_score_from_logits(uniform_logits, "expectation_value")
        tensor([1.])

        >>> # mismatched dims raises ValueError
        >>> estimate_scalar_score_from_logits(torch.tensor([1,2,3]), "argmax")
        Traceback (most recent call last):
        ...
        ValueError: Expected a 2D logits tensor, got 1D.

        >>> # unsupported mode raises ValueError
        >>> estimate_scalar_score_from_logits(torch.zeros(1,3), "foo")
        Traceback (most recent call last):
        ...
        ValueError: Unsupported mode: 'foo'. Choose 'argmax' or 'expectation_value'.
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected a 2D logits tensor, got {logits.ndim}D.")

    if mode == "argmax":
        # returns LongTensor of indices
        return torch.argmax(logits, dim=-1)

    elif mode == "expectation_value":
        # compute class probabilities along the last dim
        probs = F.softmax(logits, dim=-1)

        # build a [num_classes] index tensor on the right device/dtype
        num_classes = logits.size(-1)
        classes = torch.arange(
            num_classes, 
            device=logits.device, 
            dtype=probs.dtype
        )  # shape: [num_classes]

        # expand classes to [1, num_classes] and compute the expectation
        # result is shape [batch_size]
        return (probs * classes.unsqueeze(0)).sum(dim=-1)

    else:
        raise ValueError(f"Unsupported mode: {mode!r}. Choose 'argmax' or 'expectation_value'.")
