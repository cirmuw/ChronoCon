import torch
import torch.nn as nn
import torch.nn.functional as F
import torch


class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: str = "none"):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

        Args:
            alpha (float): Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default: 0.25.
            gamma (float): Exponent of the modulating factor (1 - p_t) to
                    balance easy vs hard examples. Default: 2.
            reduction (string): 'none' | 'mean' | 'sum'
                    'none': No reduction will be applied to the output.
                    'mean': The output will be averaged.
                    'sum': The output will be summed. Default: 'none'.
        """
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the loss calculation.

        Args:
            inputs (Tensor): A float tensor of arbitrary shape.
                    The predictions for each example.
            targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        Returns:
            Loss tensor with the reduction option applied.
        """
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(
                f"Invalid value for 'reduction': '{self.reduction}'\n"
                "Supported reduction modes: 'none', 'mean', 'sum'"
            )
