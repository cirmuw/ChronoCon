import torch
import torch.nn as nn

class DeltaHeadLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, *args, **kwargs):
        return 0.0
    