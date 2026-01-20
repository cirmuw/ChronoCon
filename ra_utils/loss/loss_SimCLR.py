"""
loss_SimCLR.py

SimCLR Contrastive Loss Implementation

Based on: "A Simple Framework for Contrastive Learning of Visual Representations"
Chen et al., ICML 2020
https://arxiv.org/abs/2002.05709

The loss maximizes agreement between two augmented views of the same image
while minimizing agreement with other images in the batch.

Loss formulation:
    L = -log(exp(sim(z_i, z_j^+) / τ) / Σ_k exp(sim(z_i, z_k) / τ))

where:
    - z_i, z_j^+ are projected features from two augmented views of the same image
    - z_k are all other samples in the batch (negatives)
    - τ is the temperature parameter
    - sim is cosine similarity (or negative L2 distance)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


# class SimCLRLoss(nn.Module):
#     """
#     SimCLR Contrastive Loss
    
#     Args:
#         temperature: Temperature parameter for scaling logits (default: 0.07)
#         feature_sim: Similarity metric - "cosine" (default) or "negative_l2"
#     """
#     def __init__(self, temperature: float = 0.07, feature_sim: Literal["cosine", "negative_l2"] = "cosine"):
#         super(SimCLRLoss, self).__init__()
#         self.temperature = temperature
#         self.feature_sim = feature_sim
    
#     def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
#         """
#         Compute SimCLR contrastive loss between two sets of projected features.
        
#         Args:
#             z1: Projected features from first augmentation [B, D]
#             z2: Projected features from second augmentation [B, D]
#                 (z1[i] and z2[i] are positive pairs)
        
#         Returns:
#             Scalar loss value
#         """
#         batch_size = z1.shape[0]
#         device = z1.device
        
#         # Normalize features for cosine similarity
#         if self.feature_sim == "cosine":
#             z1_norm = F.normalize(z1, dim=-1)
#             z2_norm = F.normalize(z2, dim=-1)
#         elif self.feature_sim == "negative_l2":
#             z1_norm = z1
#             z2_norm = z2
#         else:
#             raise ValueError(f"Unknown feature_sim: {self.feature_sim}")
        
#         # Concatenate all features: [z1; z2] -> [2B, D]
#         features = torch.cat([z1_norm, z2_norm], dim=0)
        
#         # Compute similarity matrix
#         if self.feature_sim == "cosine":
#             similarity_matrix = torch.matmul(features, features.T)  # [2B, 2B]
#         elif self.feature_sim == "negative_l2":
#             # Negative L2 distance as similarity
#             similarity_matrix = -torch.cdist(features, features, p=2)  # [2B, 2B]
        
#         # Apply temperature scaling
#         similarity_matrix = similarity_matrix / self.temperature
        
#         # Create mask for positive pairs
#         # Positive pairs: (i, i+B) and (i+B, i) for i in [0, B)
#         # For each anchor, the positive is its augmented view
#         # Anchor i (0 <= i < B): positive is i+B
#         # Anchor i+B (B <= i < 2B): positive is i-B
#         pos_mask = torch.zeros_like(similarity_matrix, dtype=torch.bool)
#         pos_mask[:batch_size, batch_size:] = torch.eye(batch_size, device=device, dtype=torch.bool)
#         pos_mask[batch_size:, :batch_size] = torch.eye(batch_size, device=device, dtype=torch.bool)
        
#         # Mask to exclude self-contrast (diagonal)
#         eye_mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
        
#         # Compute log probabilities
#         # Standard SimCLR approach: compute max from full matrix (including diagonal) for numerical stability
#         logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
#         logits = similarity_matrix - logits_max.detach()  # Numerical stability
        
#         # Mask diagonal entries before exp (set to large negative so exp ≈ 0)
#         # This excludes self-contrast from the denominator
#         large_neg_value = -1e9
#         logits_masked = logits.masked_fill(eye_mask, large_neg_value)
#         exp_logits = torch.exp(logits_masked)
        
#         # Sum over all entries (diagonal entries contribute ~0 due to exp(large_neg) ≈ 0)
#         log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-10)
        
#         # Extract log probability of positive pairs
#         # Only consider anchors that have valid positive pairs
#         pos_log_prob = (pos_mask * log_prob).sum(dim=1)  # Sum over positive positions
#         num_pos_per_anchor = pos_mask.sum(dim=1).float()  # Should be 1.0 for each anchor
        
#         # Avoid division by zero (shouldn't happen but be safe)
#         mean_log_prob_pos = pos_log_prob / torch.clamp(num_pos_per_anchor, min=1.0)
        
#         # Average over all anchors
#         loss = -mean_log_prob_pos.mean()
        
#         return loss




class SimCLRLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        assert z1.ndim == 2 and z2.ndim == 2, "Expected [B, D] tensors"
        assert z1.shape == z2.shape, "z1 and z2 must have same shape"
        B = z1.shape[0]
        device = z1.device

        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        feats = torch.cat([z1, z2], dim=0)  # [2B, D]

        logits = (feats @ feats.T) / self.temperature  # [2B, 2B]

        # mask self-similarity
        mask = torch.eye(2 * B, device=device, dtype=torch.bool)
        logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)

        # targets: i -> i+B, i+B -> i
        targets = torch.arange(2 * B, device=device)
        targets = (targets + B) % (2 * B)

        loss = F.cross_entropy(logits, targets)
        return loss