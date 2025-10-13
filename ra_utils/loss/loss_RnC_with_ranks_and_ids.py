"""
Original file from:  https://github.com/kaiwenzha/Rank-N-Contrast/blob/main/loss.py


@inproceedings{zha2023rank,
    title={Rank-N-Contrast: Learning Continuous Representations for Regression},
    author={Zha, Kaiwen and Cao, Peng and Son, Jeany and Yang, Yuzhe and Katabi, Dina},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023}
}

Modifications: 
- Add Delta_id 
- Use different label difference class. -> No absolute values for time 

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
import numpy as np
from typing import Optional



def get_instance_mask_2D(ids: np.ndarray, device="cuda"):
    """(n,n) mask: True iff same id AND i!=j."""
    assert isinstance(ids, np.ndarray) and ids.ndim == 1
    eq = ids[:, None] == ids[None, :]
    eq = torch.as_tensor(eq, device=device, dtype=torch.bool)
    offdiag = ~torch.eye(len(ids), device=device, dtype=torch.bool)
    return eq & offdiag  # (n,n) bool


class LabelDifference(nn.Module):
    def __init__(self, distance_type: Literal['l1', 'scalar difference'] = 'l1'):
        super().__init__()
        self.distance_type = distance_type

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        # labels: [n, d]
        if self.distance_type == 'l1':
            return torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)  # [n,n]
        elif self.distance_type == 'scalar difference':
            assert labels.ndim == 2 and labels.shape[1] == 1, "expected [n,1] labels"
            return labels[:, None, 0] - labels[None, :, 0]  # D_ij = y_i - y_j  note that D_ji = - D_ij
        else:
            raise ValueError(self.distance_type)




class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type: Literal["cosine","l2","euclidean squared","euclidean","tanh_euclidean squared"]='cosine'):
        super().__init__()
        self.similarity_type = similarity_type

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Return a similarity (higher is more similar). Adjust here if your pairwise_distances returns distances.
        if self.similarity_type == 'l2':
            # negative L2 distance as similarity
            return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        elif self.similarity_type == 'cosine':
            x = F.normalize(features, dim=-1)
            return (x @ x.t())  # cosine similarity
        else:
            raise ValueError(self.similarity_type)




class RnCIdLoss(nn.Module):
    """
    Rank-N-Contrast with instance (patient) constraint and time-ordered denominator.
    Full [n, n] version (no off-diagonal compaction).

    Loss:
      For each anchor i and each positive j with same id,
        L_ij = -log( exp(sim(i,j)/t) / sum_{k in S_ij^id} exp(sim(i,k)/t) ),
      where S_ij^id = { k != i | same-id, and D[i,k] >= D[i,j] }.
      Average per anchor by #positives N_i^{id}; then mean over anchors with N_i^{id} > 0.
    """
    def __init__(
        self,
        temperature: float = 2.0,
        feature_sim: Literal["cosine", "l2"] = "cosine",
        label_diff: Literal["l1", "scalar difference"] = "scalar difference",
        ignore_ids: bool = False,
    ):
        super().__init__()
        self.t = float(temperature)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)
        self.label_diff_fn = LabelDifference(label_diff)
        self.ignore_ids = ignore_ids
        self.label_difference_is_metric = label_diff in ["l1"]
        # print(f" label_difference_is_metric = {self.label_difference_is_metric}")
        # If there is a rank based inequality then we have for S^id_ij = {k != i; id[i] = id[j] = id[k]; y_k >= y_j}  
        # since the relative distance to the anchor drops out


    @staticmethod
    def _repeat_ids_for_two_crops(ids_1d: np.ndarray) -> np.ndarray:
        # Match feats = [x0..xB-1, x0'..xB-1']
        return np.tile(ids_1d, 2)

    def forward(self, features: torch.Tensor, labels: torch.Tensor, ids: Optional[np.ndarray] = None) -> torch.Tensor:
        assert features.ndim == 3 and features.shape[1] == 2, "features must be [B,2,D]"
        B = features.shape[0]
        assert labels.shape[0] == B, "labels and features batch size must match"

        device = features.device

        # Stack crops and labels
        feats = torch.cat([features[:, 0], features[:, 1]], dim=0)   # [n, D]
        labs  = labels.repeat(2, 1)                                  # [n, d]
        if not self.label_difference_is_metric: 
            assert labs.shape[1] == 1
            labs_ = labs.squeeze()


        n = feats.shape[0]                                           # n = 2B

        sim  = self.feature_sim_fn(feats) / self.t                   # [n, n]
        Dmat = self.label_diff_fn(labs)                              # [n, n]

        # same-id mask (full [n,n], diag disabled)
        if (not self.ignore_ids) and (ids is not None):
            ids_2 = self._repeat_ids_for_two_crops(ids)
            same_id_full = get_instance_mask_2D(ids_2, device=device)
        else:
            same_id_full = ~torch.eye(n, device=device, dtype=torch.bool)

        # counts per anchor
        N_id = same_id_full.sum(dim=1)                               # [n]
        valid_anchor = N_id > 0
        N_id_safe = N_id.clamp_min(1)

        per_anchor_sum = torch.zeros(n, device=device)
        neg_inf = float('-inf')

        for j in range(n):
            # first index = i -> average 
            # second index = k
            pos_mask_j = same_id_full[:, j]                          # [n] δ_{ij}^{id}
            if not pos_mask_j.any():
                continue

            pos_logits = sim[:, j]                                   # [n]
            
            if self.label_difference_is_metric: 
                # denom: same-id & D[i,k] >= D[i,j]
                D_inequality = (Dmat >= Dmat[:, j].unsqueeze(1))
            else: 
                # Note that iff signed_distance  
                # y_k >= y_j   ->  y_k - y_j = D_kj >= 0.          
                D_inequality = (labs_ >= labs_[j]).unsqueeze(0)   # for any i: y_k >= y_j
            denom_mask = same_id_full & D_inequality  # [n, n] bool   # mult. with delta^id_[i, k]

            sim_masked = sim.masked_fill(~denom_mask, neg_inf)       # [n, n]
            denom_log = torch.logsumexp(sim_masked, dim=1)           # [n]   # Sum over second index -> second index is k

            # >>> NaN-safe guard: zero-out invalid (i,j) before subtraction <<<
            denom_log = torch.where(pos_mask_j, denom_log, pos_logits)


            neg_log_prob = -(pos_logits - denom_log) * pos_mask_j.float()  # [n]


            per_anchor_sum += neg_log_prob

        per_anchor_avg = per_anchor_sum / N_id_safe
        if valid_anchor.any():
            loss = per_anchor_avg[valid_anchor].mean()
        else:
            loss = per_anchor_avg.mean() * 0.0

        return loss





# class RnCIdLoss(nn.Module):
#     """
#     Rank-N-Contrast with instance (patient) constraint and time-ordered denominator.

#     features: [B, 2, D]  two crops per anchor -> will be stacked to [n=2B, D]
#     labels:   [B, 1]     time scalar (e.g., days since a reference)
#     ids:      np.ndarray length B (instance ids, strings ok)
#     """
#     def __init__(
#         self,
#         temperature: float = 2.0,
#         feature_sim: Literal["cosine","l2"] = "cosine",
#         label_diff: Literal["l1","scalar difference"] = "scalar difference",
#         eps: float = 1e-12,
#     ):
#         super().__init__()
#         self.t = float(temperature)
#         self.eps = float(eps)
#         self.label_diff_fn = LabelDifference(label_diff)
#         self.feature_sim_fn = FeatureSimilarity(feature_sim)
#         self.ignore_ids = False



#     def forward(self, features, labels, ids=None):
#         # features: [bs, 2, feat_dim]
#         # labels: [bs, label_dim]
#         # ids: None or np.array of shape [bs]. -> strings are allowed

#         device = features.device
#         use_ids = ((not self.ignore_ids) and (ids is not None))
#         if use_ids:
#             mask_ids = get_instance_mask_2D(ids, device = device)  # [bs, bs]
#             N_delta = mask_ids.sum(1) # how many matches? [bs]
#         else: 
#             bs, _ = labels.shape
#             N_delta = torch.ones(bs, device=device) * (bs - 1)


#         features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
#         labels = labels.repeat(2, 1)  # [2bs, label_dim]
#         label_diffs = self.label_diff_fn(labels)  # be CAREFULL WITH THE SIGNEd LABELS TODO 


#         logits = self.feature_sim_fn(features).div(self.t) # CW: [bs, bs]

        
#         if use_ids: 
#             # Remove maximum logit with same id
#             logits_id = 
#             logits_max, _ = torch.max(logits, dim=1, keepdim=True)
#             logits -= logits_max.detach()
#             exp_logits = logits.exp()
#         else: 
#             logits_max, _ = torch.max(logits, dim=1, keepdim=True)
#             logits -= logits_max.detach()
#             exp_logits = logits.exp()

#         n = logits.shape[0]  # n = 2bs

#         # remove diagonal
#         logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
#         exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
#         label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

#         loss = 0.
#         for k in range(n - 1):
#             pos_logits = logits[:, k]  # 2bs
#             pos_label_diffs = label_diffs[:, k]  # 2bs
#             neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
#             pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # 2bs
#             loss += - (pos_log_probs / (n * (n - 1))).sum()

#         return loss
