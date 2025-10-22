"""
loss_RnCMono.py

Original file from:  https://github.com/kaiwenzha/Rank-N-Contrast/blob/main/loss.py


@inproceedings{zha2023rank,
    title={Rank-N-Contrast: Learning Continuous Representations for Regression},
    author={Zha, Kaiwen and Cao, Peng and Son, Jeany and Yang, Yuzhe and Katabi, Dina},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023}
}

Modifications: 

- Use different label difference class.
- Add ids
- Symmetric, monotonicity-aware extension of Rank-N-Contrast:
   encourages embeddings to evolve monotonically over time within each subject.

    L_RnC_time = -(1 / 2N) * Σ_i [
          (1 / |S_i^<|) Σ_{j ∈ S_i^<} log ( exp(s_ij) / Σ_{k ∈ S_ij^<} exp(s_ik) )
        + (1 / |S_i^>|) Σ_{j ∈ S_i^>} log ( exp(s_ij) / Σ_{k ∈ S_ij^>} exp(s_ik) )
    ]

    with

    S_lesser_ij  = {v_k | k != i   t_i <= t_j <= t_k   id_i = id_j = id_k }  
    S_greater_ij = {v_k | k != i   t_i >= t_j >= t_k   id_i = id_j = id_k }  

    S_lesser_i  = {j | j != i   t_i <= t_j    id_i = id_j }  
    S_greater_i = {j | j != i   t_i >= t_j    id_i = id_j }  

    
    Maintains the original RnC contrastive structure while enforcing
    subject-specific temporal ordering (t_i ≤ t_j ≤ t_k).
# ------------------------------------------------------------------------------


"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
import numpy as np
from typing import Optional


def get_instance_mask_2D_without_diag(ids: np.ndarray, device="cuda"):
    """(n,n) mask: True iff same id AND i!=j."""
    assert isinstance(ids, np.ndarray) and ids.ndim == 1
    eq = ids[:, None] == ids[None, :]
    eq = torch.as_tensor(eq, device=device, dtype=torch.bool)
    offdiag = ~torch.eye(len(ids), device=device, dtype=torch.bool)
    return eq & offdiag  # (n,n) bool


def get_instance_mask_2D_with_diag(ids: np.ndarray, device="cuda"):
    """(n,n) mask: True iff same id"""
    assert isinstance(ids, np.ndarray) and ids.ndim == 1
    eq = ids[:, None] == ids[None, :]
    eq = torch.as_tensor(eq, device=device, dtype=torch.bool)
    return eq


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
            return labels[None, :, 0] - labels[:, None, 0]   # D_ij = y_j - y_i  note that D_ji = - D_ij
        else:
            raise ValueError(self.distance_type)




class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type: Literal["cosine",'negative l1', 'negative l2']='cosine'):
        super().__init__()
        self.similarity_type = similarity_type

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Return a similarity (higher is more similar). Adjust here if your pairwise_distances returns distances.
        if self.similarity_type == 'negative l2':
            # negative L2 distance as similarity
            return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        if self.similarity_type == 'negative l1':
            # negative L1 distance as similarity
            return - (features[:, None, :] - features[None, :, :]).norm(1, dim=-1)
        elif self.similarity_type == 'cosine':
            x = F.normalize(features, dim=-1)
            return (x @ x.t())  # cosine similarity
        else:
            raise ValueError(self.similarity_type)




class RnCLossMono(nn.Module):  
    """
    Symmetric, monotonicity-aware extension of Rank-N-Contrast (RnC).
    Within each subject (id), it enforces temporal ordering for positives and
    uses a time-ordered denominator for each (i,j) term.

    Notation
    --------
    Let s_{ij} = sim(z_i, z_j) / τ be the (temperature-scaled) similarity.
    For each anchor i we define the time-ordered positive sets
        S_i^<  = { j | j ≠ i, id_j = id_i,  t_i ≤ t_j }
        S_i^>  = { j | j ≠ i, id_j = id_i,  t_i ≥ t_j }

    and, for each (i,j), the corresponding time-ordered denominator sets
        S_{ij}^< = { k | k ≠ i, id_k = id_i = id_j,  t_i ≤ t_j ≤ t_k }   (includes k = j)
        S_{ij}^> = { k | k ≠ i, id_k = id_i = id_j,  t_i ≥ t_j ≥ t_k }   (includes k = j)

    The per-pair contrastive term is
        l_{ij}^< = - log ( exp(s_{ij}) / Σ_{k ∈ S_{ij}^<} exp(s_{ik}) ),
        l_{ij}^> = - log ( exp(s_{ij}) / Σ_{k ∈ S_{ij}^>} exp(s_{ik}) ).

    Note: ties (t_i = t_j or t_j = t_k) are included (≤ / ≥), so a tied (i,j) can
    appear in both “<” and “>” branches.

    Normalization modes
    -------------------
    The branch losses (L^<, L^>) are aggregated from the per-pair terms in one of
    three ways; the final loss is L = L^< + L^>.

    1) Anchor-balanced  (per-anchor mean of per-pair losses)
        I^<  = { i | |S_i^<| > 0 },   I^> = { i | |S_i^>| > 0 }
        L^<  = (1 / |I^<|) Σ_{i ∈ I^<}  (1 / |S_i^<|) Σ_{j ∈ S_i^<}  l_{ij}^<
        L^>  = (1 / |I^>|) Σ_{i ∈ I^>}  (1 / |S_i^>|) Σ_{j ∈ S_i^>}  l_{ij}^>

    2) Pair-balanced  (uniform over all valid (i,j) pairs)
        P^<  = { (i,j) | j ∈ S_i^< },   P^> = { (i,j) | j ∈ S_i^> }
        L^<  = (1 / |P^<|) Σ_{(i,j) ∈ P^<}  l_{ij}^<
        L^>  = (1 / |P^>|) Σ_{(i,j) ∈ P^>}  l_{ij}^>

        (Trivial pairs with |S_{ij}^•| = 1 contribute 0 because l_{ij}^• = 0.)

    3) Pair-balanced positive  (uniform over non-trivial (i,j))   DEFAULT
        P_+^< = { (i,j) ∈ P^< | |S_{ij}^<| ≥ 2 },   P_+^> analogously
        L^<   = (1 / |P_+^<|) Σ_{(i,j) ∈ P_+^<}  l_{ij}^<
        L^>   = (1 / |P_+^>|) Σ_{(i,j) ∈ P_+^>}  l_{ij}^>
    """
      
    def __init__(
        self,
        temperature: float = 2.0,
        feature_sim: Literal["cosine",'negative l1', 'negative l2'] = "cosine",
        label_diff: Literal["l1", "scalar difference"] = "scalar difference",
        ignore_ids: bool = False,
        normalization: Literal["pair balanced", "anchor balanced", "pair balanced positive"] = "pair balanced positive",
        #return_support = True
    ):


        super().__init__()
        self.tau = float(temperature)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim = feature_sim
        self.label_diff = label_diff
        #self.return_support = return_support

        self.ignore_ids = ignore_ids
        self.normalization = normalization

    def __repr__(self):
        return f"RnCLossMono(τ = {self.tau}, " \
               f"feature_sim = {self.feature_sim}, label_diff = {self.label_diff}, " \
               f"ignore_ids = {self.ignore_ids}, "\
               f"normalization = {self.normalization})"

    @staticmethod
    def _repeat_ids_for_two_crops(ids_1d: np.ndarray) -> np.ndarray:
        # Match feats = [x0..xB-1, x0'..xB-1']
        return np.tile(ids_1d, 2)


    @staticmethod
    def _sum_contrastive_loss_terms(sim: torch.Tensor, 
                                    S_mask_ij, 
                                    S_mask_ijk,
                                    normalization: Literal["pair balanced", "anchor balanced", "pair balanced positive"] = "pair balanced positive"
                                    ):
        # sim: [2*B, 2*B]
        device = sim.device
        n = sim.shape[0]
        
        per_anchor_sum = torch.zeros(n, device=device)
        N_pos = torch.zeros((), device=device, dtype=torch.long)
        neg_inf = float('-inf')
        for j in range(n):
            # first index = i -> average; The second index of the matrix is k -> logsumexp
            pos_mask_j = S_mask_ij[:, j]                          
            if not pos_mask_j.any():
                continue
            pos_logits = sim[:, j]                                   # [n]
            denom_mask = S_mask_ijk[:, j, :]  # i, k
            sim_masked = sim.masked_fill(~denom_mask, neg_inf)       # [n, n]
            denom_log = torch.logsumexp(sim_masked, dim=1)           # [n]   # Sum over second index -> second index is k

            # >>> NaN-safe guard: zero-out invalid (i,j) before subtraction <<<
            denom_log = torch.where(pos_mask_j, denom_log, pos_logits)
            neg_log_prob = -(pos_logits - denom_log) * pos_mask_j.float()  # [n]
            per_anchor_sum += neg_log_prob

            # Sum the non-trivial i, j terms; 
            # Empty sets as well as those with a single term  only (-log(p) + log(p) = 0) are trivial. 
            # structural non-triviality: |S^<_{ij}| >= 2
            denom_size_per_i = denom_mask.sum(dim=1)             # [n]
            contrib_pairs_j = pos_mask_j & (denom_size_per_i >= 2)
            N_pos += contrib_pairs_j.sum()
        
        dtype = per_anchor_sum.dtype
        N_id = S_mask_ij.sum(dim=1).to(dtype)         # per-anchor |S^<_{i}|
        valid_anchor = N_id > 0
        N_valid_anchors = valid_anchor.sum()
        
        if normalization == "pair balanced positive":
            N_norm = N_pos.to(dtype) #.clamp_min(1)
            loss = per_anchor_sum.sum() / (N_norm + 1e-10)

        elif normalization == "pair balanced":
            N_norm = S_mask_ij.sum().to(dtype).clamp_min(1)
            loss = per_anchor_sum.sum() / (N_norm + 1e-10)

        elif normalization == "anchor balanced":
            N_norm = N_valid_anchors
            l_avg = per_anchor_sum / (N_id + 1e-10)
            loss = l_avg[valid_anchor].mean() if valid_anchor.any() else l_avg.mean() * 0.0

        else:
            raise ValueError(f"unknown normalization: {normalization}")        

        support = {
            "num_terms_normalization": N_norm.item(),
            "num_valid_anchors": N_valid_anchors.item(),
            "num_pos": N_pos.item()
        }
        return loss, support



    def forward(self, features: torch.Tensor, labels: torch.Tensor, ids: Optional[np.ndarray] = None, 
                return_support=True) -> torch.Tensor:
        """
        features : shape [B, 2, D]  Two encoded crops/ self-transforms. 
        labels   : shape [B, N_l]   Usually the label is the "time" -> shape [B, 1]
        ids      : shape [B]        Can be numpy array of strings. E.g. np.array(["pat001_L_PIPII", "pat002_L_PIPII", ...])
        """

        assert features.ndim == 3 and features.shape[1] == 2, "features must be [B,2,D]"
        B = features.shape[0]
        assert labels.shape[0] == B, "labels and features batch size must match"

        device = features.device

        # Stack crops and labels
        feats = torch.cat([features[:, 0], features[:, 1]], dim=0)   # [n, D]
        labs  = labels.repeat(2, 1)                                  # [n, d]


        n = feats.shape[0]                                           # n = 2B

        sim  = self.feature_sim_fn(feats) / self.tau                 # [n, n]
        Dmat = self.label_diff_fn(labs)                              # [n, n]

        #-----------------------------------#
        # Prepare the sets

        # same-id mask (full [n,n], diag disabled)
        off_diag = ~torch.eye(n, device=device, dtype=torch.bool)

        if (not self.ignore_ids) and (ids is not None):
            ids_2 = self._repeat_ids_for_two_crops(ids)
            same_id_full = get_instance_mask_2D_with_diag(ids_2, device=device)
        else:
            same_id_full = torch.full((n, n), fill_value=True, device=device)

        # t_i <= t_j   id_i = id_j   i!=j
        # Dmat_ij = t_j - t_i  >= 0  <->  t_j >= t_i  
        S_lesser_mask_ij = off_diag & (Dmat >= 0)  &  same_id_full

        # t_i >= t_j   id_i = id_j   i!=j
        # Dmat_ij = t_j - t_i  <= 0  <->  t_j <= t_i  
        S_greater_mask_ij = off_diag & (Dmat <= 0)  &  same_id_full  # index order: i,j


        # three index masks i,j,k
        off_diag_ik = ~torch.eye(n, device=device, dtype=torch.bool)  # [i,k]
        off_diag_ik = off_diag_ik[:, None, :]  # [i,1,k] -> broadcast over j
        ids_are_same = same_id_full.unsqueeze(2) & same_id_full.unsqueeze(0)  # id_i = id_j = id_k     

        # S_lesser_mask_ijk 
        # t_i <= t_j   <->  0 <=  t_j - t_i   <-> 0 <= Dij 
        #                               (t_i <= t_j)       &       (t_j <= t_k)  
        mask_ti_leq_tj_leq_tk = ( (0 <= Dmat).unsqueeze(2) & (0 <= Dmat).unsqueeze(0) )
        S_lesser_mask_ijk = off_diag_ik & (ids_are_same & mask_ti_leq_tj_leq_tk)

        # S_greater_mask_ijk 
        # 
        #     t_i >= t_j   <->  0 >=  t_j - t_i   <-> 0 >= Dij 
        #                               (t_i >= t_j)       &       (t_j >= t_k)          
        mask_ti_geq_tj_geq_tk = ( (0 >= Dmat).unsqueeze(2) & (0 >= Dmat).unsqueeze(0) )
        #mask_ti_geq_tj_geq_tk = ( (0 >= D3_ij) & (0 >= D3_jk) )
        S_greater_mask_ijk = off_diag_ik & (ids_are_same & mask_ti_geq_tj_geq_tk)



        #------------------------------------#
        # Lesser part of the loss: 
        loss_l, sup_l = self._sum_contrastive_loss_terms(
            sim=sim, 
            S_mask_ij=S_lesser_mask_ij, 
            S_mask_ijk=S_lesser_mask_ijk, 
            normalization=self.normalization)

        loss_g, sup_g = self._sum_contrastive_loss_terms(
            sim=sim, 
            S_mask_ij=S_greater_mask_ij, 
            S_mask_ijk=S_greater_mask_ijk, 
            normalization=self.normalization)

        loss = loss_l + loss_g

        if return_support: 
            support = {
                # "loss_lesser": loss_l.item(),
                # "loss_greater": loss_g.item(),
                # combined terms
                "num_terms_normalization": (sup_g["num_terms_normalization"] + sup_l["num_terms_normalization"]),       # This depends on the normalization model ( -> use for average over full dataset)          
                "ratio lesser / total": loss_l.item() / (loss_g.item() + loss_l.item() +  1.0e-10),
                #  lesser terms 
                "num_terms_normalization_lesser": sup_l["num_terms_normalization"], 
                "num_pos_lesser": sup_l["num_pos"], 
                "num_valid_anchors_lesser": sup_l["num_valid_anchors"], 
                #  greater terms
                "num_terms_normalization_greater": sup_g["num_terms_normalization"],
                "num_pos_greater": sup_g["num_pos"], 
                "num_valid_anchors_greater": sup_g["num_valid_anchors"], 
            }    
            return loss, support
        else: 
            return loss








