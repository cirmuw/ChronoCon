"""
delta_heads.py


Models to compare two images already in latent space. 
"""

import torch
import torch.nn as nn
from  ra_utils.networks.architecture import make_score_type_2_head_name_dct

from typing import Optional, Literal, List, Dict, Tuple, Union
from ra_utils.networks.architecture import make_mlp


class DeltaHead(nn.Module):
    """
    Efficient delta head that compares pairs of latent representations.
    
    Takes a batch of latent vectors and efficiently processes all pairs through an MLP
    to predict progression (classification or regression).
    
    Args:
        latent_dim_single: Dimension of a single latent vector
        out_dim: Output dimension (1 for regression/binary classification, >1 for multi-class)
        mlp_kwargs: Additional kwargs for make_mlp (depth, dropout_op, etc.)
    
    Example:
        >>> delta_head = DeltaHead(
        ...     latent_dim_single=512,
        ...     out_dim=1,
        ... )
        >>> z = torch.randn(8, 512)  # Batch of 8 latent vectors
        >>> predictions = delta_head(z)  # Shape: [8, 8, 1] (includes all pairs including diagonal)
    """
    def __init__(self,
                 latent_dim_single: int = 512,
                 out_dim: int = 1,
                 mlp_kwargs: dict = None):
        super(DeltaHead, self).__init__()
        
        if mlp_kwargs is None:
            mlp_kwargs = {"depth": 2, "dropout_op": nn.Dropout, "dropout_op_kwargs": {"p": 0.2}}
        
        self.out_dim = out_dim
        self.latent_dim_single = latent_dim_single
        
        # MLP takes concatenated pair of latent vectors as input
        self.mlp = make_mlp(
            latent_dim=latent_dim_single * 2,
            out_dim=out_dim,
            **mlp_kwargs
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through delta head.
        
        Computes all pairwise comparisons (any versus any) in the batch, including self-comparisons.
        Filtering of self-comparisons should be done at the loss function level.
        
        Args:
            z: Latent vectors [B, latent_dim_single]
        
        Returns:
            predictions: [B, B, out_dim] tensor containing all pairs (including diagonal)
        """
        B, F = z.shape
        assert F == self.latent_dim_single, \
            f"z.shape = {z.shape} but expected latent_dim_single = {self.latent_dim_single}"
        
        # Efficiently create all pairs: (B, B) pairs
        # z_i: [B, 1, F] -> [B, B, F] (broadcasted)
        # z_j: [1, B, F] -> [B, B, F] (broadcasted)
        z_i = z.unsqueeze(1).expand(B, B, F)  # [B, B, F]
        z_j = z.unsqueeze(0).expand(B, B, F)  # [B, B, F]
        
        # Concatenate pairs: [B, B, 2*F]
        pairs = torch.cat([z_i, z_j], dim=-1)
        
        # Reshape for MLP: [B*B, 2*F]
        pairs_flat = pairs.view(B * B, 2 * F)
        
        # Process through MLP: [B*B, out_dim]
        out_flat = self.mlp(pairs_flat)
        
        # Reshape back: [B, B, out_dim]
        out = out_flat.view(B, B, self.out_dim)
        
        return out
    
    def forward_pairs(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Convenience method to process explicit pairs of latent vectors.
        
        Args:
            z1: First latent vectors [N, latent_dim_single]
            z2: Second latent vectors [N, latent_dim_single]
        
        Returns:
            predictions: [N, out_dim]
        """
        N, F1 = z1.shape
        _, F2 = z2.shape
        
        assert F1 == self.latent_dim_single and F2 == self.latent_dim_single, \
            f"Expected latent_dim_single={self.latent_dim_single}, got z1.shape={z1.shape}, z2.shape={z2.shape}"
        assert z1.shape[0] == z2.shape[0], \
            f"z1 and z2 must have same batch size, got {z1.shape[0]} and {z2.shape[0]}"
        
        # Concatenate pairs: [N, 2*F]
        pairs = torch.cat([z1, z2], dim=-1)
        
        # Process: [N, out_dim]
        out = self.mlp(pairs)
        
        return out
    
    

import numpy as np 


class DeltaHeads(nn.Module):
    def __init__(self,
                 delta_head_infos: dict,
                 latent_dim: int = 512,
                 mlp_kwargs = {"depth": 1, 
                               "dropout_op": None}):
        super(DeltaHeads, self).__init__()
        self.delta_head_infos = delta_head_infos.copy()
        
        # input checks on classifier_head_infos -> No overlaps in score_types
        self.score_type_2_head_name = make_score_type_2_head_name_dct(delta_head_infos)
        self.heads = nn.ModuleDict({k: DeltaHead(latent_dim_single=latent_dim, 
                                                 out_dim = v["out_dim"], 
                                                 mlp_kwargs = mlp_kwargs) 
                                    for k,v in delta_head_infos.items()})
        
    def forward(
        self,
        z: torch.Tensor,            # [B, latent_dim]
        score_types: np.array,     # decides which head to use  np.array of type string ["PIPII_JSN", "PIPII_JSN", "PIPII_ERO", ...]
        ids: List[str],     # ids np.array of strings e.g. ["<patientID>_<side>_<roi_or_score_type>", ...]
    ) -> Dict[str, Dict[str, Union[torch.Tensor, List[Tuple[str, str]]]]]:
        """
        Forward pass through delta heads.
        
        Groups samples by head based on score_types, computes all pairwise comparisons
        within each group, and returns results organized by head.
        
        Args:
            z: Latent vectors [B, latent_dim]
            score_types: Array of score type strings, length B
            ids: List of id strings, length B
        
        Returns:
            Dictionary with structure:
            {
                "<head_name>": {
                    "logits_or_value": torch.Tensor [N_pairs, out_dim],
                    "index_pairs": torch.Tensor [N_pairs, 2],  # (i, j) indices in original batch
                    "ids_pairs": List[Tuple[str, str]],        # (id_i, id_j) for each pair
                },
                ...
            }
            where N_pairs = N_head * N_head includes all pairs (diagonal and off-diagonal).
            Filtering of self-comparisons (diagonal pairs) should be done at the loss function level.
        
        Note:
            To save results for image pairs, you can use the "ids_pairs" field to identify which images
            were compared, and "logits_or_value" for the predictions. For example:
            
            ```python
            results = delta_heads(z, score_types, ids)
            for head_name, head_results in results.items():
                for (id_i, id_j), pred in zip(head_results["ids_pairs"], head_results["logits_or_value"]):
                    # Save: id_i, id_j, prediction
                    save_pair_result(id_i, id_j, pred.cpu().numpy())
            ```
        """ 


        if len(z) != len(score_types):
            raise ValueError(
                f"z has {len(z)} rows but score_types has {len(score_types)} items."
            )
        if len(z) != len(ids):
            raise ValueError(
                f"z has {len(z)} rows but ids has {len(ids)} items."
            )

        device = z.device
        B = z.shape[0]
        
        # Map each sample to its head
        active_heads = [self.score_type_2_head_name[s] for s in score_types]
        
        # Initialize output dictionary
        output = {}
        
        # Process each head separately
        for head_name, head in self.heads.items():
            # Get indices of samples belonging to this head
            head_indices = torch.tensor(
                [i for i, h in enumerate(active_heads) if h == head_name],
                dtype=torch.long,
                device=device
            )
            
            if head_indices.numel() == 0:
                # No samples for this head - return empty tensors
                out_dim = self.delta_head_infos[head_name]["out_dim"]
                output[head_name] = {
                    "logits_or_value": torch.empty((0, out_dim), device=device, dtype=z.dtype),
                    "index_pairs": torch.empty((0, 2), device=device, dtype=torch.long),
                    "ids_pairs": [],  # Empty list for ids
                }
                continue
            
            # Get latent vectors for this head's samples
            z_head = z[head_indices]  # [N_head, latent_dim]
            N_head = z_head.shape[0]
            
            # Compute all pairwise comparisons using the head
            # Returns [N_head, N_head, out_dim] - includes all pairs (diagonal and off-diagonal)
            predictions_all = head(z_head)
            
            # Extract all pairs (including diagonal)
            # Create all (i, j) pairs for the head's local coordinate system
            # Create index pairs for all combinations: (0,0), (0,1), ..., (0,N-1), (1,0), ..., (N-1,N-1)
            i_local = torch.arange(N_head, device=device).unsqueeze(1).expand(N_head, N_head).flatten()
            j_local = torch.arange(N_head, device=device).unsqueeze(0).expand(N_head, N_head).flatten()
            
            # Extract predictions for all pairs using the indices
            logits_or_value = predictions_all[i_local, j_local]  # [N_head * N_head, out_dim]
            
            # Map local indices back to global batch indices
            index_pairs_global = torch.stack([
                head_indices[i_local],
                head_indices[j_local]
            ], dim=1)  # [N_head * N_head, 2]
            
            # Get ids for each pair (store as list of tuples for readability)
            ids_pairs = [
                (ids[int(idx_global[0])], ids[int(idx_global[1])])
                for idx_global in index_pairs_global.cpu().numpy()
            ]
            
            # Store results for this head
            output[head_name] = {
                "logits_or_value": logits_or_value,
                "index_pairs": index_pairs_global,
                "ids_pairs": ids_pairs,
            }
        
        return output

        
# Note: For classification, the number of classes is determined by out_dim in delta_head_infos.
# Typically, for comparison tasks, we have three classes:
#   C1 = score(image1) < score(image2)
#   C2 = score(image1) == score(image2)  
#   C3 = score(image1) > score(image2)
# C2 is typically much more likely than the others. The DeltaHeadLoss supports
# weighted cross entropy loss with class_weights to handle class imbalance.
# The number of classes per head is automatically extracted from delta_head_infos["out_dim"]. 
        



def _extract_base_id(id_str: str) -> str:
    """
    Extract base ID (patient+side+ROI) from an ID string.
    
    ID format is assumed to be: "<patientID>_<side>_<roi_or_score_type>[_<optional_time_or_other>]"
    This function extracts the first three components (patient, side, ROI) which should match
    for pairs from the same patient, side, and ROI (but potentially different time points).
    
    Args:
        id_str: ID string in format "<patientID>_<side>_<roi_or_score_type>[_...]"
    
    Returns:
        Base ID string with format "<patientID>_<side>_<roi_or_score_type>"
    """
    parts = id_str.split("_")
    if len(parts) < 3:
        # If ID doesn't have at least 3 parts, return as-is (fallback)
        return id_str
    # Return first 3 parts: patient, side, ROI
    return "_".join(parts[:3])


class DeltaHeadsLoss(nn.Module):
    """
    Loss function for multiple delta heads.
    
    Aggregates losses from individual heads, optionally with per-head weights.
    Can be initialized either from pre-created DeltaHeadLoss instances or from delta_head_infos.
    Supports filtering pairs to only compare samples with matching patient+side+ROI when ids_must_match=True.
    """
    def __init__(
        self,
        head_losses: Optional[Dict[str, "DeltaHeadLoss"]] = None,
        delta_head_infos: Optional[Dict] = None,
        head_weights: Optional[Dict[str, float]] = None,
        class_weights: Optional[Dict[str, torch.Tensor]] = None,
        task_type: Optional[Literal["classification", "regression"]] = None,
        ids_must_match: bool = False,
        no_self_difference: bool = True,
    ):
        super().__init__()
        self.ids_must_match = ids_must_match
        self.no_self_difference = no_self_difference
        
        # Create head_losses from delta_head_infos if not provided
        if head_losses is None:
            if delta_head_infos is None:
                raise ValueError(
                    "Either head_losses or delta_head_infos must be provided"
                )
            head_losses = {}
            for head_name, head_info in delta_head_infos.items():
                out_dim = head_info.get("out_dim")
                if out_dim is None:
                    raise ValueError(
                        f"delta_head_infos['{head_name}'] must contain 'out_dim'"
                    )
                
                # Get class weights for this head if provided
                head_class_weights = None
                if class_weights is not None and head_name in class_weights:
                    head_class_weights = class_weights[head_name]
                
                # Create DeltaHeadLoss for this head
                head_losses[head_name] = DeltaHeadLoss(
                    num_classes=out_dim,
                    task_type=task_type,
                    class_weights=head_class_weights,
                )
        
        self.head_losses = nn.ModuleDict(head_losses)
        
        # Set head weights
        if head_weights is None:
            # Try to extract from delta_head_infos if available
            if delta_head_infos is not None:
                head_weights = {
                    name: info.get("loss_weight", 1.0)
                    for name, info in delta_head_infos.items()
                }
            else:
                # Uniform weights
                head_weights = {name: 1.0 for name in head_losses.keys()}
        else:
            head_weights = head_weights.copy()
        
        # Ensure all heads in head_losses have weights
        for head_name in self.head_losses.keys():
            if head_name not in head_weights:
                head_weights[head_name] = 1.0
        
        self.head_weights = head_weights

    def forward(
        self,
        predictions: Dict[str, Dict[str, Union[torch.Tensor, List[Tuple[str, str]]]]],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        total_loss = None
        
        for head_name, head_loss in self.head_losses.items():
            if head_name not in predictions:
                # Head not present in predictions (empty batch for this head)
                # Get device from first non-empty head or use CPU
                if total_loss is None:
                    # Try to infer device from targets if available
                    if targets and head_name in targets:
                        device = targets[head_name].device
                    else:
                        device = next(head_loss.parameters()).device if list(head_loss.parameters()) else torch.device("cpu")
                else:
                    device = total_loss.device
                losses[head_name] = torch.tensor(0.0, device=device)
                continue
            
            if head_name not in targets:
                raise ValueError(
                    f"Head '{head_name}' present in predictions but not in targets"
                )
            
            head_pred = predictions[head_name]["logits_or_value"]
            head_target = targets[head_name]
            head_ids_pairs = predictions[head_name]["ids_pairs"]
            
            # Ensure targets match predictions
            if len(head_pred) != len(head_target):
                raise ValueError(
                    f"Head '{head_name}': predictions have {len(head_pred)} pairs, "
                    f"but targets have {len(head_target)}"
                )
            if len(head_pred) != len(head_ids_pairs):
                raise ValueError(
                    f"Head '{head_name}': predictions have {len(head_pred)} pairs, "
                    f"but ids_pairs have {len(head_ids_pairs)}"
                )
            
            # Get index_pairs for filtering
            head_index_pairs = predictions[head_name]["index_pairs"]
            if len(head_pred) != len(head_index_pairs):
                raise ValueError(
                    f"Head '{head_name}': predictions have {len(head_pred)} pairs, "
                    f"but index_pairs have {len(head_index_pairs)}"
                )
            
            # Filter pairs based on self-comparison exclusion if no_self_difference is True
            if self.no_self_difference:
                # Create mask for non-self comparisons (i != j)
                valid_mask = (head_index_pairs[:, 0] != head_index_pairs[:, 1])
                
                # Filter predictions, targets, ids_pairs, and index_pairs
                head_pred = head_pred[valid_mask]
                head_target = head_target[valid_mask]
                head_ids_pairs = [pair for pair, valid in zip(head_ids_pairs, valid_mask.cpu().tolist()) if valid]
                head_index_pairs = head_index_pairs[valid_mask]
                
                # If no valid pairs after filtering, set loss to zero
                if len(head_pred) == 0:
                    # Determine device for zero loss tensor (use original target's device)
                    if total_loss is not None:
                        device = total_loss.device
                    elif targets and head_name in targets:
                        device = targets[head_name].device
                    else:
                        device = next(head_loss.parameters()).device if list(head_loss.parameters()) else torch.device("cpu")
                    losses[head_name] = torch.tensor(0.0, device=device)
                    continue
            
            # Filter pairs based on ID matching if ids_must_match is True
            if self.ids_must_match:
                # Create mask for pairs with matching base IDs (patient+side+ROI)
                valid_mask = []
                for id_i, id_j in head_ids_pairs:
                    base_id_i = _extract_base_id(id_i)
                    base_id_j = _extract_base_id(id_j)
                    valid_mask.append(base_id_i == base_id_j)
                
                # Convert to tensor on the same device as predictions/targets
                # Use head_pred.device as primary, but ensure compatibility
                mask_device = head_pred.device
                valid_mask_tensor = torch.tensor(valid_mask, device=mask_device, dtype=torch.bool)
                
                # Filter predictions, targets, ids_pairs, and index_pairs
                head_pred = head_pred[valid_mask_tensor]
                # Ensure target mask is on correct device if targets are on different device
                if head_target.device != mask_device:
                    valid_mask_tensor = valid_mask_tensor.to(head_target.device)
                head_target = head_target[valid_mask_tensor]
                head_ids_pairs = [pair for pair, valid in zip(head_ids_pairs, valid_mask) if valid]
                head_index_pairs = head_index_pairs[valid_mask_tensor]
                
                # If no valid pairs after filtering, set loss to zero
                if len(head_pred) == 0:
                    # Determine device for zero loss tensor (use original target's device)
                    if total_loss is not None:
                        device = total_loss.device
                    elif targets and head_name in targets:
                        device = targets[head_name].device
                    else:
                        device = next(head_loss.parameters()).device if list(head_loss.parameters()) else torch.device("cpu")
                    losses[head_name] = torch.tensor(0.0, device=device)
                    continue
            
            # Compute loss for this head
            head_loss_value = head_loss(head_pred, head_target)
            losses[head_name] = head_loss_value
            
            # Initialize total_loss on first iteration
            if total_loss is None:
                total_loss = torch.tensor(0.0, device=head_loss_value.device)
            
            # Add to total with weight
            weight = self.head_weights.get(head_name, 1.0)
            total_loss = total_loss + weight * head_loss_value
        
        # If no losses were computed, return zero loss
        if total_loss is None:
            # Try to get device from any target or use CPU
            device = torch.device("cpu")
            if targets:
                for target in targets.values():
                    device = target.device
                    break
            total_loss = torch.tensor(0.0, device=device)
        
        losses["total_loss"] = total_loss
        return losses
    




class DeltaHeadLoss(nn.Module):
    """
    Loss function for a single delta head.
    
    Computes loss for pairwise comparisons. Supports both classification and regression.
    For classification: predicts one of num_classes classes (typically: score1 < score2, score1 == score2, score1 > score2).
    For regression: predicts the difference (score1 - score2).
    """
    def __init__(
        self,
        num_classes: int,
        task_type: Optional[Literal["classification", "regression"]] = None,
        class_weights: Optional[torch.Tensor] = None,
        loss_fn: Optional[nn.Module] = None,
    ):
        """
        Args:
            num_classes: Number of classes for classification (typically 3 for <, ==, > comparisons),
                        or 1 for regression. If task_type is None, inferred from num_classes (1 -> regression, >1 -> classification).
            task_type: "classification" or "regression". If None, inferred from num_classes.
            class_weights: Optional weights for classification classes [num_classes]
                          If None, uses uniform weights
            loss_fn: Optional custom loss function. If None, uses default (CE for classification, MSE for regression)
        """
        super().__init__()
        
        # Infer task_type from num_classes if not provided
        if task_type is None:
            if num_classes == 1:
                task_type = "regression"
            else:
                task_type = "classification"
        
        self.task_type = task_type
        self.num_classes = num_classes
        
        if class_weights is not None:
            if len(class_weights) != num_classes:
                raise ValueError(
                    f"class_weights length {len(class_weights)} does not match num_classes {num_classes}"
                )
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None
        
        if loss_fn is not None:
            self.loss_fn = loss_fn
        elif task_type == "classification":
            # Weighted CrossEntropyLoss if class_weights provided
            self.loss_fn = nn.CrossEntropyLoss(
                weight=class_weights,  # Pass directly, can be None
                reduction="mean"
            )
        else:  # regression
            if num_classes != 1:
                raise ValueError(
                    f"For regression, num_classes must be 1, got {num_classes}"
                )
            self.loss_fn = nn.MSELoss(reduction="mean")

    def forward(
        self,
        logits_or_value: torch.Tensor,  # [N_pairs, out_dim]
        targets: torch.Tensor,           # [N_pairs] for classification (class indices), [N_pairs] for regression (differences)
    ) -> torch.Tensor:
        """
        Compute loss for a single head.
        
        Args:
            logits_or_value: Predictions from DeltaHead [N_pairs, out_dim]
                            - For classification: logits [N_pairs, num_classes]
                            - For regression: values [N_pairs, 1] or [N_pairs]
            targets: Ground truth targets [N_pairs]
                    - For classification: class indices (0 to num_classes-1)
                    - For regression: score differences (score_i - score_j)
        
        Returns:
            Scalar loss value
        """
        if self.task_type == "classification":
            # Ensure logits are 2D [N_pairs, num_classes]
            if logits_or_value.dim() == 1:
                logits_or_value = logits_or_value.unsqueeze(1)
            if logits_or_value.shape[1] != self.num_classes:
                raise ValueError(
                    f"Expected {self.num_classes} classes for classification, got {logits_or_value.shape[1]}"
                )
            # Targets should be class indices
            targets = targets.long()
        else:  # regression
            # Squeeze if needed to get [N_pairs]
            if logits_or_value.dim() > 1:
                logits_or_value = logits_or_value.squeeze(-1)
            # Targets should be continuous values
            targets = targets.float()
        
        return self.loss_fn(logits_or_value, targets)
    