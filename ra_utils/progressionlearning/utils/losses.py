import torch
import torch.nn as nn


def cross_covariance_loss(F1, F2):
    B, C, H, W = F1.shape
    
    # Flatten batch and spatial dimensions into one: [B*H*W, C]
    F1_flat = F1.permute(1, 0, 2, 3).reshape(C, -1).T  # [B*H*W, C]
    F2_flat = F2.permute(1, 0, 2, 3).reshape(C, -1).T
    
    # Centered data
    F1_centered = F1_flat - F1_flat.mean(dim=0, keepdim=True)
    F2_centered = F2_flat - F2_flat.mean(dim=0, keepdim=True)
    
    # Cross-covariance [C, C]
    cov = (F1_centered.T @ F2_centered) / (B * H * W - 1)
    loss = torch.sum(cov**2) / (C**2)  # Normalize
    
    return loss

def compute_cross_cov_loss(all_maps):
    num_instances = len(all_maps)
    num_maps = len(all_maps[0])
    loss = 0.0
    iter = 0
    for i in range(num_maps):
        # get same-scale maps across instances 
        maps_positive = []
        maps_negative = []
        for j in range(num_instances):
            maps_positive.append(all_maps[j][i][0])
            maps_negative.append(all_maps[j][i][1])
        # combine into one tensor   
        maps_positive = torch.cat(maps_positive, dim=0)
        maps_negative = torch.cat(maps_negative, dim=0)
        # compute the cross covariance
        l = cross_covariance_loss(maps_negative, maps_negative)
        loss += l
        iter += 1

    return loss/iter
