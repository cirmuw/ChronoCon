import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.losses import compute_cross_cov_loss


def train_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        temporal_distances_mapping: dict = {'T0': 1.0, 'T1': 0.75, 'T2': 0.5, 'T3': 0.25},
        timepoints: list = ['T0', 'T1', 'T2', 'T3'],
        device: torch.device = 'cuda',
        mode: str = 'all',
):
    model.train()
    loss_temp_epoch = []
    loss_sup_epoch = []
    loss_rec_epoch = []
    loss_cov_epoch = []
    loss_total_epoch = []
    for batch_data in tqdm(loader):
        for i in range(len(timepoints)):
            for j in range(len(timepoints)):
                if i != j: 
                    t1 = timepoints[i]
                    t2 = timepoints[i]
                    t3 = timepoints[j]
                    margin = np.abs(temporal_distances_mapping[t1] - temporal_distances_mapping[t3])
                    img_1 = batch_data[t1][0].float().to(device) # anchor - timepoint i aug 0
                    img_2 = batch_data[t2][1].float().to(device) # positive - timepoint i aug 1
                    img_3 = batch_data[t3][0].float().to(device) # negative - timepoint j

                    label = batch_data['pcr'].float().to(device)
                    rec1, latent_1, maps1 = model(img_1)
                    latent_1 = nn.functional.normalize(latent_1)
                    _, latent_2, maps2 = model(img_2)
                    latent_2 = nn.functional.normalize(latent_2)
                    rec3, latent_3, maps3 = model(img_3)
                    latent_3 = nn.functional.normalize(latent_3)
                    a = latent_1
                    p = latent_2
                    n = latent_3
                    
                    # temporal loss
                    loss_temp = nn.functional.triplet_margin_with_distance_loss(
                        anchor=a,
                        positive=p,
                        negative=n,
                        margin=margin,
                        distance_function=lambda x, y: 1.0 - nn.functional.cosine_similarity(x, y),
                        reduction='mean'
                    )
                
                    p_pcr_1 = p[label==1.0]
                    a_pcr_1 = a[label==1.0]
                    
                    # population level for pcr == 1 patients
                    perm_a_pcr_1 = torch.randperm(a_pcr_1.size(0))
                    a_pcr_1 = a_pcr_1[perm_a_pcr_1]

                    loss_sup =  1.0 - nn.functional.cosine_similarity(a_pcr_1, p_pcr_1, dim=-1).mean()
                    loss_sup_epoch.append(loss_sup.item())

                    rec_img = torch.cat([rec1, rec3], dim=0)
                    target_img = torch.cat([batch_data[f'target_{t1}'], batch_data[f'target_{t3}']], dim=0).to(device)
                    loss_rec = nn.functional.mse_loss(rec_img, target_img, reduction='mean')

                    if mode == 'sup':
                        loss = loss_rec + loss_sup
                    if mode == 'temp':
                        loss = loss_rec + loss_temp
                    if mode == 'sup_temp':
                        loss = loss_rec + loss_temp + loss_sup

                    loss_temp_epoch.append(loss_temp.item())
                    loss_rec_epoch.append(loss_rec.item())
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_total_epoch.append(loss.item())
    return {
        'total' : sum(loss_total_epoch)/len(loss_total_epoch),
        'temp' : sum(loss_temp_epoch)/len(loss_temp_epoch),
        'sup' : sum(loss_sup_epoch)/len(loss_sup_epoch),
        'rec' : sum(loss_rec_epoch)/len(loss_rec_epoch),
    }




def eval_epoch(
        model: nn.Module,
        loader: DataLoader,
        temporal_distances_mapping: dict = {'T0': 1.0, 'T1': 0.75, 'T2': 0.5, 'T3': 0.25},
        timepoints: list = ['T0', 'T1', 'T2', 'T3'],
        device: torch.device = 'cuda',
):
    model.eval()
    loss_temp_epoch = []
    loss_sup_epoch = []
    loss_rec_epoch = []
    with torch.no_grad():
        for batch_data in tqdm(loader):
            for i in range(len(timepoints)):
                for j in range(len(timepoints)):
                    if i != j: 
                        t1 = timepoints[i]
                        t2 = timepoints[i]
                        t3 = timepoints[j]
                        margin = np.abs(temporal_distances_mapping[t1] - temporal_distances_mapping[t3])
                        img_1 = batch_data[t1].float().to(device) # anchor 
                        img_2 = batch_data[t2].float().to(device) # positive 
                        img_3 = batch_data[t3].float().to(device) # negative 

                        label = batch_data['pcr'].float().to(device)
                        rec1, latent_1, _ = model(img_1)
                        latent_1 = nn.functional.normalize(latent_1)
                        _, latent_2, _ = model(img_2)
                        latent_2 = nn.functional.normalize(latent_2)
                        rec3, latent_3, _ = model(img_3)
                        latent_3 = nn.functional.normalize(latent_3)
                        a = latent_1
                        p = latent_2
                        n = latent_3
                        
                        # temporal loss
                        loss_temp = nn.functional.triplet_margin_with_distance_loss(
                            anchor=a,
                            positive=p,
                            negative=n,
                            margin=margin,
                            distance_function=lambda x, y: 1.0 - nn.functional.cosine_similarity(x, y),
                            reduction='mean'
                        )
                    
                        p_pcr_1 = p[label==1.0]
                        a_pcr_1 = a[label==1.0]
                        
                        # population level for pcr == 1 patients
                        perm_a_pcr_1 = torch.randperm(a_pcr_1.size(0))
                        a_pcr_1 = a_pcr_1[perm_a_pcr_1]

                        loss_sup =  1.0 - nn.functional.cosine_similarity(a_pcr_1, p_pcr_1, dim=-1).mean()
                        loss_sup_epoch.append(loss_sup.item())

                        rec_img = torch.cat([rec1, rec3], dim=0)
                        target_img = torch.cat([batch_data[f'target_{t1}'], batch_data[f'target_{t3}']], dim=0).to(device)
                        loss_rec = nn.functional.mse_loss(rec_img, target_img, reduction='mean')

                        loss_temp_epoch.append(loss_temp.item())
                        loss_rec_epoch.append(loss_rec.item())

    return {
        'temp' : sum(loss_temp_epoch)/len(loss_temp_epoch),
        'sup' : sum(loss_sup_epoch)/len(loss_sup_epoch),
        'rec' : sum(loss_rec_epoch)/len(loss_rec_epoch),
    }