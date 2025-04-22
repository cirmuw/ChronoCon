import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm


def train_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion_tessl: nn.Module,
        temporal_distances_mapping: dict = {'T0': 1.0, 'T1': 0.75, 'T2': 0.5, 'T3': 0.25},
        timepoints: list = ['T0', 'T1', 'T2', 'T3'],
        device: torch.device = 'cuda',
    ):

    loss_total_epoch = []
    acc_steps = 8
    model.train()
    for i, batch_data in enumerate(tqdm(loader)):
        bsz = batch_data['T0'][0].shape[0]
        data_view1 = torch.cat([batch_data[t][0] for t in timepoints], dim=0).to(device) # concat data from 4 timepoints
        data_view2 = torch.cat([batch_data[t][1] for t in timepoints], dim=0).to(device)  # concat data from 4 timepoints
        events = torch.cat([batch_data['pcr'] for i in range(4)], dim=0).to(device)  # concat response information from 4 timepoints
        times = torch.tensor([temporal_distances_mapping[t]  for t in timepoints for i in range(bsz)]).to(device)  # temporal label based on the temporal mapping
        # Forward pass
        out_1 = model(data_view1.float())
        out_2 = model(data_view2.float())
        features = torch.cat([nn.functional.normalize(out_1.unsqueeze(1)), nn.functional.normalize(out_2.unsqueeze(1))], 1)
        loss_tessl = criterion_tessl(features, labels=events, times=times)
   
        loss =  loss_tessl 

        loss_total_epoch.append(loss.item())
        loss = loss / acc_steps
        loss.backward()  # Accumulate gradients
        if i % acc_steps == 0 or i == len(loader) - 1:  # Backpropagate every 8 steps
            optimizer.step()
            optimizer.zero_grad()
    
    return {
    'total' : sum(loss_total_epoch)/len(loss_total_epoch),
    }
    
def eval_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion_tessl: nn.Module,
        temporal_distances_mapping: dict = {'T0': 1.0, 'T1': 0.75, 'T2': 0.5, 'T3': 0.25},
        timepoints: list = ['T0', 'T1', 'T2', 'T3'],
        device: torch.device = 'cuda',
    ):
    loss_tessl_epoch = []
    model.eval()
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(loader)):
            bsz = batch_data['T0'].shape[0]
            data_view1 = torch.cat([batch_data[t] for t in timepoints], dim=0).to(device) # concat data from 4 timepoints
            data_view2 = torch.cat([batch_data[t] for t in timepoints], dim=0).to(device)  # concat data from 4 timepoints
            events = torch.cat([batch_data['pcr'] for i in range(4)], dim=0).to(device)  # concat response information from 4 timepoints
            times = torch.tensor([temporal_distances_mapping[t]  for t in timepoints for i in range(bsz)]).to(device)  # temporal label based on the temporal mapping
            # Forward pass
            out_1 = model(data_view1.float())
            out_2 = model(data_view2.float())
            features = torch.cat([nn.functional.normalize(out_1.unsqueeze(1)), nn.functional.normalize(out_2.unsqueeze(1))], 1)
            loss_tessl = criterion_tessl(features, labels=events, times=times)
        
            loss_tessl_epoch.append(loss_tessl.item())
        
    return {
    'tessl' : sum(loss_tessl_epoch)/len(loss_tessl_epoch)
    }