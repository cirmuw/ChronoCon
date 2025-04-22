import torch
import torch.nn as nn

def get_feature_dict(dl, model, device, timepoints=['T0', 'T1', 'T2', 'T3'], is_eval=False):

    latent_dict_temp = {
        'T0': [],
        'T1': [],
        'T2': [],
        'T3': [],
        'pcr' : [], 
    }
    model.eval()
    with torch.no_grad():
        for batch_data in dl:
            for t in timepoints:
                if is_eval:
                    x = batch_data[t].float().to(device)
                else:
                    x = batch_data[t][0].float().to(device)
                lab = batch_data['pcr'].to(device)
                _, z, _ = model(x)
                z = nn.functional.normalize(z)
                latent_dict_temp[t].append(z.detach().cpu())
            latent_dict_temp['pcr'].append(lab.detach().cpu())

 
    for k in latent_dict_temp.keys():
            v = latent_dict_temp.get(k)
            latent_dict_temp[k] = torch.cat(v, dim=0)
    return latent_dict_temp