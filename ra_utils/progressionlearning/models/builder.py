from monai.networks.nets import BasicUNet

import ra_utils.progressionlearning.models as models

from ra_utils.progressionlearning.models.MTANUNet import MTANRecUnet, MTANRecUnet_v2, MTANRecUnet_v3
from ra_utils.progressionlearning.models.RecUNet import RecUnet
from ra_utils.progressionlearning.models.Classifier import AttClassifier
from ra_utils.progressionlearning.models.TESSLCNN import MRIEncoder2D
import torch
from typing import Dict, List



def build_MTANAE(
        filters:list = [16,32,64,128,256,32], 
        disentangle: bool = False ,
        device: str ='cuda', 
        in_channels: int = 3,
        out_channels: int = 3): 
    unet = BasicUNet(
        spatial_dims=2, 
        in_channels=in_channels, 
        out_channels=out_channels, 
        features=filters).to(device)
    
    model = MTANRecUnet(
        unet=unet, 
        filters = filters[1:5], 
        disentangle=disentangle).to(device)
    
    return model


def build_MTANAE_v1p1(
        filters: list = [16, 32, 64, 128, 256, 32],
        device: str = 'cuda',
        u_net_arch="BasicUNet",
        u_net_kwargs=dict(
            spacial_dims=2,
            in_channels=3,
            out_channels=3,
        )):
    if u_net_arch == "BasicUNet":
        unet = BasicUNet(
            features=filters,
            **u_net_kwargs).to(device)
    elif u_net_arch == "UNet":
        raise NotImplementedError
    else: 
        raise NotImplementedError

    model = MTANRecUnet_v2(
        unet=unet,
        filters=filters[1:5]).to(device)

    return model


def build_MTANAE_v2(
        attention_paths: Dict[str, List[str]],
        filters: list = [16, 32, 64, 128, 256, 32],
        device: str = 'cuda',
        u_net_arch="BasicUNet",
        u_net_kwargs=dict(
            spacial_dims=2,
            in_channels=3,
            out_channels=3,
        )):
    if u_net_arch == "BasicUNet":
        unet = BasicUNet(
            features=filters,
            **u_net_kwargs).to(device)
    elif u_net_arch == "UNet":
        raise NotImplementedError
    else: 
        raise NotImplementedError

    model = MTANRecUnet_v3(
        unet=unet,
        filters=filters[1:5],
        attention_paths=attention_paths).to(device)

    return model





def build_RecUNet(
        filters:list = [16,32,64,128,256,32], 
        disentangle: bool = False ,
        device: str ='cuda'): 
    unet = BasicUNet(
        spatial_dims=2, 
        in_channels=3, 
        out_channels=3, 
        features=filters).to(device)
    
    model = RecUnet(
        unet=unet, 
        filters = filters[1:5], 
        disentangle=disentangle).to(device)

    return model

def build_classifier(
        pretrained_model_path: str, 
        num_heads: int, 
        filters: list = [16,32,64,128,256,32], 
        disentangle=True, 
        freeze: bool = True,
        device='cuda'):
    
    pretrained_model = build_MTANAE(filters=filters, disentangle=disentangle, device=device).to(device)
    pretrained_model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    classifier = AttClassifier(
        pretrained_model, 
        num_heads=num_heads, 
        latent_dim=480,
        freeze=freeze).to(device)
    return classifier



def build_TESSLCNN(
    mri_out=128, dropout=0.5, 
    expansion=8, norm_type='Instance', 
    activation='relu', device='cuda'
    ):
     # Create image embedding model
    image_embeding_model = MRIEncoder2D(in_channel=3, feat_dim=mri_out, expansion=expansion, norm_type=norm_type, activation=activation, dropout=dropout).to(device)

    return image_embeding_model
