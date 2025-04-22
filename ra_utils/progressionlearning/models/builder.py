from monai.networks.nets import BasicUNet
from models.MTANUNet import MTANRecUnet
from models.RecUNet import RecUnet
from models.Classifier import AttClassifier
from models.TESSLCNN import MRIEncoder2D
import torch

def build_MTANAE(
        filters:list = [16,32,64,128,256,32], 
        disentangle: bool = False ,
        device: str ='cuda'): 
    unet = BasicUNet(
        spatial_dims=2, 
        in_channels=3, 
        out_channels=3, 
        features=filters).to(device)
    
    model = MTANRecUnet(
        unet=unet, 
        filters = filters[1:5], 
        disentangle=disentangle).to(device)
    
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
