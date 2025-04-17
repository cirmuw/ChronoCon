import torch
import torch.nn as nn
import torchvision

import os 
# import cseg_utils
# import cseg_utils.utils
# from cseg_utils.project_specific.hvpg.network.detached_bottleneck_regressor import make_mlp
# from cseg_utils.project_specific.hvpg.slices.model.interface import (
#     # EncoderDecoderNetwork, 
#     # EncoderDecoderRegressionNetwork,
#     EncoderDecoderRegressionNetwork,
#     EncoderRegressionNetwork
# )

import datetime
# from cseg_utils.project_specific.hvpg.network.autoencoder.resnet_18_autoencoder.classes.resnet_using_basic_block_encoder import Encoder, BasicBlockEnc
# from cseg_utils.project_specific.hvpg.network.autoencoder.resnet_18_autoencoder.classes.resnet_using_basic_block_decoder import Decoder, BasicBlockDec
from typing import Literal

# def get_resnet_hvpg_model(config: dict):
#     # Download weights to model_weights_dir
#     os.makedirs(config["model_weights_dir"], exist_ok=True)
#     os.environ['TORCH_HOME'] = config["model_weights_dir"]

#     model_params = cseg_utils.utils.model_parameter_imports_(config["model_params"], 
#                             model_dct_keys_to_convert_to_lists=config["model_dct_keys_to_convert_to_lists"],
#                             model_kw_requires_import=config["model_kw_requires_import"])


#     preprocessor = PreprocessingResNetRGBMaker()

#     encoder = ResNet18Encoder(weights='ResNet18_Weights.DEFAULT')

#     latent_dim = encoder.out_features
#     p = {**{"latent_dim": latent_dim}, **model_params}
#     mlp = make_mlp(**p)

#     net = EncoderRegressionNetwork(encoder=encoder, 
#                                 regressor=mlp, 
#                                 preprocssor=preprocessor,
#                                 return_latent_representation=False)
#     return net




# def get_AE_ResNet18_model(config: dict):
#     cfg = config["model"].get("preprocessing", {})
#     preprocessor = PreprocessingResNetRGBMaker(**cfg)
#     cfg = config["model"].get("postprocessing", {})
#     postprocessor = PostprocessingGrayScaleMaker(**cfg)
#     cfg = config["model"].get("encoder", {"layers": [2,2,2,2]})
#     encoder = Encoder(BasicBlockEnc, **cfg) 
#     cfg = config["model"].get("decoder", {"layers": [2,2,2,2]})
#     decoder = Decoder(BasicBlockDec, **cfg) 


#     # cae = AE('default')
#     # encoder = cae.encoder
#     # decoder = cae.decoder
#     # encoder = ResNet18EncoderWithoutPooling(weights='ResNet18_Weights.DEFAULT')

#     cfg = config["model"]["regressor"]
#     model_params = cseg_utils.utils.model_parameter_imports_(cfg["model_params"], 
#                             model_dct_keys_to_convert_to_lists=cfg["model_dct_keys_to_convert_to_lists"],
#                             model_kw_requires_import=cfg["model_kw_requires_import"])



#     # get latentspace dim
#     last_block = encoder.layer4[-1]  # Get the last block in layer4
#     if isinstance(last_block, nn.Sequential):  # If it's wrapped in a Sequential
#         last_conv = last_block[-1].conv2  # Get the last conv layer
#     else:
#         last_conv = last_block.conv2  # Access the last conv layer directly
#     latent_dim = last_conv.out_channels
#     # latent_dim = encoder.out_features
#     p = {**{"latent_dim": latent_dim}, **model_params}
#     mlp = make_mlp(**p)

#     reducer = nn.Sequential(*[nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1)])

#     net = EncoderDecoderRegressionNetwork(encoder=encoder,
#                                 decoder=decoder,  
#                                 preprocessor=preprocessor,
#                                 postprocessor=postprocessor,
#                                 return_latent_representation=True, 
#                                 reducer=reducer, 
#                                 regressor=mlp)
#     return net





class PreprocessingResNetRGBMaker(nn.Module):
    """
    input  dim (Nb, Nc=1, ...)
    output dim (Nb, Nc=3, ...)
    can also normalize channels (which pretrained resnet expects)
    """
        
    def __init__(self, 
                 mean = (0.485, 0.456, 0.406),
                 std = (0.229, 0.224, 0.225)
                 ):
        super().__init__()
        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)
                    
    def forward(self, x):
        # Repeat the single channel to create 3 channels
        x = x.repeat(1, 3, 1, 1)
        # Normalize the channels
        x = self.normalize(x)
        return x

class PostprocessingGrayScaleMaker(nn.Module):
    """
    input  dim (N, 3, H, W)
    output dim (N, 1, H, W)
    """
    def __init__(self, 
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), 
                 output_function : Literal["None", "relu", "sigmoid"] = "None"
                 ):
        super().__init__()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        
        self.output_function_str = output_function
        if output_function == "None":
            self.output_function = None
        elif output_function == "relu":
            self.output_function = nn.ReLU()
        elif output_function == "sigmoid":
            self.output_function = nn.Sigmoid()
        else: 
            raise ValueError(f"{output_function=}")
                
    def forward(self, x):
        # Unnormalize the channels
        x = x * self.std.to(x.device) + self.mean.to(x.device)
        # Average over the channels
        x = torch.mean(x, dim=1, keepdim=True)
        if self.output_function != None:
            x = self.output_function(x)
        return x

# class PostprocessingGrayScaleMaker(nn.Module):
#     """
#     input  dim (N, 3, H, W)
#     output dim (N, 1, H, W)
#     """
#     def __init__(self, 
#                  mean=(0.485, 0.456, 0.406),
#                  std=(0.229, 0.224, 0.225)
#                  ):
#         super().__init__()
#         self.mean = torch.tensor(mean).view(1, 3, 1, 1)
#         self.std = torch.tensor(std).view(1, 3, 1, 1)
            
#     def forward(self, x):
#         # Unnormalize the channels
#         x = x * self.std.to(x.device) + self.mean.to(x.device)
#         # Average over the channels
#         x = torch.mean(x, dim=1, keepdim=True)
#         return x


# class PostprocessingGrayScaleMakerNoAverage(nn.Module):
#     """
#     input  dim (N, Nc, H, W)
#     output dim (N, 0:out_channels, H, W)
#     """
#     def __init__(self, out_channels=1):
#         super().__init__()
#         self.out_channels = out_channels
            
#     def forward(self, x):
#         return x[..., 0 : self.out_channels, :, :]

class PostprocessingGrayScaleMakerNoAverage(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x):
        # Slice and then clone to avoid returning a view
        x_sliced = x[..., :self.out_channels, :, :]
        return x_sliced.clone()



class TrivialReturnInput(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x


class ResNet18Encoder(nn.Module):
    def __init__(self, **resenet_kwargs):
        """
        encoder = ResNet18Encoder(weights='ResNet18_Weights.DEFAULT')
        """
        
        super().__init__()
        self.net = torchvision.models.resnet18(**resenet_kwargs)
        self.out_features = self.net.fc.in_features  # in features of last layer 
        self.net.fc = TrivialReturnInput() # remove last layer of 
        
    def forward(self, x):
        z = self.net(x)
        return z
    
# class ResNet18EncoderWithoutPooling(nn.Module):
#     def __init__(self, **resenet_kwargs):
#         """
#         encoder = ResNet18Encoder(weights='ResNet18_Weights.DEFAULT')
#         """
        
#         super().__init__()
#         self.net = torchvision.models.resnet18(**resenet_kwargs) 
#         self.net.avgpool = TrivialReturnInput() # remove average pooling
#         self.net.fc = TrivialReturnInput() # remove last layer of 
        
#     def forward(self, x):
#         z = self.net(x)
#         return z
    
    
