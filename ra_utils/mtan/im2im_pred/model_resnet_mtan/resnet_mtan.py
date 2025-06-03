import torch
import torch.nn as nn
import torch.nn.functional as F
import ra_utils.mtan.im2im_pred.model_resnet_mtan.resnet as resnet
import ra_utils.mtan.im2im_pred.model_resnet_mtan.resnetT as resnetT


from ra_utils.mtan.im2im_pred.model_resnet_mtan.resnet_dilated import ResnetDilated
from ra_utils.mtan.im2im_pred.model_resnet_mtan.aspp import DeepLabHead
from ra_utils.mtan.im2im_pred.model_resnet_mtan.resnet import Bottleneck, conv1x1
from ra_utils.mtan.im2im_pred.model_resnet_mtan.resnetT import BottleneckT


resnet_infos = {
    "resnet18": {
        "ch": [64, 128, 256, 512],
        "decoder_name": "resnet18_decoder",
        "encoder_name": "resnet18"}, 
    "resnet34": {
        "ch": [64, 128, 256, 512],
        "decoder_name": "resnet34_decoder",
        "encoder_name": "resnet34"},
    "resnet50": {
        "ch": [256, 512, 1024, 2048],
        "decoder_name": "resnet50_decoder",
        "encoder_name": "resnet50"},
    "resnet101": {
        "ch": [256, 512, 1024, 2048],
        "decoder_name": "resnet101_decoder",
        "encoder_name": "resnet101"},
    "resnet152": {
        "ch": [256, 512, 1024, 2048],
        "decoder_name": "resnet152_decoder",
        "encoder_name": "resnet152"},
}



class MTANDeepLabv3(nn.Module):
    def __init__(self):
        super(MTANDeepLabv3, self).__init__()
        backbone = ResnetDilated(resnet.__dict__['resnet50'](pretrained=True))
        ch = [256, 512, 1024, 2048]
        
        self.tasks = ['segmentation', 'depth', 'normal']
        self.num_out_channels = {'segmentation': 13, 'depth': 1, 'normal': 3}
        
        self.shared_conv = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu1, backbone.maxpool)

        # We will apply the attention over the last bottleneck layer in the ResNet. 
        self.shared_layer1_b = backbone.layer1[:-1] 
        self.shared_layer1_t = backbone.layer1[-1]

        self.shared_layer2_b = backbone.layer2[:-1]
        self.shared_layer2_t = backbone.layer2[-1]

        self.shared_layer3_b = backbone.layer3[:-1]
        self.shared_layer3_t = backbone.layer3[-1]

        self.shared_layer4_b = backbone.layer4[:-1]
        self.shared_layer4_t = backbone.layer4[-1]

        # Define task specific attention modules using a similar bottleneck design in residual block
        # (to avoid large computations)
        self.encoder_att_1 = nn.ModuleList([self.att_layer(ch[0], ch[0] // 4, ch[0]) for _ in self.tasks])
        self.encoder_att_2 = nn.ModuleList([self.att_layer(2 * ch[1], ch[1] // 4, ch[1]) for _ in self.tasks])
        self.encoder_att_3 = nn.ModuleList([self.att_layer(2 * ch[2], ch[2] // 4, ch[2]) for _ in self.tasks])
        self.encoder_att_4 = nn.ModuleList([self.att_layer(2 * ch[3], ch[3] // 4, ch[3]) for _ in self.tasks])

        # Define task shared attention encoders using residual bottleneck layers
        # We do not apply shared attention encoders at the last layer,
        # so the attended features will be directly fed into the task-specific decoders.
        self.encoder_block_att_1 = self.conv_layer(ch[0], ch[1] // 4)
        self.encoder_block_att_2 = self.conv_layer(ch[1], ch[2] // 4)
        self.encoder_block_att_3 = self.conv_layer(ch[2], ch[3] // 4)
        
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define task-specific decoders using ASPP modules
        self.decoders = nn.ModuleList([DeepLabHead(2048, self.num_out_channels[t]) for t in self.tasks])
        
    def forward(self, x, out_size):
        # Shared convolution
        x = self.shared_conv(x)
        
        # Shared ResNet block 1
        u_1_b = self.shared_layer1_b(x)
        u_1_t = self.shared_layer1_t(u_1_b)

        # Shared ResNet block 2
        u_2_b = self.shared_layer2_b(u_1_t)
        u_2_t = self.shared_layer2_t(u_2_b)

        # Shared ResNet block 3
        u_3_b = self.shared_layer3_b(u_2_t)
        u_3_t = self.shared_layer3_t(u_3_b)
        
        # Shared ResNet block 4
        u_4_b = self.shared_layer4_b(u_3_t)
        u_4_t = self.shared_layer4_t(u_4_b)

        # Attention block 1 -> Apply attention over last residual block
        a_1_mask = [att_i(u_1_b) for att_i in self.encoder_att_1]  # Generate task specific attention map
        a_1 = [a_1_mask_i * u_1_t for a_1_mask_i in a_1_mask]  # Apply task specific attention map to shared features
        a_1 = [self.down_sampling(self.encoder_block_att_1(a_1_i)) for a_1_i in a_1]
        
        # Attention block 2 -> Apply attention over last residual block
        a_2_mask = [att_i(torch.cat((u_2_b, a_1_i), dim=1)) for a_1_i, att_i in zip(a_1, self.encoder_att_2)]
        a_2 = [a_2_mask_i * u_2_t for a_2_mask_i in a_2_mask]
        a_2 = [self.encoder_block_att_2(a_2_i) for a_2_i in a_2]
        
        # Attention block 3 -> Apply attention over last residual block
        a_3_mask = [att_i(torch.cat((u_3_b, a_2_i), dim=1)) for a_2_i, att_i in zip(a_2, self.encoder_att_3)]
        a_3 = [a_3_mask_i * u_3_t for a_3_mask_i in a_3_mask]
        a_3 = [self.encoder_block_att_3(a_3_i) for a_3_i in a_3]
        
        # Attention block 4 -> Apply attention over last residual block (without final encoder)
        a_4_mask = [att_i(torch.cat((u_4_b, a_3_i), dim=1)) for a_3_i, att_i in zip(a_3, self.encoder_att_4)]
        a_4 = [a_4_mask_i * u_4_t for a_4_mask_i in a_4_mask]
        
        # Task specific decoders
        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](a_4[i]), size=out_size, mode='bilinear', align_corners=True)
            if t == 'segmentation':
                out[i] = F.log_softmax(out[i], dim=1)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        return out
    
    def att_layer(self, in_channel, intermediate_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=intermediate_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(intermediate_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=intermediate_channel, out_channels=out_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid())
        
    def conv_layer(self, in_channel, out_channel):
        downsample = nn.Sequential(conv1x1(in_channel, 4 * out_channel, stride=1),
                                   nn.BatchNorm2d(4 * out_channel))
        return Bottleneck(in_channel, out_channel, downsample=downsample)





# Should do exactly the same as MTANDeepLabv3 but with arbitrary ResNet as backbone

class MTANDeepLabv3p1(nn.Module):
    def __init__(self, backbone_name: str = "resnet18"):
        super(MTANDeepLabv3p1, self).__init__()
        encoder_name = resnet_infos[backbone_name]["encoder_name"]
        
        backbone = ResnetDilated(resnet.__dict__[encoder_name](pretrained=True))
        ch = resnet_infos[backbone_name]["ch"]
        
        self.tasks = ['segmentation', 'depth', 'normal']
        self.num_out_channels = {'segmentation': 13, 'depth': 1, 'normal': 3}
        
        self.shared_conv = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu1, backbone.maxpool)

        # We will apply the attention over the last bottleneck layer in the ResNet. 
        self.shared_layer1_b = backbone.layer1[:-1] 
        self.shared_layer1_t = backbone.layer1[-1]

        self.shared_layer2_b = backbone.layer2[:-1]
        self.shared_layer2_t = backbone.layer2[-1]

        self.shared_layer3_b = backbone.layer3[:-1]
        self.shared_layer3_t = backbone.layer3[-1]

        self.shared_layer4_b = backbone.layer4[:-1]
        self.shared_layer4_t = backbone.layer4[-1]

        # Define task specific attention modules using a similar bottleneck design in residual block
        # (to avoid large computations)
        self.encoder_att_1 = nn.ModuleList([self.att_layer(ch[0], ch[0] // 4, ch[0]) for _ in self.tasks])
        self.encoder_att_2 = nn.ModuleList([self.att_layer(2 * ch[1], ch[1] // 4, ch[1]) for _ in self.tasks])
        self.encoder_att_3 = nn.ModuleList([self.att_layer(2 * ch[2], ch[2] // 4, ch[2]) for _ in self.tasks])
        self.encoder_att_4 = nn.ModuleList([self.att_layer(2 * ch[3], ch[3] // 4, ch[3]) for _ in self.tasks])

        # Define task shared attention encoders using residual bottleneck layers
        # We do not apply shared attention encoders at the last layer,
        # so the attended features will be directly fed into the task-specific decoders.
        self.encoder_block_att_1 = self.conv_layer(ch[0], ch[1] // 4)
        self.encoder_block_att_2 = self.conv_layer(ch[1], ch[2] // 4)
        self.encoder_block_att_3 = self.conv_layer(ch[2], ch[3] // 4)
        
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define task-specific decoders using ASPP modules
        self.decoders = nn.ModuleList([DeepLabHead(ch[-1], self.num_out_channels[t]) for t in self.tasks])
        
    def forward(self, x, out_size):
        # Shared convolution
        x = self.shared_conv(x)
        
        # Shared ResNet block 1
        u_1_b = self.shared_layer1_b(x)
        u_1_t = self.shared_layer1_t(u_1_b)

        # Shared ResNet block 2
        u_2_b = self.shared_layer2_b(u_1_t)
        u_2_t = self.shared_layer2_t(u_2_b)

        # Shared ResNet block 3
        u_3_b = self.shared_layer3_b(u_2_t)
        u_3_t = self.shared_layer3_t(u_3_b)
        
        # Shared ResNet block 4
        u_4_b = self.shared_layer4_b(u_3_t)
        u_4_t = self.shared_layer4_t(u_4_b)

        # Attention block 1 -> Apply attention over last residual block
        a_1_mask = [att_i(u_1_b) for att_i in self.encoder_att_1]  # Generate task specific attention map
        a_1 = [a_1_mask_i * u_1_t for a_1_mask_i in a_1_mask]  # Apply task specific attention map to shared features
        a_1 = [self.down_sampling(self.encoder_block_att_1(a_1_i)) for a_1_i in a_1]
        
        # Attention block 2 -> Apply attention over last residual block
        a_2_mask = [att_i(torch.cat((u_2_b, a_1_i), dim=1)) for a_1_i, att_i in zip(a_1, self.encoder_att_2)]
        a_2 = [a_2_mask_i * u_2_t for a_2_mask_i in a_2_mask]
        a_2 = [self.encoder_block_att_2(a_2_i) for a_2_i in a_2]
        
        # Attention block 3 -> Apply attention over last residual block
        a_3_mask = [att_i(torch.cat((u_3_b, a_2_i), dim=1)) for a_2_i, att_i in zip(a_2, self.encoder_att_3)]
        a_3 = [a_3_mask_i * u_3_t for a_3_mask_i in a_3_mask]
        a_3 = [self.encoder_block_att_3(a_3_i) for a_3_i in a_3]
        
        # Attention block 4 -> Apply attention over last residual block (without final encoder)
        a_4_mask = [att_i(torch.cat((u_4_b, a_3_i), dim=1)) for a_3_i, att_i in zip(a_3, self.encoder_att_4)]
        a_4 = [a_4_mask_i * u_4_t for a_4_mask_i in a_4_mask]
        
        #return a_4
        # Task specific decoders
        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](a_4[i]), size=out_size, mode='bilinear', align_corners=True)
            if t == 'segmentation':
                out[i] = F.log_softmax(out[i], dim=1)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        return out
    
    def att_layer(self, in_channel, intermediate_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=intermediate_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(intermediate_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=intermediate_channel, out_channels=out_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid())
        
    def conv_layer(self, in_channel, out_channel):
        downsample = nn.Sequential(conv1x1(in_channel, 4 * out_channel, stride=1),
                                   nn.BatchNorm2d(4 * out_channel))
        return Bottleneck(in_channel, out_channel, downsample=downsample)
    

# ------------------------------------------------------------------ #
#                       Reconstruction model                         #
# ------------------------------------------------------------------ #
from ra_utils.mtan.im2im_pred.model_resnet_mtan.resnetT  import resnet18_decoder, resnet34_decoder, resnet50_decoder
# from ra_utils.mtan.im2im_pred.model_resnet_mtan.resnet_using_basic_block_decoder import (
#     resnet18_decoder_v2
# )

decoder_by_name = dict(resnet18=resnet18_decoder,
                       #resnet18 = resnet18_decoder_v2, 
                       resnet34=resnet34_decoder,
                       resnet50=resnet50_decoder)


from ra_utils.progressionlearning.models.MTANUNet import make_score_type_2_attention_paths_dct
from typing import List, Dict, Tuple


class MTANReconv1(nn.Module):
    def __init__(self, 
                 attention_paths: Dict[str, List[str]], 
                 backbone_name: str = "resnet18", 
                 dilation=True,
                 skip_recon=True,
                 recon_out_ch = 3):
        super(MTANReconv1, self).__init__()
        encoder_name = resnet_infos[backbone_name]["encoder_name"]
        self.skip_recon = skip_recon
        self.dilation = dilation
        if dilation: 
            backbone = ResnetDilated(resnet.__dict__[encoder_name](pretrained=True))
            self.shared_conv = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu1, backbone.maxpool)
        else: 
            backbone = resnet.__dict__[encoder_name](pretrained=True)
            self.shared_conv = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)   
        ch = resnet_infos[backbone_name]["ch"]
        self.latent_dim = ch[-1]
        if self.skip_recon:
            self.encoder_decoder_tasks = []
            self.num_out_channels_decoder = {}
        else: 
            self.encoder_decoder_tasks = ["recon"]
            self.num_out_channels_decoder = {"recon": recon_out_ch}
        self.score_types_to_attention_paths = make_score_type_2_attention_paths_dct(attention_paths)
        self.attention_paths = attention_paths.copy()
        self.encoder_only_tasks = list(self.attention_paths.keys())
        self.tasks = self.encoder_decoder_tasks + self.encoder_only_tasks




        # We will apply the attention over the last bottleneck layer in the ResNet. 
        self.shared_layer1_b = backbone.layer1[:-1] 
        self.shared_layer1_t = backbone.layer1[-1]

        self.shared_layer2_b = backbone.layer2[:-1]
        self.shared_layer2_t = backbone.layer2[-1]

        self.shared_layer3_b = backbone.layer3[:-1]
        self.shared_layer3_t = backbone.layer3[-1]

        self.shared_layer4_b = backbone.layer4[:-1]
        self.shared_layer4_t = backbone.layer4[-1]

        # Define task specific attention modules using a similar bottleneck design in residual block
        # (to avoid large computations)
        self.encoder_att_1 = nn.ModuleDict({k: self.att_layer(ch[0], ch[0] // 4, ch[0]) for k in self.tasks})
        self.encoder_att_2 = nn.ModuleDict({k: self.att_layer(2 * ch[1], ch[1] // 4, ch[1]) for k in self.tasks})
        self.encoder_att_3 = nn.ModuleDict({k: self.att_layer(2 * ch[2], ch[2] // 4, ch[2]) for k in self.tasks})
        self.encoder_att_4 = nn.ModuleDict({k: self.att_layer(2 * ch[3], ch[3] // 4, ch[3]) for k in self.tasks})

        # Define task shared attention encoders using residual bottleneck layers
        # We do not apply shared attention encoders at the last layer,
        # so the attended features will be directly fed into the task-specific decoders.
        self.encoder_block_att_1 = self.conv_layer(ch[0], ch[1] // 4)
        self.encoder_block_att_2 = self.conv_layer(ch[1], ch[2] // 4)
        self.encoder_block_att_3 = self.conv_layer(ch[2], ch[3] // 4)
        
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pooling_at_bottleneck = nn.AdaptiveAvgPool2d((1,1))


        # Define task-specific decoders using ASPP modules
        if not self.skip_recon: 
            # self.decoders = nn.ModuleDict({t: DeepLabHead(ch[-1], self.num_out_channels_decoder[t]) for t in self.encoder_decoder_tasks})
            self.decoders = nn.ModuleDict({t: decoder_by_name[backbone_name](out_ch=self.num_out_channels_decoder[t]) for t in self.encoder_decoder_tasks})


    def forward(self, x: torch.Tensor, score_type: List[str], out_size: Tuple[int] = (128,128)):
        
        # Only one encoder_only path is active (for now)!
        assert len(score_type) == x.shape[0], f"shapes of input dont match. Should both be batch dim"
        active_encoder_only_paths = [self.score_types_to_attention_paths[s] for s in score_type]
        assert len(set(active_encoder_only_paths)) == 1, f"More than one path is active for these types {(set(score_type))}  {set(active_encoder_only_paths)}"
        active_attention_task = active_encoder_only_paths[0]
        active_tasks = self.encoder_decoder_tasks + [active_attention_task]

        if self.skip_recon: 
            x_fake_recon = x
        else:
            assert len(self.encoder_decoder_tasks) == 1, "Assuming only one decoder task for now!"
            active_decoder_task = list(self.encoder_decoder_tasks)[0]

        # Shared convolution
        x = self.shared_conv(x)
        
        # Shared ResNet block 1
        u_1_b = self.shared_layer1_b(x)
        u_1_t = self.shared_layer1_t(u_1_b)

        # Shared ResNet block 2
        u_2_b = self.shared_layer2_b(u_1_t)
        u_2_t = self.shared_layer2_t(u_2_b)

        # Shared ResNet block 3
        u_3_b = self.shared_layer3_b(u_2_t)
        u_3_t = self.shared_layer3_t(u_3_b)
        
        # Shared ResNet block 4
        u_4_b = self.shared_layer4_b(u_3_t)
        u_4_t = self.shared_layer4_t(u_4_b)

        # Attention block 1 -> Apply attention over last residual block
        active_encoder_att_1 = [v for (task,v) in self.encoder_att_1.items() if task in active_tasks]
        a_1_mask = [att_i(u_1_b) for att_i in active_encoder_att_1]
        a_1 = [a_1_mask_i * u_1_t for a_1_mask_i in a_1_mask]
        a_1 = [self.down_sampling(self.encoder_block_att_1(a_1_i)) for a_1_i in a_1]
        
        # Attention block 2 -> Apply attention over last residual block
        active_encoder_att_2 = [v for (task,v) in self.encoder_att_2.items() if task in active_tasks]
        a_2_mask = [att_i(torch.cat((u_2_b, a_1_i), dim=1)) for a_1_i, att_i in zip(a_1, active_encoder_att_2)]
        a_2 = [a_2_mask_i * u_2_t for a_2_mask_i in a_2_mask]
        a_2 = [self.encoder_block_att_2(a_2_i) for a_2_i in a_2]
        if not self.dilation:
            a_2 = [self.down_sampling(a2_i) for a2_i in a_2]
        
        # Attention block 3 -> Apply attention over last residual block
        active_encoder_att_3 = [v for (task, v) in self.encoder_att_3.items() if task in active_tasks]
        a_3_mask = [att_i(torch.cat((u_3_b, a_2_i), dim=1)) for a_2_i, att_i in zip(a_2, active_encoder_att_3)]
        a_3 = [a_3_mask_i * u_3_t for a_3_mask_i in a_3_mask]
        a_3 = [self.encoder_block_att_3(a_3_i) for a_3_i in a_3]
        if not self.dilation:
            a_3 = [self.down_sampling(a3_i) for a3_i in a_3]
        
        # Attention block 4 -> Apply attention over last residual block (without final encoder)
        active_encoder_att_4 = [v for (task, v) in self.encoder_att_4.items() if task in active_tasks]
        a_4_mask = [att_i(torch.cat((u_4_b, a_3_i), dim=1)) for a_3_i, att_i in zip(a_3, active_encoder_att_4)]
        a_4 = [a_4_mask_i * u_4_t for a_4_mask_i in a_4_mask]
        
        # Brain dump: 
        # Maybe add extra down_sampling prior to reconstruction to force same features for recon and classification tasks. 
        
        #return a_4
        # Task specific decoders
        out =  {t: 0 for t in active_tasks}
        for i, t in enumerate(active_tasks):
            if t in self.encoder_decoder_tasks:
                if self.dilation: 
                    out[t] = F.interpolate(self.decoders[t](a_4[i]), size=out_size, mode='bilinear', align_corners=True)
                else: 
                    # not neccesary w.o. dilation. But maybe usefull if differently sized image is used
                    out[t] = self.decoders[t](a_4[i])
            else: 
                out[t] = torch.flatten(self.pooling_at_bottleneck(a_4[i]), 1)

        if self.skip_recon: 
            recon = x_fake_recon*0.0
        else: 
            recon = out[active_decoder_task]

        latent = out[active_attention_task]
        return recon, latent 

    
    def att_layer(self, in_channel, intermediate_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=intermediate_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(intermediate_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=intermediate_channel, out_channels=out_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid())
        
    def conv_layer(self, in_channel, out_channel):
        downsample = nn.Sequential(conv1x1(in_channel, 4 * out_channel, stride=1),
                                   nn.BatchNorm2d(4 * out_channel))
        return Bottleneck(in_channel, out_channel, downsample=downsample)
    


    ## The same model, but the reconstruction is directly coupled to the backbone

class MTANReconv2(nn.Module):
    def __init__(self, 
                 attention_paths: Dict[str, List[str]], 
                 backbone_name: str = "resnet18", 
                 dilation=True,
                 skip_recon=True,
                 recon_out_ch = 3):
        super(MTANReconv2, self).__init__()
        encoder_name = resnet_infos[backbone_name]["encoder_name"]
        self.skip_recon = skip_recon
        self.dilation = dilation
        if dilation: 
            backbone = ResnetDilated(resnet.__dict__[encoder_name](pretrained=True))
            self.shared_conv = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu1, backbone.maxpool)
        else: 
            backbone = resnet.__dict__[encoder_name](pretrained=True)
            self.shared_conv = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)   
        ch = resnet_infos[backbone_name]["ch"]
        self.latent_dim = ch[-1]
        if self.skip_recon:
            self.encoder_decoder_tasks = []
            self.num_out_channels_decoder = {}
        else: 
            self.encoder_decoder_tasks = ["recon"]
            self.num_out_channels_decoder = {"recon": recon_out_ch}

        self.score_types_to_attention_paths = make_score_type_2_attention_paths_dct(attention_paths)
        self.attention_paths = attention_paths.copy()
        self.encoder_only_tasks = list(self.attention_paths.keys())
        self.tasks = self.encoder_only_tasks




        # We will apply the attention over the last bottleneck layer in the ResNet. 
        self.shared_layer1_b = backbone.layer1[:-1] 
        self.shared_layer1_t = backbone.layer1[-1]

        self.shared_layer2_b = backbone.layer2[:-1]
        self.shared_layer2_t = backbone.layer2[-1]

        self.shared_layer3_b = backbone.layer3[:-1]
        self.shared_layer3_t = backbone.layer3[-1]

        self.shared_layer4_b = backbone.layer4[:-1]
        self.shared_layer4_t = backbone.layer4[-1]

        # Define task specific attention modules using a similar bottleneck design in residual block
        # (to avoid large computations)
        self.encoder_att_1 = nn.ModuleDict({k: self.att_layer(ch[0], ch[0] // 4, ch[0]) for k in self.tasks})
        self.encoder_att_2 = nn.ModuleDict({k: self.att_layer(2 * ch[1], ch[1] // 4, ch[1]) for k in self.tasks})
        self.encoder_att_3 = nn.ModuleDict({k: self.att_layer(2 * ch[2], ch[2] // 4, ch[2]) for k in self.tasks})
        self.encoder_att_4 = nn.ModuleDict({k: self.att_layer(2 * ch[3], ch[3] // 4, ch[3]) for k in self.tasks})

        # Define task shared attention encoders using residual bottleneck layers
        # We do not apply shared attention encoders at the last layer,
        # so the attended features will be directly fed into the task-specific decoders.
        self.encoder_block_att_1 = self.conv_layer(ch[0], ch[1] // 4)
        self.encoder_block_att_2 = self.conv_layer(ch[1], ch[2] // 4)
        self.encoder_block_att_3 = self.conv_layer(ch[2], ch[3] // 4)
        
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pooling_at_bottleneck = nn.AdaptiveAvgPool2d((1,1))


        # Define task-specific decoders using ASPP modules
        if not self.skip_recon: 
            # self.decoders = nn.ModuleDict({t: DeepLabHead(ch[-1], self.num_out_channels_decoder[t]) for t in self.encoder_decoder_tasks})
            self.decoders = nn.ModuleDict({t: decoder_by_name[backbone_name](out_ch=self.num_out_channels_decoder[t]) for t in self.encoder_decoder_tasks})


    def forward(self, x: torch.Tensor, score_type: List[str], out_size: Tuple[int] = (128,128)):
        
        # Only one encoder_only path is active (for now)!
        assert len(score_type) == x.shape[0], f"shapes of input dont match. Should both be batch dim"
        active_encoder_only_paths = [self.score_types_to_attention_paths[s] for s in score_type]
        assert len(set(active_encoder_only_paths)) == 1, f"More than one path is active for these types {(set(score_type))}  {set(active_encoder_only_paths)}"
        active_attention_task = active_encoder_only_paths[0]
        active_tasks = [active_attention_task]  # no decoder task!!

        if self.skip_recon: 
            x_fake_recon = x
        else:
            assert len(self.encoder_decoder_tasks) == 1, "Assuming only one decoder task for now!"
            active_decoder_task = list(self.encoder_decoder_tasks)[0]

        # Shared convolution
        x = self.shared_conv(x)
        
        # Shared ResNet block 1
        u_1_b = self.shared_layer1_b(x)
        u_1_t = self.shared_layer1_t(u_1_b)

        # Shared ResNet block 2
        u_2_b = self.shared_layer2_b(u_1_t)
        u_2_t = self.shared_layer2_t(u_2_b)

        # Shared ResNet block 3
        u_3_b = self.shared_layer3_b(u_2_t)
        u_3_t = self.shared_layer3_t(u_3_b)
        
        # Shared ResNet block 4
        u_4_b = self.shared_layer4_b(u_3_t)
        u_4_t = self.shared_layer4_t(u_4_b)

        # Attention block 1 -> Apply attention over last residual block
        active_encoder_att_1 = [v for (task,v) in self.encoder_att_1.items() if task in active_tasks]
        a_1_mask = [att_i(u_1_b) for att_i in active_encoder_att_1]
        a_1 = [a_1_mask_i * u_1_t for a_1_mask_i in a_1_mask]
        a_1 = [self.down_sampling(self.encoder_block_att_1(a_1_i)) for a_1_i in a_1]
        
        # Attention block 2 -> Apply attention over last residual block
        active_encoder_att_2 = [v for (task,v) in self.encoder_att_2.items() if task in active_tasks]
        a_2_mask = [att_i(torch.cat((u_2_b, a_1_i), dim=1)) for a_1_i, att_i in zip(a_1, active_encoder_att_2)]
        a_2 = [a_2_mask_i * u_2_t for a_2_mask_i in a_2_mask]
        a_2 = [self.encoder_block_att_2(a_2_i) for a_2_i in a_2]
        if not self.dilation:
            a_2 = [self.down_sampling(a2_i) for a2_i in a_2]
        
        # Attention block 3 -> Apply attention over last residual block
        active_encoder_att_3 = [v for (task, v) in self.encoder_att_3.items() if task in active_tasks]
        a_3_mask = [att_i(torch.cat((u_3_b, a_2_i), dim=1)) for a_2_i, att_i in zip(a_2, active_encoder_att_3)]
        a_3 = [a_3_mask_i * u_3_t for a_3_mask_i in a_3_mask]
        a_3 = [self.encoder_block_att_3(a_3_i) for a_3_i in a_3]
        if not self.dilation:
            a_3 = [self.down_sampling(a3_i) for a3_i in a_3]
        
        # Attention block 4 -> Apply attention over last residual block (without final encoder)
        active_encoder_att_4 = [v for (task, v) in self.encoder_att_4.items() if task in active_tasks]
        a_4_mask = [att_i(torch.cat((u_4_b, a_3_i), dim=1)) for a_3_i, att_i in zip(a_3, active_encoder_att_4)]
        a_4 = [a_4_mask_i * u_4_t for a_4_mask_i in a_4_mask]
        
        # Brain dump: 
        # Maybe add extra down_sampling prior to reconstruction to force same features for recon and classification tasks. 
        
        #return a_4
        # Task specific decoders
        out =  {t: 0 for t in active_tasks}
        for i, t in enumerate(active_tasks):
            out[t] = torch.flatten(self.pooling_at_bottleneck(a_4[i]), 1)
            
        if self.skip_recon: 
            recon = x_fake_recon*0.0
        else: 
            decoder_input = u_4_t  # no attention mask for this task
            if self.dilation: 
                recon = F.interpolate(self.decoders[active_decoder_task](decoder_input), size=out_size, mode='bilinear', align_corners=True)
            else: 
                # not neccesary w.o. dilation. But maybe usefull if differently sized image is used
                recon = self.decoders[active_decoder_task](decoder_input)

        latent = out[active_attention_task]
        return recon, latent 

    
    def att_layer(self, in_channel, intermediate_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=intermediate_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(intermediate_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=intermediate_channel, out_channels=out_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid())
        
    def conv_layer(self, in_channel, out_channel):
        downsample = nn.Sequential(conv1x1(in_channel, 4 * out_channel, stride=1),
                                   nn.BatchNorm2d(4 * out_channel))
        return Bottleneck(in_channel, out_channel, downsample=downsample)