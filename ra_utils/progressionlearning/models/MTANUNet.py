import torch
import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, in_dim, out_dim, pool_param):
        super(Projector, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_param)
        self.projector = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        x = self.pool(x).flatten(1)
        return self.projector(x)
    

class MTANRecUnet(nn.Module):
    def __init__(self, unet, filters=[16,32,64,128,256,32], merge_with_attention=False, disentangle=False):
        super(MTANRecUnet, self).__init__()
        self.unet = unet
        self.merge_with_attention = merge_with_attention
        self.disentangle = disentangle
        self.projector_1 = Projector(in_dim=filters[0], out_dim=filters[0], pool_param=(1,1))
        self.projector_2 = Projector(in_dim=filters[1], out_dim=filters[1], pool_param=(1,1))
        self.projector_3 = Projector(in_dim=filters[2], out_dim=filters[2], pool_param=(1,1))
        self.projector_4 = Projector(in_dim=filters[3], out_dim=filters[3], pool_param=(1,1))
        self.attention_blocks = nn.ModuleList([self.att_layer([filters[i], filters[i], filters[i]]) for i in range(4)])

    def att_layer(self, channel):
        att_block = nn.Sequential(
            nn.Conv2d(channel[0], channel[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel[1], channel[2], kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        return att_block
    
    def mask_feature_maps(self, encoder_features, i):
        
        attn_mask = self.attention_blocks[i](encoder_features)
        refined_feature = attn_mask * encoder_features  # Apply attention
        if self.disentangle:
            #return refined_feature, (attn_mask, 1-attn_mask)
            return refined_feature, (refined_feature, 1-refined_feature)
        else:
            return refined_feature, None
    
    def forward(self, x):
        rec = None
        latent = None
        rec = self.unet(x)
        x = self.unet.conv_0(x)
        out1 = self.unet.down_1(x)
        out2 = self.unet.down_2(out1)
        out3 = self.unet.down_3(out2)
        out4 = self.unet.down_4(out3)
        
        # apply attention masks
        out1, masks1 = self.mask_feature_maps(out1, 0)
        out2, masks2 = self.mask_feature_maps(out2, 1)
        out3, masks3 = self.mask_feature_maps(out3, 2)
        out4, masks4 = self.mask_feature_maps(out4, 3)

        # get the multi-scale embeddings
        out1 = self.projector_1(out1)
        out2 = self.projector_2(out2)
        out3 = self.projector_3(out3)
        out4 = self.projector_4(out4)
        latent = torch.cat([out1, out2, out3, out4], dim=-1) 

        if self.disentangle: 
            return rec, latent, (masks1, masks2, masks3, masks4)
        else:
            return rec, latent
        

#----------------------------------------------------------------------------#
#          CW: Additions                                                     #
#----------------------------------------------------------------------------#

class MTANRecUnet_v2(nn.Module):
    def __init__(self, unet, filters=[16,32,64,128,256,32]):
        super(MTANRecUnet_v2, self).__init__()
        self.unet = unet
        self.projector_1 = Projector(in_dim=filters[0], out_dim=filters[0], pool_param=(1,1))
        self.projector_2 = Projector(in_dim=filters[1], out_dim=filters[1], pool_param=(1,1))
        self.projector_3 = Projector(in_dim=filters[2], out_dim=filters[2], pool_param=(1,1))
        self.projector_4 = Projector(in_dim=filters[3], out_dim=filters[3], pool_param=(1,1))
        self.attention_blocks = nn.ModuleList([self.att_layer([filters[i], filters[i], filters[i]]) for i in range(4)])

    def att_layer(self, channel):
        att_block = nn.Sequential(
            nn.Conv2d(channel[0], channel[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel[1], channel[2], kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        return att_block
    
    def mask_feature_maps(self, encoder_features, i):
        attn_mask = self.attention_blocks[i](encoder_features)
        refined_feature = attn_mask * encoder_features  # Apply attention
        return refined_feature
    
    def forward(self, x):
        rec = None
        latent = None
        rec = self.unet(x)
        x = self.unet.conv_0(x)
        out1 = self.unet.down_1(x)
        out2 = self.unet.down_2(out1)
        out3 = self.unet.down_3(out2)
        out4 = self.unet.down_4(out3)
        
        # apply attention masks
        out1 = self.mask_feature_maps(out1, 0)
        out2 = self.mask_feature_maps(out2, 1)
        out3 = self.mask_feature_maps(out3, 2)
        out4 = self.mask_feature_maps(out4, 3)

        # get the multi-scale embeddings
        out1 = self.projector_1(out1)
        out2 = self.projector_2(out2)
        out3 = self.projector_3(out3)
        out4 = self.projector_4(out4)
        latent = torch.cat([out1, out2, out3, out4], dim=-1) 

        return rec, latent
    
from typing import Dict, List

def make_score_type_2_attention_paths_dct(attention_paths: Dict[str, List[str]]):
    out = {}
    used_score_types = []
    for k,v in attention_paths.items():
        assert set(v) & set(used_score_types) == set(), f"duplicate score type in dct {v} | {used_score_types}"
        for vv in v:
            out[vv] = k
    return out

class MTANRecUnet_v3(nn.Module):
    def __init__(self, 
                 unet, 
                 attention_paths: Dict[str, List[str]],
                 filters=[16,32,64,128,256,32],
                 ):
        super(MTANRecUnet_v3, self).__init__()
        self.attention_paths = attention_paths.copy()
        self.score_types_to_attention_paths = make_score_type_2_attention_paths_dct(attention_paths)
        self.unet = unet
        self.projector_1 = Projector(in_dim=filters[0], out_dim=filters[0], pool_param=(1,1))
        self.projector_2 = Projector(in_dim=filters[1], out_dim=filters[1], pool_param=(1,1))
        self.projector_3 = Projector(in_dim=filters[2], out_dim=filters[2], pool_param=(1,1))
        self.projector_4 = Projector(in_dim=filters[3], out_dim=filters[3], pool_param=(1,1))
        
        # Each "task" gets their own attention blocks 
        # A task corresponds to a ROI type not really a MultiTask setting per se. 
        self.attention_blocks = nn.ModuleDict(
            {k: nn.ModuleList([self.att_layer([filters[i], filters[i], filters[i]]) for i in range(4)])
             for k in attention_paths})

    def att_layer(self, channel):
        att_block = nn.Sequential(
            nn.Conv2d(channel[0], channel[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel[1], channel[2], kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        return att_block
    
    def mask_feature_maps(self, encoder_features, i, active_attention_path: str):
        attn_mask = self.attention_blocks[active_attention_path][i](encoder_features)
        refined_feature = attn_mask * encoder_features  # Apply attention
        return refined_feature
    
    def forward(self, x: torch.Tensor, score_type: List[str]):
        assert len(score_type) == x.shape[0], f"shapes of input dont match. Should both be batch dim"
        active_attention_paths = [self.score_types_to_attention_paths[s] for s in score_type]
        assert len(set(active_attention_paths)) == 1, f"More than one path is active for these types {(set(score_type))}  {set(active_attention_paths)}"
        active_attention_path = active_attention_paths[0]
        
        rec = None
        latent = None
        rec = self.unet(x)
        x = self.unet.conv_0(x)
        out1 = self.unet.down_1(x)
        out2 = self.unet.down_2(out1)
        out3 = self.unet.down_3(out2)
        out4 = self.unet.down_4(out3)
        
        # apply attention masks
        out1 = self.mask_feature_maps(out1, 0, active_attention_path)
        out2 = self.mask_feature_maps(out2, 1, active_attention_path)
        out3 = self.mask_feature_maps(out3, 2, active_attention_path)
        out4 = self.mask_feature_maps(out4, 3, active_attention_path)

        # get the multi-scale embeddings
        out1 = self.projector_1(out1)
        out2 = self.projector_2(out2)
        out3 = self.projector_3(out3)
        out4 = self.projector_4(out4)
        latent = torch.cat([out1, out2, out3, out4], dim=-1) 

        return rec, latent