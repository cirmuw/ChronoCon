import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------
# ❶ building blocks you already have
# ---------------------------------------------------------------------
# from ra_utils.mtan.im2im_pred.model_resnet_mtan.resnet import Bottleneck, conv1x1, resnet50
# from ra_utils.mtan.im2im_pred.model_resnet_mtan.resnetT import BottleneckT, resnet50_decoder




import torch, torch.nn as nn, torch.nn.functional as F
from functools import partial

# ---------------------------------------------------------------
# ❶ factory helpers
# ---------------------------------------------------------------
ENCODER_CTOR = {
    'resnet18':  ('basic', 512),
    'resnet34':  ('basic', 512),
    'resnet50':  ('bottleneck', 2048),
}
DECODER_CTOR = {
    'resnet18':  'resnet18_decoder',
    'resnet34':  'resnet34_decoder',
    'resnet50':  'resnet50_decoder',
}

from ra_utils.mtan.im2im_pred.model_resnet_mtan.resnet   import resnet18, resnet34, resnet50, BasicBlock, Bottleneck, conv1x1
from ra_utils.mtan.im2im_pred.model_resnet_mtan.resnetT  import resnet18_decoder, resnet34_decoder, resnet50_decoder

encoder_by_name = dict(resnet18=resnet18, resnet34=resnet34, resnet50=resnet50)
decoder_by_name = dict(resnet18=resnet18_decoder,
                       resnet34=resnet34_decoder,
                       resnet50=resnet50_decoder)

# ------------------------------------------------------------------ #
# ❷  utils
# ------------------------------------------------------------------ #
def get_stage_channels(backbone: nn.Module):
    """Return output channels of conv2_x … conv5_x for any torchvision ResNet."""
    stages = [backbone.layer1[-1], backbone.layer2[-1],
              backbone.layer3[-1], backbone.layer4[-1]]
    chans = []
    for blk in stages:
        if isinstance(blk, BasicBlock):           # expansion = 1
            chans.append(blk.bn2.num_features)    # 64,128,256,512
        else:                                     # Bottleneck (exp=4)
            chans.append(blk.bn3.num_features)    # 256,512,1024,2048
    return chans                                  # length = 4


def bridging_block(in_ch: int, out_ch: int, block_kind: str):
    """Residual block + (optional) 1×1 downsample so that out == out_ch."""
    if block_kind == "basic":                     # ResNet-18 / 34
        down = None
        if in_ch != out_ch:
            down = nn.Sequential(conv1x1(in_ch, out_ch, 1),
                                   nn.BatchNorm2d(out_ch))
        return BasicBlock(in_ch, out_ch, stride=1, downsample=down)

    # bottleneck
    planes = out_ch // 4                          # because expansion = 4
    down = nn.Sequential(conv1x1(in_ch, out_ch, 1),
                         nn.BatchNorm2d(out_ch))
    return Bottleneck(in_ch, planes, stride=1, downsample=down)


def att_layer(in_ch: int, out_ch: int, ratio: int = 4):
    mid = max(in_ch // ratio, 16)
    return nn.Sequential(
        nn.Conv2d(in_ch, mid, 1),
        nn.BatchNorm2d(mid),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid, out_ch, 1),
        nn.BatchNorm2d(out_ch),
        nn.Sigmoid(),
    )

# ------------------------------------------------------------------ #
# ❸  MTAN module
# ------------------------------------------------------------------ #
class MTANReconCls(nn.Module):
    """
    Multi-Task Attention Network
      * one image reconstruction decoder
      * arbitrary number of classification heads
    """

    def __init__(
        self,
        classification_task_names,         # list[str]
        classification_task_n_classes,     # list[int] (same order)
        backbone_name: str = "resnet18",
        img_size=(128, 128),
    ):
        super().__init__()
        assert len(classification_task_names) == len(
            classification_task_n_classes
        ), "task_names and n_classes length mismatch"

        # ----- backbone / decoder -----------------------------------
        self.backbone_name = backbone_name
        block_kind = "basic" if backbone_name in ("resnet18", "resnet34") else "bottleneck"
        encoder = encoder_by_name[backbone_name](pretrained=True)
        decoder = decoder_by_name[backbone_name](out_ch=3)

        c1, c2, c3, c4 = get_stage_channels(encoder)
        ch = [c1, c2, c3, c4]                          # convenience

        # ----- split encoder exactly once ---------------------------
        self.conv0 = nn.Sequential(
            encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool
        )
        self.l1_b, self.l1_t = encoder.layer1[:-1], encoder.layer1[-1]
        self.l2_b, self.l2_t = encoder.layer2[:-1], encoder.layer2[-1]
        self.l3_b, self.l3_t = encoder.layer3[:-1], encoder.layer3[-1]
        self.l4_b, self.l4_t = encoder.layer4[:-1], encoder.layer4[-1]

        # ----- attention for every task at every stage --------------
        self.tasks = ["reconstruction"] + list(classification_task_names)
        self.cls_tasks = list(classification_task_names)
        self.n_cls = dict(zip(classification_task_names, classification_task_n_classes))

        self.att1 = nn.ModuleList([att_layer(ch[0], ch[0]) for _ in self.tasks])
        self.att2 = nn.ModuleList([att_layer(2 * ch[1], ch[1]) for _ in self.tasks])
        self.att3 = nn.ModuleList([att_layer(2 * ch[2], ch[2]) for _ in self.tasks])
        self.att4 = nn.ModuleList([att_layer(2 * ch[3], ch[3]) for _ in self.tasks])

        # shared bridging convs
        self.br1 = bridging_block(ch[0], ch[1], block_kind)
        self.br2 = bridging_block(ch[1], ch[2], block_kind)
        self.br3 = bridging_block(ch[2], ch[3], block_kind)
        self.pool = nn.MaxPool2d(2, 2)

        # heads
        self.recon_decoder = decoder
        self.cls_heads = nn.ModuleDict(
            {t: nn.Linear(ch[3], self.n_cls[t]) for t in self.cls_tasks}
        )
        self.out_img_size = img_size

    # ----------------------------------------------------------------
    def forward(self, x):
        B = x.size(0)

        # ---------- encoder -----------------------------------------
        x0 = self.conv0(x)
        u1b = self.l1_b(x0)
        u1t = self.l1_t(u1b)
        u2b = self.l2_b(u1t)
        u2t = self.l2_t(u2b)
        u3b = self.l3_b(u2t)
        u3t = self.l3_t(u3b)
        u4b = self.l4_b(u3t)
        u4t = self.l4_t(u4b)                 # latent representation
        latent = u4t

        # ---------- MTAN attention ----------------------------------
        a1 = [
            self.pool(self.br1(mask(u1b) * u1t))
            for mask in self.att1
        ]

        a2 = [
            self.br2(mask(torch.cat([u2b, a1_i], 1)) * u2t)
            for mask, a1_i in zip(self.att2, a1)
        ]

        a3 = [
            self.br3(mask(torch.cat([u3b, a2_i], 1)) * u3t)
            for mask, a2_i in zip(self.att3, a2)
        ]

        a4 = [
            mask(torch.cat([u4b, a3_i], 1)) * u4t
            for mask, a3_i in zip(self.att4, a3)
        ]

        # ---------- reconstruction head -----------------------------
        recon = F.interpolate(
            self.recon_decoder(a4[0]),
            size=self.out_img_size,
            mode="bilinear",
            align_corners=True,
        )

        # ---------- classification heads ----------------------------
        logits = {}
        for idx, task in enumerate(self.cls_tasks, start=1):
            pooled = F.adaptive_avg_pool2d(a4[idx], 1).view(B, -1)
            logits[task] = self.cls_heads[task](pooled)

        return {
            "reconstruction": recon,
            "latent": latent,
            "logits": logits,
        }

# ------------------------------------------------------------------ #
# ❹  tiny factory for convenience
# ------------------------------------------------------------------ #
def build_mtan_recon_cls(
    classification_task_names,
    classification_task_n_classes,
    backbone_name="resnet18",
    img_size=(128, 128),
):
    return MTANReconCls(
        classification_task_names,
        classification_task_n_classes,
        backbone_name=backbone_name,
        img_size=img_size,
    )