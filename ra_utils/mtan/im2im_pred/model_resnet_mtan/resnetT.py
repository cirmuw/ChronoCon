import torch
import torch.nn as nn

# ---------------------------------------------------------------------
# helper layers – transposed variants of the 3×3 / 1×1 convolutions
# ---------------------------------------------------------------------
def deconv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3×3 transposed‐conv with output size matching the forward 3×3 conv"""
    return nn.ConvTranspose2d(
        in_planes, out_planes,
        kernel_size=3, stride=stride,
        padding=dilation, output_padding=stride - 1,
        groups=groups, bias=False, dilation=dilation
    )


def deconv1x1(in_planes, out_planes, stride=1):
    """1×1 transposed‐conv (optionally upsamples if stride > 1)"""
    return nn.ConvTranspose2d(
        in_planes, out_planes,
        kernel_size=1, stride=stride,
        output_padding=stride - 1, bias=False
    )

# ---------------------------------------------------------------------
# transposed basic / bottleneck residual blocks
# ---------------------------------------------------------------------
class BasicBlockT(nn.Module):
    expansion = 1
    __constants__ = ['skip']

    def __init__(self, inplanes, planes, stride=1,
                 skip=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlockT only supports groups=1, base_width=64')
        if dilation > 1:
            raise NotImplementedError("dilation > 1 not supported in BasicBlockT")

        self.deconv1 = deconv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.deconv2 = deconv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.skip = skip    # might upsample the identity path
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.deconv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.deconv2(out)
        out = self.bn2(out)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        out = self.relu(out)
        return out


class BottleneckT(nn.Module):
    expansion = 4
    __constants__ = ['skip']

    def __init__(self, inplanes, planes, stride=1,
                 skip=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.deconv3 = deconv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.deconv2 = deconv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.deconv1 = deconv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.relu = nn.ReLU(inplace=True)
        self.skip = skip
        self.stride = stride

    # forward is written in the natural order (1‑2‑3) for clarity,
    # but mirror of the encoder’s 3‑2‑1 is also valid.
    def forward(self, x):
        identity = x

        out = self.deconv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.deconv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.deconv3(out)
        out = self.bn3(out)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        out = self.relu(out)
        return out

# ---------------------------------------------------------------------
# the full decoder – a “ResNet in reverse”
# ---------------------------------------------------------------------
class ResNetDecoder(nn.Module):
    """
    Mirrors the encoder you posted.  For an encoder that outputs a
    feature map of shape [B, 512, H/32, W/32] (ResNet‑18/34),
    this decoder returns a reconstruction of shape [B, out_ch, H, W].
    """
    def __init__(self, block, layers, out_ch=3,
                 groups=1, width_per_group=64, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm = norm_layer

        self.inplanes = 512 * block.expansion   # start where the encoder ends
        self.groups = groups
        self.base_width = width_per_group

        # ❶ reverse order of encoder layers
        self.up4 = self._make_layer(block, 256, layers[3], stride=2)  # H/32 → H/16
        self.up3 = self._make_layer(block, 128, layers[2], stride=2)  # H/16 → H/8
        self.up2 = self._make_layer(block,  64, layers[1], stride=2)  # H/8  → H/4
        self.up1 = self._make_layer(block,  64, layers[0], stride=1)  # keep H/4

        # ❷ undo max‑pool and the first 7×7/stride‑2 conv from the encoder
        self.deconv_mp = nn.Sequential(
            deconv3x3(64 * block.expansion, 64, stride=2),  # H/4 → H/2
            norm_layer(64),
            nn.ReLU(inplace=True)
        )
        self.deconv0 = nn.ConvTranspose2d(
            64, out_ch, kernel_size=7, stride=2,
            padding=3, output_padding=1, bias=True
        )
        # optional output non‑linearity
        # self.out_act = nn.Sigmoid()  # or nn.Tanh()

        # init like torchvision
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # -----------------------------------------------------------------
    # helper identical in spirit to the encoder’s _make_layer
    # -----------------------------------------------------------------
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm
        skip = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            skip = nn.Sequential(
                deconv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, skip,
                        self.groups, self.base_width, 1, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                groups=self.groups, base_width=self.base_width,
                                dilation=1, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------
    def forward(self, x):
        x = self.up4(x)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)

        x = self.deconv_mp(x)
        x = self.deconv0(x)
        # x = self.out_act(x)
        return x


# ---------------------------------------------------------------------
# convenience constructors (mirror names of torchvision ResNets)
# ---------------------------------------------------------------------
def _resnet_decoder(block, layers, **kwargs):
    return ResNetDecoder(block, layers, **kwargs)


def resnet18_decoder(out_ch=3, **kwargs):
    return _resnet_decoder(BasicBlockT, [2, 2, 2, 2], out_ch=out_ch, **kwargs)


def resnet34_decoder(out_ch=3, **kwargs):
    return _resnet_decoder(BasicBlockT, [3, 4, 6, 3], out_ch=out_ch, **kwargs)



# --------------------------------------------------------------------
# ResNet-Bottleneck family
# --------------------------------------------------------------------
def resnet50_decoder(out_ch=3, **kwargs):
    """Mirror of ResNet-50 but with transposed layers (decoder)."""
    return _resnet_decoder(BottleneckT, [3, 4, 6, 3],
                           out_ch=out_ch, **kwargs)


def resnet101_decoder(out_ch=3, **kwargs):
    """Mirror of ResNet-101 (3-4-23-3)."""
    return _resnet_decoder(BottleneckT, [3, 4, 23, 3],
                           out_ch=out_ch, **kwargs)


def resnet152_decoder(out_ch=3, **kwargs):
    """Mirror of ResNet-152 (3-8-36-3)."""
    return _resnet_decoder(BottleneckT, [3, 8, 36, 3],
                           out_ch=out_ch, **kwargs)


# --------------------------------------------------------------------
# ResNeXt family (grouped convolutions)
# --------------------------------------------------------------------
def resnext50_32x4d_decoder(out_ch=3, **kwargs):
    """Mirror of ResNeXt-50 32×4d (cardinality = 32, base width = 4)."""
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet_decoder(BottleneckT, [3, 4, 6, 3],
                           out_ch=out_ch, **kwargs)


def resnext101_32x8d_decoder(out_ch=3, **kwargs):
    """Mirror of ResNeXt-101 32×8d."""
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet_decoder(BottleneckT, [3, 4, 23, 3],
                           out_ch=out_ch, **kwargs)


# --------------------------------------------------------------------
# Wide-ResNet family (double bottleneck width)
# --------------------------------------------------------------------
def wide_resnet50_2_decoder(out_ch=3, **kwargs):
    """Mirror of Wide-ResNet-50-2 (bottleneck width × 2)."""
    kwargs['width_per_group'] = 64 * 2   # torchvision convention
    return _resnet_decoder(BottleneckT, [3, 4, 6, 3],
                           out_ch=out_ch, **kwargs)


def wide_resnet101_2_decoder(out_ch=3, **kwargs):
    """Mirror of Wide-ResNet-101-2."""
    kwargs['width_per_group'] = 64 * 2
    return _resnet_decoder(BottleneckT, [3, 4, 23, 3],
                           out_ch=out_ch, **kwargs)