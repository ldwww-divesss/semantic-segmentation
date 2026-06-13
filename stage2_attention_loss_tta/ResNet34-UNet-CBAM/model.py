import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

# =====================================================
# 1. Channel Attention
# =====================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))

        out = avg_out + max_out

        return self.sigmoid(out)

# =====================================================
# 2. Spatial Attention
# =====================================================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)

        max_out, _ = torch.max(x, dim=1, keepdim=True)

        x_cat = torch.cat([avg_out, max_out], dim=1)

        out = self.conv(x_cat)

        return self.sigmoid(out)

# =====================================================
# 3. CBAM
# =====================================================
class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):

        x = x * self.ca(x)

        x = x * self.sa(x)

        return x

# =====================================================
# 4. Decoder Block + CBAM
# =====================================================
# 将原来的 DecoderBlock 改名为 DecoderBlockCBAM，
# 再复制一份去掉 self.cbam 的版本

class DecoderBlock(nn.Module):
    """不带注意力的普通解码块（用于浅层）"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class DecoderBlockCBAM(nn.Module):
    """带 CBAM 的解码块（用于深层）"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.cbam = CBAM(out_channels)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.cbam(x)
        return x

# =====================================================
# 5. ResNet34-UNet-CBAM
# =====================================================
class ResNetUNet(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()

        resnet = resnet34(
            weights=ResNet34_Weights.IMAGENET1K_V1
        )

        # ================= Encoder =================
        self.enc1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )

        self.pool = resnet.maxpool

        self.enc2 = resnet.layer1
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3
        self.enc5 = resnet.layer4

        # ================= Decoder =================
        self.dec4 = DecoderBlockCBAM(512, 256)   # 最深，用CBAM
        self.dec3 = DecoderBlockCBAM(256, 128)   # 深层，用CBAM
        self.dec2 = DecoderBlock(128, 64)        # 浅层，不用
        self.dec1 = DecoderBlock(64, 64)         # 最浅，不用

        # ================= Output =================
        self.final_conv = nn.Conv2d(
            64,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        # ================= Encoder =================
        e1 = self.enc1(x)

        e2 = self.enc2(self.pool(e1))

        e3 = self.enc3(e2)

        e4 = self.enc4(e3)

        e5 = self.enc5(e4)

        # ================= Decoder =================
        d4 = self.dec4(e5, e4)

        d3 = self.dec3(d4, e3)

        d2 = self.dec2(d3, e2)

        d1 = self.dec1(d2, e1)

        # 恢复原图尺寸
        d0 = F.interpolate(
            d1,
            scale_factor=2,
            mode='bilinear',
            align_corners=False
        )

        out = self.final_conv(d0)

        return out