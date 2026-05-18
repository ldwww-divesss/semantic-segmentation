import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

# ==========================================
# 1. CBAM 注意力模块定义
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

# ==========================================
# 2. U-Net 解码器与主网络组装
# ==========================================
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ResNetUNetCBAM(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        # 加载预训练 ResNet34 作为 Encoder
        resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # 64
        self.pool = resnet.maxpool
        self.enc2 = resnet.layer1 # 64
        self.enc3 = resnet.layer2 # 128
        self.enc4 = resnet.layer3 # 256
        self.enc5 = resnet.layer4 # 512
        
        # 为每一层编码特征嵌入 CBAM 模块进行提纯
        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(64)
        self.cbam3 = CBAM(128)
        self.cbam4 = CBAM(256)
        self.cbam5 = CBAM(512)

        # Decoder
        self.dec4 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec2 = DecoderBlock(128, 64)
        self.dec1 = DecoderBlock(64, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # 编码 -> CBAM 注意力强化
        e1 = self.cbam1(self.enc1(x))
        e2 = self.cbam2(self.enc2(self.pool(e1)))
        e3 = self.cbam3(self.enc3(e2))
        e4 = self.cbam4(self.enc4(e3))
        e5 = self.cbam5(self.enc5(e4))

        # 解码与特征拼接
        d4 = self.dec4(e5, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        
        d0 = nn.functional.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)
        return self.final_conv(d0)