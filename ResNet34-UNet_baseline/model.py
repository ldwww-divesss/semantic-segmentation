import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

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
        x = self.conv(x)
        return x

class ResNetUNet(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        
        resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        
        # Encoder (使用 ResNet 的各个阶段)
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # 64通道
        self.pool = resnet.maxpool
        self.enc2 = resnet.layer1 # 64通道
        self.enc3 = resnet.layer2 # 128通道
        self.enc4 = resnet.layer3 # 256通道
        self.enc5 = resnet.layer4 # 512通道

        # Decoder
        self.dec4 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec2 = DecoderBlock(128, 64)
        self.dec1 = DecoderBlock(64, 64)

        # 最终输出层
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        # 解码与跳跃连接
        d4 = self.dec4(e5, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        
        # 由于 resnet.conv1 步长为 2，我们多做一次上采样恢复原图分辨率
        d1 = self.dec1(d2, e1)
        d0 = nn.functional.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)
        
        return self.final_conv(d0)