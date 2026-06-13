import torch
import torch.nn as nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights


def _conv_bn_relu(in_channels, out_channels):
    """SegNet 的基本卷积单元：Conv3x3 + BN + ReLU。"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class SegNet(nn.Module):
    """
    经典 SegNet（Badrinarayanan et al., 2017）。

    特点：编码器与 VGG16 对齐；解码器通过最大池化索引(max-pooling indices)做
    非线性上采样(MaxUnpool)，而不是转置卷积或双线性插值——这是 SegNet 区别于
    UNet/FCN 的核心。编码器权重默认从 ImageNet 预训练的 VGG16-BN 初始化。
    """

    def __init__(self, num_classes=12, pretrained=True):
        super().__init__()

        # ---------------- 编码器（对应 VGG16 的 5 个阶段）----------------
        self.enc1 = nn.Sequential(_conv_bn_relu(3, 64),    _conv_bn_relu(64, 64))
        self.enc2 = nn.Sequential(_conv_bn_relu(64, 128),  _conv_bn_relu(128, 128))
        self.enc3 = nn.Sequential(_conv_bn_relu(128, 256), _conv_bn_relu(256, 256), _conv_bn_relu(256, 256))
        self.enc4 = nn.Sequential(_conv_bn_relu(256, 512), _conv_bn_relu(512, 512), _conv_bn_relu(512, 512))
        self.enc5 = nn.Sequential(_conv_bn_relu(512, 512), _conv_bn_relu(512, 512), _conv_bn_relu(512, 512))

        # 池化时记录索引，解码端凭索引精确还原位置
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        # ---------------- 解码器（与编码器镜像，通道逐级降回）----------------
        self.dec5 = nn.Sequential(_conv_bn_relu(512, 512), _conv_bn_relu(512, 512), _conv_bn_relu(512, 512))
        self.dec4 = nn.Sequential(_conv_bn_relu(512, 512), _conv_bn_relu(512, 512), _conv_bn_relu(512, 256))
        self.dec3 = nn.Sequential(_conv_bn_relu(256, 256), _conv_bn_relu(256, 256), _conv_bn_relu(256, 128))
        self.dec2 = nn.Sequential(_conv_bn_relu(128, 128), _conv_bn_relu(128, 64))
        self.dec1 = nn.Sequential(_conv_bn_relu(64, 64))

        # 最终 1x1 分类头
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

        if pretrained:
            self._init_encoder_from_vgg()

    def _init_encoder_from_vgg(self):
        """把 ImageNet 预训练 VGG16-BN 的卷积/BN 权重拷贝进编码器。"""
        try:
            vgg = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
        except Exception as e:  # 离线环境下载失败时退回随机初始化
            print(f"[SegNet] 无法获取 VGG16-BN 预训练权重，改用随机初始化：{e}")
            return
        # vgg.features 与编码器的 Conv/BN 顺序完全一致，逐层对齐拷贝
        vgg_layers = [m for m in vgg.features if isinstance(m, (nn.Conv2d, nn.BatchNorm2d))]
        enc_layers = [
            m for blk in (self.enc1, self.enc2, self.enc3, self.enc4, self.enc5)
            for m in blk.modules() if isinstance(m, (nn.Conv2d, nn.BatchNorm2d))
        ]
        for dst, src in zip(enc_layers, vgg_layers):
            dst.load_state_dict(src.state_dict())

    def forward(self, x):
        # ---- 编码：记录每级池化前的尺寸 s 与索引 i ----
        x = self.enc1(x); s1 = x.size(); x, i1 = self.pool(x)
        x = self.enc2(x); s2 = x.size(); x, i2 = self.pool(x)
        x = self.enc3(x); s3 = x.size(); x, i3 = self.pool(x)
        x = self.enc4(x); s4 = x.size(); x, i4 = self.pool(x)
        x = self.enc5(x); s5 = x.size(); x, i5 = self.pool(x)

        # ---- 解码：用编码端索引做最大反池化，再卷积细化 ----
        x = self.dec5(self.unpool(x, i5, output_size=s5))
        x = self.dec4(self.unpool(x, i4, output_size=s4))
        x = self.dec3(self.unpool(x, i3, output_size=s3))
        x = self.dec2(self.unpool(x, i2, output_size=s2))
        x = self.dec1(self.unpool(x, i1, output_size=s1))

        return self.classifier(x)
