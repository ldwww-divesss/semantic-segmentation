"""
@author: ThengAndrew
@created_at: 2026-5-3
@updated_at: 2026-5-9
@usage:
    # 完整 CBAM（消融 iii）+ ImageNet 预训练骨干
    model = DeepLabV3PlusCBAM(
        num_classes=11,
        attention_type="cbam",
        cbam_use_channel=True,
        cbam_use_spatial=True,
        pretrained_backbone=True,
    ).to(device)
    logits = model(images)   # [B, 11, H, W]

    # ECA-Net / Coordinate Attention（四 Stage 插桩同 CBAM）
    model = DeepLabV3PlusCBAM(num_classes=11, attention_type="eca")
    model = DeepLabV3PlusCBAM(num_classes=11, attention_type="coord")

    # 无注意力（等价于标准 DeepLabV3+）
    model = DeepLabV3PlusCBAM(num_classes=11, attention_type="none")
@description: DeepLabV3+（ASPP + 低层特征解码器）+ 可选 Stage 注意力（CBAM / ECA / CoordAtt）。
    骨干：torchvision ResNet50，layer3/4 替换空洞卷积（OS=8），
    注意力插在 layer1-4 各 Stage 输出后；CBAM 可通过 use_channel/use_spatial 做三组消融。
    解码头遵循原论文（Chen et al., ECCV 2018）：ASPP → 上采样 → 拼接低层特征 → 预测。
@warning:
    - attention_type=eca|coord 时不使用 cbam 消融开关；
    - attention_type=none 时不插入任何注意力模块（纯 DeepLabV3+）；
    - 输入须为 ImageNet 归一化的 float tensor [B, 3, H, W]；
    - 输出尺寸与输入尺寸相同（内部双线性上采样），不需要手动 resize；
    - pretrained_backbone=True 首次运行会下载 ResNet50 权重（~98MB）。
@note:
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

from .cbam import CBAM
from .coordinate_attention import CoordinateAttention
from .eca import ECA


# ============================================================
# Stage 注意力工厂
# ============================================================

def make_stage_attention(
    attention_type: str,
    channels: int,
    *,
    cbam_use_channel: bool,
    cbam_use_spatial: bool,
) -> nn.Module:
    """在各 ResNet Stage 后插入的注意力模块（输入输出均为 [B,C,H,W]）。"""
    at = attention_type.lower()
    if at == "none":
        return nn.Identity()
    if at == "eca":
        return ECA(channels)
    if at == "coord":
        return CoordinateAttention(channels)
    if at == "cbam":
        need_cbam = cbam_use_channel or cbam_use_spatial
        if not need_cbam:
            return nn.Identity()
        return CBAM(
            channels,
            use_channel=cbam_use_channel,
            use_spatial=cbam_use_spatial,
        )
    raise ValueError(f"未知 attention_type: {attention_type!r}")


# ============================================================
# 骨干：ResNet50 with dilated conv + CBAM
# ============================================================

def _make_dilated(layer: nn.Sequential, stride: int, dilation: int) -> None:
    """把 ResNet layer 的第一个 Bottleneck stride 改为 1，并全层加 dilation。
    原地修改，不返回新对象。
    """
    for i, block in enumerate(layer):
        if i == 0 and stride != 1:
            # 修改第一个 block 的 conv2 stride（Bottleneck 的 3×3 卷积）
            block.conv2.stride = (1, 1)
            # 修改 downsample conv 的 stride
            if block.downsample is not None:
                block.downsample[0].stride = (1, 1)
        # 对所有 block 的 conv2 设置 dilation 和 padding
        block.conv2.dilation = (dilation, dilation)
        block.conv2.padding  = (dilation, dilation)


class ResNet50Backbone(nn.Module):
    """ResNet50 骨干（OS=8）+ 各 Stage 后可选注意力（CBAM / ECA / CoordAtt / 无）。

    输出两组特征：
      low  : layer1 输出，[B, 256, H/4, W/4]，DeepLabV3+ 解码器使用
      high : layer4 输出，[B, 2048, H/8, W/8]，送入 ASPP

    属性 cbam1～cbam4 为历史命名，泛指各 Stage 后的注意力模块。
    """

    def __init__(
        self,
        pretrained: bool = True,
        *,
        attention_type: str = "cbam",
        cbam_use_channel: bool = True,
        cbam_use_spatial: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base    = resnet50(weights=weights)

        # ---- Stem & pool ----
        self.stem = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool
        )

        # ---- Stage 1-4 (修改 layer3/4 为空洞卷积，OS=8) ----
        self.layer1 = base.layer1                        # stride=1, ch=256
        self.layer2 = base.layer2                        # stride=2, ch=512
        self.layer3 = base.layer3                        # 将改为 dilation=2
        self.layer4 = base.layer4                        # 将改为 dilation=4

        _make_dilated(self.layer3, stride=2, dilation=2)
        _make_dilated(self.layer4, stride=2, dilation=4)

        _attn = lambda c: make_stage_attention(
            attention_type,
            c,
            cbam_use_channel=cbam_use_channel,
            cbam_use_spatial=cbam_use_spatial,
        )
        self.cbam1 = _attn(256)
        self.cbam2 = _attn(512)
        self.cbam3 = _attn(1024)
        self.cbam4 = _attn(2048)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)                   # [B, 64,  H/4,  W/4]
        x = self.cbam1(self.layer1(x))     # [B, 256, H/4,  W/4]  ← low-level
        low = x
        x = self.cbam2(self.layer2(x))     # [B, 512, H/8,  W/8]
        x = self.cbam3(self.layer3(x))     # [B,1024, H/8,  W/8]
        x = self.cbam4(self.layer4(x))     # [B,2048, H/8,  W/8]  ← ASPP input
        return low, x


# ============================================================
# ASPP：Atrous Spatial Pyramid Pooling
# ============================================================

class _ASPPConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, dilation: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _ASPPPool(nn.Module):
    """ASPP 的全局平均池化分支。"""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return F.interpolate(
            self.proj(self.pool(x)), size=(h, w), mode="bilinear", align_corners=False
        )


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling（OS=8 时 rates=12,24,36）。"""

    # 论文 OS=8 dilations；OS=16 用 [6,12,18]
    _RATES = (12, 24, 36)

    def __init__(self, in_ch: int = 2048, out_ch: int = 256) -> None:
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(                                         # 1×1 conv
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ),
            *[_ASPPConv(in_ch, out_ch, r) for r in self._RATES],  # 3×3 dilated
            _ASPPPool(in_ch, out_ch),                              # global avg
        ])
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(self._RATES) + 2), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([c(x) for c in self.convs], dim=1))


# ============================================================
# DeepLabV3+ 解码器
# ============================================================

class DeepLabV3PlusDecoder(nn.Module):
    """低层特征 + ASPP 特征融合解码器。

    low-level features (256ch) → 1×1 conv → 48ch
    ASPP output (256ch)        → ×4 上采样
    concat → 3×3 conv × 2 → num_classes
    """

    def __init__(self, num_classes: int, low_ch: int = 256, aspp_ch: int = 256) -> None:
        super().__init__()
        # 低层特征降维
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_ch, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        # 融合后精细化
        self.refine = nn.Sequential(
            nn.Conv2d(aspp_ch + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Conv2d(256, num_classes, 1)

    def forward(
        self,
        low:  torch.Tensor,   # [B, 256, H/4, W/4]
        aspp: torch.Tensor,   # [B, 256, H/8, W/8]
        input_size: tuple[int, int],
    ) -> torch.Tensor:
        # ASPP 特征上采样到 low 的空间尺寸
        aspp_up = F.interpolate(
            aspp, size=low.shape[-2:], mode="bilinear", align_corners=False
        )
        low_feat = self.low_proj(low)
        x = self.refine(torch.cat([aspp_up, low_feat], dim=1))
        # 上采样到输入分辨率
        return F.interpolate(
            self.cls(x), size=input_size, mode="bilinear", align_corners=False
        )


# ============================================================
# 完整模型：DeepLabV3PlusCBAM
# ============================================================

class DeepLabV3PlusCBAM(nn.Module):
    """DeepLabV3+ with optional stage attention（阶段二主模型）。

    Parameters
    ----------
    num_classes
        输出类别数（CamVid=11，含 void=11 则仍输出 11 类，void 在 loss 中被 ignore）。
    attention_type
        cbam | eca | coord | none；eca/coord 时不使用 cbam 消融开关。
    cbam_use_channel / cbam_use_spatial
        仅 attention_type=cbam 时生效；两者都为 False 时不插入 CBAM（消融基准）。
    pretrained_backbone
        是否加载 ImageNet 预训练权重（首次需联网下载 ~98MB）。
    """

    def __init__(
        self,
        num_classes: int = 11,
        *,
        attention_type: str = "cbam",
        cbam_use_channel: bool = True,
        cbam_use_spatial:  bool = True,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.attention_type = attention_type.lower()
        self.backbone = ResNet50Backbone(
            pretrained=pretrained_backbone,
            attention_type=self.attention_type,
            cbam_use_channel=cbam_use_channel,
            cbam_use_spatial=cbam_use_spatial,
        )
        self.aspp    = ASPP(in_ch=2048, out_ch=256)
        self.decoder = DeepLabV3PlusDecoder(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : FloatTensor [B, 3, H, W]，ImageNet 归一化

        Returns
        -------
        logits : FloatTensor [B, num_classes, H, W]
        """
        input_size = (x.shape[-2], x.shape[-1])
        low, high  = self.backbone(x)
        aspp_out   = self.aspp(high)
        return self.decoder(low, aspp_out, input_size)

    @property
    def variant_name(self) -> str:
        """返回当前消融变体名称（用于日志/文件名）。"""
        ch = self.backbone.cbam1
        if isinstance(ch, CBAM):
            return f"DeepLabV3Plus_{ch.variant_name}"
        if isinstance(ch, ECA):
            return "DeepLabV3Plus_ECA"
        if isinstance(ch, CoordinateAttention):
            return "DeepLabV3Plus_CoordAtt"
        return "DeepLabV3Plus_no_attention"

    def param_groups(self, lr: float, backbone_lr_scale: float = 0.1) -> list[dict]:
        """返回可直接传给 optimizer 的参数组（骨干 lr 降低）。"""
        backbone_params = list(self.backbone.parameters())
        head_params = (
            list(self.aspp.parameters()) +
            list(self.decoder.parameters())
        )
        return [
            {"params": backbone_params, "lr": lr * backbone_lr_scale},
            {"params": head_params,     "lr": lr},
        ]


def deeplab_from_train_args(
    train_args: dict,
    *,
    num_classes: int = 11,
    pretrained_backbone: bool = False,
) -> DeepLabV3PlusCBAM:
    """根据 train.py 写入 checkpoint 的 args 字典重建模型（兼容无 attention 字段的旧 ckpt）。"""
    attention = train_args.get("attention", "cbam")
    cbam_variant = train_args.get("cbam", "full")
    use_ch = cbam_variant in ("full", "channel")
    use_sp = cbam_variant in ("full", "spatial")
    return DeepLabV3PlusCBAM(
        num_classes=num_classes,
        attention_type=attention,
        cbam_use_channel=use_ch,
        cbam_use_spatial=use_sp,
        pretrained_backbone=pretrained_backbone,
    )


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== DeepLabV3PlusCBAM 自测 (device={device}) ===")

    x = torch.randn(2, 3, 360, 480).to(device)

    variants = [
        ("channel_only", dict(attention_type="cbam", cbam_use_channel=True,  cbam_use_spatial=False)),
        ("spatial_only", dict(attention_type="cbam", cbam_use_channel=False, cbam_use_spatial=True)),
        ("full_cbam",    dict(attention_type="cbam", cbam_use_channel=True,  cbam_use_spatial=True)),
        ("no_cbam",      dict(attention_type="cbam", cbam_use_channel=False, cbam_use_spatial=False)),
        ("eca",          dict(attention_type="eca")),
        ("coord",        dict(attention_type="coord")),
        ("none",         dict(attention_type="none")),
    ]

    for tag, kwargs in variants:
        m = DeepLabV3PlusCBAM(num_classes=11, pretrained_backbone=False, **kwargs).to(device)
        m.eval()
        with torch.no_grad():
            out = m(x)
        assert out.shape == (2, 11, 360, 480), f"输出形状错误: {out.shape}"
        total = sum(p.numel() for p in m.parameters()) / 1e6
        cbam_p = sum(
            p.numel() for mod in [m.backbone.cbam1, m.backbone.cbam2,
                                   m.backbone.cbam3, m.backbone.cbam4]
            for p in mod.parameters()
        )
        print(f"  [{tag:<14}]  {m.variant_name:<32} 输出: {tuple(out.shape)}  "
              f"总参数: {total:.1f}M  Stage 注意力参数: {cbam_p:,}")

    print("=== 自测通过 ===")
