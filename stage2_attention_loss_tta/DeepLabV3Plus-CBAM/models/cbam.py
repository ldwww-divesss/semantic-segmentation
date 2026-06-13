"""
@author: ThengAndrew
@created_at: 2026-5-3
@updated_at: 2026-5-3
@usage:
    # 完整 CBAM（消融实验 iii）
    cbam = CBAM(channels=256)
    # 仅通道注意力（消融实验 i）
    cbam = CBAM(channels=256, use_spatial=False)
    # 仅空间注意力（消融实验 ii）
    cbam = CBAM(channels=256, use_channel=False)

    out = cbam(feat)  # feat: [B, C, H, W]
@description: CBAM 通道注意力 + 空间注意力模块（阶段二改进 A）。
    通道注意力对「什么特征重要」建模；空间注意力对「哪个位置重要」建模。
    use_channel / use_spatial 两个开关直接支持三组消融实验，无需额外代码。
    实现完全按照原论文：Woo et al., ECCV 2018.
@warning:
    - reduction 须能整除 channels，否则自动 clamp 到最小值 1；
    - 空间注意力卷积核固定 7×7（论文推荐值），padding=3 保持尺寸不变；
    - 不建议对极小特征图（H/W < 7）使用，否则感受野超出图像边界。
@note:
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 通道注意力
# ============================================================

class ChannelAttention(nn.Module):
    """CBAM 通道注意力分支。

    结构：GlobalAvgPool + GlobalMaxPool → 共享 MLP → 逐元素相加 → Sigmoid。

    Parameters
    ----------
    channels
        输入特征图通道数。
    reduction
        MLP 瓶颈压缩比（默认 16）。
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # 全局平均池化 & 最大池化  →  [B, C]
        avg = x.view(B, C, -1).mean(dim=2)
        mx  = x.view(B, C, -1).max(dim=2).values
        # 共享 MLP
        scale = torch.sigmoid(self.mlp(avg) + self.mlp(mx))   # [B, C]
        return x * scale.view(B, C, 1, 1)


# ============================================================
# 空间注意力
# ============================================================

class SpatialAttention(nn.Module):
    """CBAM 空间注意力分支。

    结构：沿通道维 AvgPool + MaxPool（各 [B,1,H,W]）→ 拼接 → Conv7×7 → Sigmoid。

    Parameters
    ----------
    kernel_size
        空间注意力卷积核大小，论文推荐 7；小特征图可改为 3。
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        avg = x.mean(dim=1, keepdim=True)              # [B, 1, H, W]
        mx  = x.max(dim=1, keepdim=True).values        # [B, 1, H, W]
        feat = torch.cat([avg, mx], dim=1)              # [B, 2, H, W]
        scale = torch.sigmoid(self.conv(feat))          # [B, 1, H, W]
        return x * scale


# ============================================================
# CBAM 主模块（含消融开关）
# ============================================================

class CBAM(nn.Module):
    """Convolutional Block Attention Module（Woo et al., ECCV 2018）。

    Parameters
    ----------
    channels
        输入特征图的通道数 C。
    reduction
        通道注意力 MLP 压缩比（默认 16）。
    spatial_kernel
        空间注意力卷积核大小（默认 7）。
    use_channel
        是否使用通道注意力分支（消融开关）。
    use_spatial
        是否使用空间注意力分支（消融开关）。

    消融实验对应设置
    ----------------
    +--------------------------------------+--------------+--------------+
    | 实验变体                             | use_channel  | use_spatial  |
    +======================================+==============+==============+
    | (i)  仅通道注意力                    | True         | False        |
    +--------------------------------------+--------------+--------------+
    | (ii) 仅空间注意力                    | False        | True         |
    +--------------------------------------+--------------+--------------+
    | (iii) 完整 CBAM（论文原版）          | True         | True         |
    +--------------------------------------+--------------+--------------+
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel: int = 7,
        *,
        use_channel: bool = True,
        use_spatial: bool = True,
    ) -> None:
        super().__init__()
        if not use_channel and not use_spatial:
            raise ValueError("use_channel 与 use_spatial 不能同时为 False（等价于无注意力）")
        self.use_channel = use_channel
        self.use_spatial = use_spatial
        self.channel_att = ChannelAttention(channels, reduction) if use_channel else None
        self.spatial_att = SpatialAttention(spatial_kernel)      if use_spatial else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_att is not None:
            x = self.channel_att(x)
        if self.spatial_att is not None:
            x = self.spatial_att(x)
        return x

    @property
    def variant_name(self) -> str:
        """返回当前消融变体的可读名称（用于日志/报告）。"""
        if self.use_channel and self.use_spatial:
            return "CBAM_full"
        if self.use_channel:
            return "CBAM_channel_only"
        return "CBAM_spatial_only"


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    print("=== CBAM 自测 ===")
    x = torch.randn(2, 256, 45, 60)

    for use_ch, use_sp in [(True, False), (False, True), (True, True)]:
        m = CBAM(256, use_channel=use_ch, use_spatial=use_sp)
        out = m(x)
        assert out.shape == x.shape, f"形状不匹配: {out.shape}"
        params = sum(p.numel() for p in m.parameters())
        print(f"  {m.variant_name:<22}  输出: {tuple(out.shape)}  参数量: {params:,}")

    # 参数量验证（完整 CBAM，C=256，reduction=16）
    # ChannelAttention: (256*16 + 16*256)*2 = 8192 → 实际 Linear 含 bias=False → 256/16=16, 16→256
    #   Linear1: 256*16=4096, Linear2: 16*256=4096 → 共享两次 = (4096+4096)=8192
    # SpatialAttention: 2*1*7*7=98
    # 合计 8192+98=8290
    full = CBAM(256)
    n = sum(p.numel() for p in full.parameters())
    print(f"\n  完整 CBAM(256) 参数量: {n:,}  (期望 8290)")
    assert n == 8290, f"参数量异常: {n}"
    print("=== 自测通过 ===")
