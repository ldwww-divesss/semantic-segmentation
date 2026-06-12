"""
@author: ThengAndrew
@created_at: 2026-5-9
@updated_at: 2026-5-9
@usage:
    from models.coordinate_attention import CoordinateAttention

    m = CoordinateAttention(channels=256)
    out = m(feat)  # feat: [B, C, H, W]
@description: Coordinate Attention（Hou et al., CVPR 2021）：沿 H、W 分别编码位置统计，
    融合后经 split 得到高度与宽度注意力图，与输入逐元素相乘。
@warning:
    - reduction 与 mip 下限 8 与常见开源实现一致；
    - 高分辨率特征图上计算量随 H+W 线性增长。
@note:
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CoordinateAttention(nn.Module):
    """Coordinate Attention：Encodes vertical + horizontal positional statistics."""

    def __init__(self, channels: int, reduction: int = 32) -> None:
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        _, _, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h_out, x_w_out = torch.split(y, [h, w], dim=2)
        x_w_out = x_w_out.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h_out))
        a_w = torch.sigmoid(self.conv_w(x_w_out))
        return identity * a_h * a_w

    @property
    def variant_name(self) -> str:
        return "CoordAtt"


if __name__ == "__main__":
    print("=== CoordinateAttention 自测 ===")
    x = torch.randn(2, 256, 45, 60)
    m = CoordinateAttention(256)
    y = m(x)
    assert y.shape == x.shape
    n = sum(p.numel() for p in m.parameters())
    print(f"  CoordAtt(256) 输出 {tuple(y.shape)}  参数量 {n:,}")
    print("=== 自测通过 ===")
