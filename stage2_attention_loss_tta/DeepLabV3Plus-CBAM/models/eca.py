"""
@author: ThengAndrew
@created_at: 2026-5-9
@updated_at: 2026-5-9
@usage:
    from models.eca import ECA

    m = ECA(channels=256)
    out = m(feat)  # feat: [B, C, H, W]
@description: Efficient Channel Attention（Wang et al., CVPR 2020）：GAP 后经通道维 1D 卷积
    建模局部跨通道交互，kernel 尺寸随通道数自适应，无降维瓶颈。
@warning:
    - 仅通道方向重标定，无独立空间分支；
    - k 为奇数且 ≥3，极小通道数时行为与论文表一致。
@note:
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _eca_kernel_size(channels: int, gamma: int = 2, b: int = 1) -> int:
    """论文 Table 3：k = ψ(C)，取最近奇数且不小于 3。"""
    t = int(abs(math.log2(max(channels, 2)) / gamma + b / gamma))
    k = t if t % 2 == 1 else t + 1
    return max(k, 3)


class ECA(nn.Module):
    """Efficient Channel Attention：1×1 GAP → Conv1d(k) → Sigmoid，输出与输入同形。"""

    def __init__(self, channels: int, *, gamma: int = 2, b: int = 1) -> None:
        super().__init__()
        self.channels = channels
        k = _eca_kernel_size(channels, gamma=gamma, b=b)
        padding = (k - 1) // 2
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        y = x.mean(dim=(2, 3), keepdim=True)
        # [B, C, 1, 1] → [B, 1, C] for Conv1d on last dim
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        scale = torch.sigmoid(y)
        return x * scale

    @property
    def variant_name(self) -> str:
        return "ECA"


if __name__ == "__main__":
    print("=== ECA 自测 ===")
    x = torch.randn(2, 256, 45, 60)
    m = ECA(256)
    y = m(x)
    assert y.shape == x.shape
    n = sum(p.numel() for p in m.parameters())
    print(f"  ECA(256) 输出 {tuple(y.shape)}  参数量 {n:,}")
    print("=== 自测通过 ===")
