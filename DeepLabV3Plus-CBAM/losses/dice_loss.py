"""
@author: ThengAndrew
@created_at: 2026-5-3
@updated_at: 2026-5-3
@usage:
    dice = DiceLoss(ignore_index=11)
    ce   = WeightedCrossEntropyLoss(class_weights=w, ignore_index=11)
    loss = 0.5 * ce(logits, targets) + 0.5 * dice(logits, targets)
@description: 多类 Soft Dice Loss（阶段二改进 B 方案二）。
    对类别面积不平衡不敏感，与 CE 联合使用可增强小类边界约束。
    实现参考：V-Net (Milletari 2016)，扩展至多类分割。
@warning:
    - logits 须为未经 softmax 的原始输出 [B, C, H, W]；
    - ignore_index 像素在 softmax 概率计算后通过 mask 置零，
      不参与分子/分母累加。
@note:
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """多类 Soft Dice Loss。

    Parameters
    ----------
    smooth
        分子/分母加的平滑系数，防止零除（默认 1e-6）。
    ignore_index
        void 像素 id，不参与 Dice 计算（默认 11）。
    reduction
        "mean"：对各有效类别的 Dice Loss 取平均；
        "sum" ：直接求和。
    """

    def __init__(
        self,
        smooth: float = 1e-6,
        ignore_index: int = 11,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.smooth       = smooth
        self.ignore_index = ignore_index
        self.reduction    = reduction

    def forward(
        self,
        logits:  torch.Tensor,   # [B, C, H, W]
        targets: torch.Tensor,   # [B, H, W]  long
    ) -> torch.Tensor:
        B, C, H, W = logits.shape

        # softmax 概率  [B, C, H, W]
        probs = F.softmax(logits, dim=1)

        # 构建 void mask  [B, 1, H, W]  →  [B, C, H, W]
        void_mask = (targets == self.ignore_index)                 # [B, H, W] bool
        void_mask = void_mask.unsqueeze(1).expand_as(probs)       # [B, C, H, W]
        probs = probs.masked_fill(void_mask, 0.0)

        # one-hot 编码（void 位置保持全 0，不计入任何类）
        # 先把 void 临时映射到 0（后续通过 void_mask 置零）
        targets_safe = targets.clone()
        targets_safe[targets == self.ignore_index] = 0
        one_hot = torch.zeros(B, C, H, W, device=logits.device, dtype=probs.dtype)
        one_hot.scatter_(1, targets_safe.unsqueeze(1), 1.0)
        one_hot = one_hot.masked_fill(void_mask, 0.0)             # void→全 0

        # 逐类 Dice：沿 B, H, W 维度求和  →  [C]
        inter   = (probs * one_hot).sum(dim=(0, 2, 3))
        denom   = (probs + one_hot).sum(dim=(0, 2, 3))
        dice_per_class = 1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)

        # 统计各类是否有真值，仅对有真值的类求平均
        has_gt = one_hot.sum(dim=(0, 2, 3)) > 0               # [C] bool
        if has_gt.any():
            loss = dice_per_class[has_gt]
            return loss.mean() if self.reduction == "mean" else loss.sum()
        # 极端情况：无有效像素（整批全是 void）
        return dice_per_class.mean() * 0.0
