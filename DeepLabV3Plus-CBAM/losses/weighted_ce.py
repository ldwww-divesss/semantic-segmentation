"""
@author: ThengAndrew
@created_at: 2026-5-3
@updated_at: 2026-5-3
@usage:
    weights = train_ds.compute_class_weights("inv_freq").to(device)
    loss_fn = WeightedCrossEntropyLoss(class_weights=weights, ignore_index=11)
    loss = loss_fn(logits, targets)   # logits:[B,C,H,W]  targets:[B,H,W]
@description: 按类频加权的交叉熵（阶段二改进 B 方案一）。
    权重由 CamVidDataset.compute_class_weights() 预先计算，
    稀有类别（Bicyclist ~1%）获得更高梯度权重。
@warning:
    - class_weights 需在调用前移至与 logits 相同的 device；
    - ignore_index 须与标注 void id 一致（默认 11）。
@note:
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WeightedCrossEntropyLoss(nn.Module):
    """按类别频率加权的 Cross-Entropy Loss。

    Parameters
    ----------
    class_weights
        形状 [num_classes] 的 float Tensor；None 时退化为标准 CE。
    ignore_index
        该 id 对应的像素不参与 loss 计算（CamVid void=11）。
    label_smoothing
        标签平滑系数（0.0 表示不使用），可在训练后期辅助防过拟合。
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = 11,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.ignore_index    = ignore_index
        self.label_smoothing = label_smoothing
        # 注册为 buffer，使其随模型 .to(device) 一起移动
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        logits:  torch.Tensor,   # [B, C, H, W]
        targets: torch.Tensor,   # [B, H, W]  long
    ) -> torch.Tensor:
        weight = self.class_weights  # None 或 [C] Tensor（已在正确 device 上）
        return F.cross_entropy(
            logits,
            targets,
            weight=weight,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction="mean",
        )
