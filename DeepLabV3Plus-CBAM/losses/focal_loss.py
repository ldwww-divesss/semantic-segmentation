"""
@author: ThengAndrew
@created_at: 2026-5-3
@updated_at: 2026-5-3
@usage:
    alpha = train_ds.compute_class_weights("inv_freq").to(device)
    loss_fn = FocalLoss(gamma=2.0, alpha=alpha, ignore_index=11)
    loss = loss_fn(logits, targets)   # logits:[B,C,H,W]  targets:[B,H,W]
@description: 多类 Focal Loss（阶段二改进 B 方案三）。
    gamma=2 对难样本自动加权，alpha 按类频初始化，
    抑制天空/道路主导梯度，提升小类（Pole, Bicyclist）学习效果。
@warning:
    - alpha 含义为逐类权重 [C]，与原论文二分类 alpha 略有差异；
    - ignore_index 像素在 CE 后被掩掉，不贡献 loss；
    - gamma=0 时退化为加权 CE，可用来对齐验证。
@note:
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """多类 Focal Loss。

    Parameters
    ----------
    gamma
        聚焦参数，越大对易分样本惩罚越强（论文推荐 2.0）。
    alpha
        逐类权重 Tensor [C]；None 时各类权重相等；
        通常由 CamVidDataset.compute_class_weights() 提供。
    ignore_index
        void 像素 id（默认 11）。
    reduction
        "mean" 或 "sum"。
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        ignore_index: int = 11,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma       = gamma
        self.ignore_index = ignore_index
        self.reduction   = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha: Optional[torch.Tensor] = None

    def forward(
        self,
        logits:  torch.Tensor,   # [B, C, H, W]
        targets: torch.Tensor,   # [B, H, W]  long
    ) -> torch.Tensor:
        B, C, H, W = logits.shape

        # ---- 逐像素 log-softmax  →  CE（reduction=none）----
        log_p = F.log_softmax(logits, dim=1)              # [B, C, H, W]

        # 为 nll_loss 准备：targets 中 void 先替换为 0，后续 mask 掉
        targets_safe = targets.clone()
        void_mask    = targets == self.ignore_index        # [B, H, W] bool
        targets_safe[void_mask] = 0

        # ce_loss [B, H, W]（每像素、无约减）
        ce_loss = F.nll_loss(
            log_p,
            targets_safe,
            weight=self.alpha,        # None 或 [C]
            ignore_index=-1,          # 不在这里 ignore，手动 mask
            reduction="none",
        )

        # ---- focal 调制因子  (1 - p_t)^gamma ----
        # 取目标类的概率 p_t  [B, H, W]
        p = torch.exp(log_p)
        p_t = p.gather(1, targets_safe.unsqueeze(1)).squeeze(1)  # [B, H, W]
        focal_weight = (1.0 - p_t).pow(self.gamma)

        loss = focal_weight * ce_loss                      # [B, H, W]

        # ---- 掩掉 void 像素 ----
        loss = loss.masked_fill(void_mask, 0.0)

        valid_count = (~void_mask).sum().clamp(min=1)
        if self.reduction == "mean":
            return loss.sum() / valid_count
        return loss.sum()
