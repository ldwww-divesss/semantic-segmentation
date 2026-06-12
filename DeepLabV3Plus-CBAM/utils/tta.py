"""
@author: ThengAndrew
@created_at: 2026-5-3
@updated_at: 2026-5-3
@usage:
    from utils.tta import SegmentationTTA
    tta = SegmentationTTA(model, scales=(0.75, 1.0, 1.25), flip=True)
    probs = tta(image_tensor)          # [1, C, H, W] softmax 均值
    preds = probs.argmax(dim=1)        # [1, H, W]
@description: 测试时增强（TTA）：对每张图像执行多尺度推理和水平翻转，
    将所有变体的 softmax 概率平均后 argmax，不修改模型权重。
@warning:
    - 输入 x 须为已归一化的 FloatTensor，形状 [1, 3, H, W]（batch=1）；
    - scales 中含 1.0 代表保留原始分辨率；
    - flip=True 时增强数量翻倍（每个尺度各做一次翻转），推理时间同步倍增；
    - 不支持 batch>1，请在 DataLoader val_batch_size=1 下使用。
@note:
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationTTA:
    """分割测试时增强（TTA）封装。

    对输入图像在多个尺度下前向推理，并可选做水平翻转，
    将所有变体的 softmax 概率平均后返回。

    Parameters
    ----------
    model
        已加载权重、处于 eval() 模式的分割模型。
        要求 model(x) 返回 logits，形状 [B, C, H, W]。
    scales
        推理尺度列表（相对原图宽高的缩放比）。
        建议 (0.75, 1.0, 1.25)；仅用 (1.0,) 等同于无多尺度。
    flip
        是否在每个尺度额外做一次水平翻转推理。
    """

    def __init__(
        self,
        model: nn.Module,
        scales: tuple[float, ...] = (0.75, 1.0, 1.25),
        flip: bool = True,
    ) -> None:
        self.model = model
        self.scales = scales
        self.flip = flip

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """执行 TTA 推理，返回概率均值。

        Parameters
        ----------
        x
            归一化后的图像张量，形状 [1, 3, H, W]。

        Returns
        -------
        torch.Tensor
            各增强变体 softmax 概率的均值，形状 [1, C, H, W]。
        """
        if x.dim() != 4 or x.shape[0] != 1:
            raise ValueError(
                f"SegmentationTTA 要求 batch=1 输入，实际 shape={tuple(x.shape)}"
            )

        _, _, H, W = x.shape
        probs_sum: torch.Tensor | None = None
        n_aug = 0

        for scale in self.scales:
            x_in = self._scale(x, scale, H, W)

            # 原图推理
            probs_sum, n_aug = self._accumulate(
                x_in, H, W, probs_sum, n_aug, scale
            )

            # 水平翻转推理
            if self.flip:
                x_flip = torch.flip(x_in, dims=[3])
                probs_sum, n_aug = self._accumulate(
                    x_flip, H, W, probs_sum, n_aug, scale, hflip=True
                )

        assert probs_sum is not None
        return probs_sum / n_aug

    # ------------------------------------------------------------------
    # 私有辅助
    # ------------------------------------------------------------------

    def _scale(
        self, x: torch.Tensor, scale: float, H: int, W: int
    ) -> torch.Tensor:
        """将输入双线性缩放到目标尺寸。scale=1.0 时直接返回原张量。"""
        if scale == 1.0:
            return x
        new_h = max(1, int(round(H * scale)))
        new_w = max(1, int(round(W * scale)))
        return F.interpolate(
            x, size=(new_h, new_w), mode="bilinear", align_corners=False
        )

    def _accumulate(
        self,
        x_in: torch.Tensor,
        orig_H: int,
        orig_W: int,
        probs_sum: torch.Tensor | None,
        n_aug: int,
        scale: float,
        hflip: bool = False,
    ) -> tuple[torch.Tensor, int]:
        """前向一次，概率图 resize 回原尺寸后累加。"""
        logits = self.model(x_in)

        if scale != 1.0:
            logits = F.interpolate(
                logits, size=(orig_H, orig_W), mode="bilinear", align_corners=False
            )

        if hflip:
            logits = torch.flip(logits, dims=[3])

        probs = torch.softmax(logits, dim=1)

        if probs_sum is None:
            probs_sum = probs
        else:
            probs_sum = probs_sum + probs

        return probs_sum, n_aug + 1
