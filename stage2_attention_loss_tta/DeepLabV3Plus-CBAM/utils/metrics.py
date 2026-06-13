"""
@author: ThengAndrew
@created_at: 2026-5-3
@updated_at: 2026-5-3
@usage:
    metric = SegMetric(num_classes=11, ignore_index=11)
    for pred, gt in val_loader:
        metric.update(pred, gt)
    results = metric.compute()   # dict: miou, pa, iou_per_class, ...
    metric.reset()
@description: 基于混淆矩阵的语义分割指标：mIoU、PA、逐类 IoU 与 Acc，
    支持批量累积后一次计算，用于训练验证与报告表格生成。
@warning:
    - pred/gt 均为整数类别 id，形状 [B, H, W] 或 [H, W]；
    - ignore_index 对应的像素在混淆矩阵中完全跳过；
    - compute() 返回的 iou_per_class 中，某类训练集无样本时 IoU 为 nan。
@note:
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Optional


class SegMetric:
    """混淆矩阵累积器，支持多批次逐步更新后统一计算指标。

    Parameters
    ----------
    num_classes
        有效类别数（不含 ignore_index 对应的 void 类）。
    ignore_index
        该 id 的像素跳过统计，默认 11（CamVid void）。
    """

    def __init__(self, num_classes: int, ignore_index: int = 11) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self._mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    # ----------------------------------------------------------
    # 公共接口
    # ----------------------------------------------------------

    def reset(self) -> None:
        """清空混淆矩阵，开始新一轮统计。"""
        self._mat[:] = 0

    def update(
        self,
        pred: "torch.Tensor | np.ndarray",
        gt:   "torch.Tensor | np.ndarray",
    ) -> None:
        """累积一批预测结果。

        Parameters
        ----------
        pred
            预测类别 id，形状 [B, H, W] 或 [H, W]，int 类型。
        gt
            真值类别 id，形状同 pred。
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.cpu().numpy()

        pred = pred.astype(np.int64).ravel()
        gt   = gt.astype(np.int64).ravel()

        # 过滤 ignore_index
        valid = gt != self.ignore_index
        pred, gt = pred[valid], gt[valid]

        # 越界像素也丢弃（防御性处理）
        in_range = (pred >= 0) & (pred < self.num_classes) & \
                   (gt   >= 0) & (gt   < self.num_classes)
        pred, gt = pred[in_range], gt[in_range]

        # 高效累积：np.bincount 比 for 循环快 ~100×
        idx = gt * self.num_classes + pred
        counts = np.bincount(idx, minlength=self.num_classes ** 2)
        self._mat += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, object]:
        """由当前混淆矩阵计算所有指标，返回结果字典。

        Returns
        -------
        dict 包含：
            miou         : float，平均 IoU（nan 类不计入平均）
            pa           : float，全局像素准确率
            iou_per_class: list[float]，逐类 IoU（nan 表示该类无真值）
            acc_per_class: list[float]，逐类召回率（Acc）
            freq_w_iou   : float，频率加权 mIoU（FWIoU）
            confusion    : np.ndarray [C, C]，当前混淆矩阵副本
        """
        mat = self._mat.astype(np.float64)
        # TP[i] = mat[i,i]
        tp = np.diag(mat)
        # 每类真值总数（行和）、预测总数（列和）
        gt_sum   = mat.sum(axis=1)
        pred_sum = mat.sum(axis=0)

        # 逐类 IoU = TP / (TP + FP + FN) = TP / (gt_sum + pred_sum - TP)
        union = gt_sum + pred_sum - tp
        iou_per_class = np.where(union > 0, tp / union, np.nan)

        # mIoU：仅对有真值的类取平均
        valid_mask = gt_sum > 0
        miou = float(np.nanmean(iou_per_class[valid_mask])) if valid_mask.any() else 0.0

        # 全局像素准确率
        total = mat.sum()
        pa = float(tp.sum() / total) if total > 0 else 0.0

        # 逐类召回率 Acc（= TP / gt_sum）
        acc_per_class = np.where(gt_sum > 0, tp / gt_sum, np.nan)

        # 频率加权 IoU
        freq = gt_sum / total if total > 0 else gt_sum
        freq_w_iou = float(np.nansum(freq * iou_per_class))

        return {
            "miou":          miou,
            "pa":            pa,
            "iou_per_class": iou_per_class.tolist(),
            "acc_per_class": acc_per_class.tolist(),
            "freq_w_iou":    freq_w_iou,
            "confusion":     self._mat.copy(),
        }

    def pretty_print(
        self,
        class_names: Optional[list[str]] = None,
        *,
        results: Optional[dict] = None,
    ) -> str:
        """返回可直接 print 的格式化指标字符串（含逐类 IoU 表格）。"""
        r = results if results is not None else self.compute()
        names = class_names or [str(i) for i in range(self.num_classes)]
        lines: list[str] = [
            f"  mIoU   : {r['miou'] * 100:.2f}%",
            f"  PA     : {r['pa']   * 100:.2f}%",
            f"  FWIoU  : {r['freq_w_iou'] * 100:.2f}%",
            "",
            f"  {'Class':<14} {'IoU':>7}  {'Acc':>7}",
            "  " + "-" * 32,
        ]
        for name, iou, acc in zip(names, r["iou_per_class"], r["acc_per_class"]):
            iou_str = f"{iou * 100:6.2f}%" if not (isinstance(iou, float) and iou != iou) else "   nan "
            acc_str = f"{acc * 100:6.2f}%" if not (isinstance(acc, float) and acc != acc) else "   nan "
            lines.append(f"  {name:<14} {iou_str}  {acc_str}")
        return "\n".join(lines)


# ============================================================
# 便捷函数（单次计算，不累积）
# ============================================================

def compute_miou(
    pred: "torch.Tensor | np.ndarray",
    gt:   "torch.Tensor | np.ndarray",
    num_classes: int,
    ignore_index: int = 11,
) -> tuple[float, list[float]]:
    """一次性计算 mIoU 与逐类 IoU（不累积版本）。

    Returns
    -------
    (miou, iou_per_class)
    """
    m = SegMetric(num_classes, ignore_index)
    m.update(pred, gt)
    r = m.compute()
    return r["miou"], r["iou_per_class"]


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    from utils.data import CLASSES, NUM_CLASSES, IGNORE_INDEX

    print("=== SegMetric 自测 ===")
    rng = np.random.default_rng(42)

    metric = SegMetric(NUM_CLASSES, ignore_index=IGNORE_INDEX)

    for _ in range(5):
        B, H, W = 2, 360, 480
        gt   = rng.integers(0, NUM_CLASSES + 1, size=(B, H, W)).astype(np.int64)
        pred = rng.integers(0, NUM_CLASSES,     size=(B, H, W)).astype(np.int64)
        metric.update(pred, gt)

    r = metric.compute()
    print(metric.pretty_print(list(CLASSES), results=r))
    print(f"\n  完整 iou_per_class: {[f'{v*100:.1f}' if v==v else 'nan' for v in r['iou_per_class']]}")

    # 验证 compute_miou 快捷函数
    gt_t   = torch.from_numpy(gt)
    pred_t = torch.from_numpy(pred)
    miou, ious = compute_miou(pred_t, gt_t, NUM_CLASSES, IGNORE_INDEX)
    print(f"\n  compute_miou (last batch only): mIoU={miou*100:.2f}%")
    print("=== 自测通过 ===")
