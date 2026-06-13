import numpy as np
import torch

from dataset import CAMVID_CLASSES, NUM_CLASSES, IGNORE_INDEX


class SegmentationMetrics:
    """Accumulates predictions via confusion matrix; reports mIoU and Pixel Accuracy."""

    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX):
        self.num_classes   = num_classes
        self.ignore_index  = ignore_index
        self.cm            = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.cm[:] = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """pred, target: (B, H, W) or (H, W) int64 tensors on any device."""
        pred   = pred.cpu().numpy().flatten().astype(np.int64)
        target = target.cpu().numpy().flatten().astype(np.int64)

        mask   = target != self.ignore_index
        pred   = np.clip(pred[mask], 0, self.num_classes - 1)
        target = target[mask]

        flat   = self.num_classes * target + pred
        self.cm += np.bincount(flat, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes
        )

    def iou_per_class(self) -> np.ndarray:
        tp    = np.diag(self.cm)
        fp    = self.cm.sum(axis=0) - tp
        fn    = self.cm.sum(axis=1) - tp
        union = tp + fp + fn
        return np.where(union > 0, tp / union, np.nan)

    def miou(self) -> float:
        return float(np.nanmean(self.iou_per_class()))

    def pixel_accuracy(self) -> float:
        total = self.cm.sum()
        return float(np.diag(self.cm).sum() / total) if total > 0 else 0.0

    def print_report(self):
        iou_arr = self.iou_per_class()
        print(f'  {"Class":<12} {"IoU":>7}')
        print('  ' + '-' * 21)
        for name, iou in zip(CAMVID_CLASSES, iou_arr):
            iou_str = f'{iou*100:6.2f}%' if not np.isnan(iou) else '   N/A '
            print(f'  {name:<12} {iou_str}')
        print('  ' + '-' * 21)
        print(f'  {"mIoU":<12} {self.miou()*100:6.2f}%')
        print(f'  {"Pixel Acc":<12} {self.pixel_accuracy()*100:6.2f}%')
