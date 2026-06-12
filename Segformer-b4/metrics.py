import numpy as np


def compute_metrics(pred, mask, num_classes=12):
    """计算 mIoU (排除 Void=11), PA, 及各类别 IoU/Acc

    Args:
        pred:  [H, W] numpy 预测类别索引
        mask:  [H, W] numpy 真实标签
        num_classes: 类别总数 (含 Void)

    Returns:
        mIoU (float), PA (float), class_metrics (dict)
    """
    ious = []
    accs = []
    class_metrics = {}

    total_correct = 0
    total_pixels = 0

    for cls in range(num_classes):
        pred_i = (pred == cls)
        mask_i = (mask == cls)

        inter = (pred_i & mask_i).sum()
        union = (pred_i | mask_i).sum()
        cls_pixels = mask_i.sum()

        iou = inter / union if union > 0 else np.nan
        acc = inter / cls_pixels if cls_pixels > 0 else np.nan

        class_metrics[cls] = {"iou": iou, "acc": acc}
        ious.append(iou)
        accs.append(acc)

        total_correct += inter
        total_pixels += cls_pixels

    # mIoU over valid clases only (exclude Void=11)
    valid_ious = [iou for i, iou in enumerate(ious) if i != 11 and not np.isnan(iou)]
    mIoU = np.mean(valid_ious) if valid_ious else 0.0

    # PA over all valid pixels (exclude Void)
    PA = total_correct / total_pixels if total_pixels > 0 else 0.0

    return mIoU, PA, class_metrics
