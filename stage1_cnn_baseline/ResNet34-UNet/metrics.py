import numpy as np

def compute_metrics(pred, mask, num_classes=14):
    """
    计算 mIoU, PA 以及每个类别的 IoU 和 Acc
    """
    ious = []
    accs = []
    
    # 记录每个类的详细数据字典
    class_metrics = {}

    # 整体混淆矩阵元素
    total_inter = 0
    total_union = 0
    total_correct = 0
    total_pixels = 0

    for cls in range(num_classes):
        pred_i = (pred == cls)
        mask_i = (mask == cls)

        inter = (pred_i & mask_i).sum()
        union = (pred_i | mask_i).sum()
        
        # 类别准确率：预测正确且属于该类的像素 / 该类的真实像素总数
        cls_pixels = mask_i.sum()
        cls_correct = inter 

        if union > 0:
            iou = inter / union
            ious.append(iou)
        else:
            iou = np.nan

        if cls_pixels > 0:
            acc = cls_correct / cls_pixels
            accs.append(acc)
        else:
            acc = np.nan
            
        class_metrics[cls] = {'iou': iou, 'acc': acc}
        
        total_correct += inter
        total_pixels += cls_pixels

    mIoU = np.nanmean(ious)
    PA = total_correct / total_pixels if total_pixels > 0 else 0

    return mIoU, PA, class_metrics