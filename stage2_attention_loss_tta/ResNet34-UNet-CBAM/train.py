import os
import math
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import SegDataset
from model import ResNetUNet


# =========================================================
# 1. 基础配置
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 12

EPOCHS = 100

BATCH_SIZE = 4

BASE_LR = 1e-4
CBAM_LR = 5e-4
WARMUP_EPOCHS = 5

CLASS_NAMES = [
    "Sky",
    "Building",
    "Pole",
    "Road",
    "Pavement",
    "Tree",
    "SignSymbol",
    "Fence",
    "Car",
    "Pedestrian",
    "Bicyclist",
    "Void"
]

# =========================================================
# 2. 数据集
# =========================================================
print("Loading datasets...")

train_dataset = SegDataset(
    "/mnt/workspace/semantic_segmentation/语义分割",
    train=True
)

val_dataset = SegDataset(
    "/mnt/workspace/semantic_segmentation/语义分割",
    train=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# =========================================================
# 3. 计算类别权重（用于损失函数）
# =========================================================
def compute_class_weights(dataset, num_classes, ignore_index=11):
    """统计训练集中每个类的像素数，返回权重张量"""
    counts = torch.zeros(num_classes, dtype=torch.float64)
    print("Computing class weights from training set...")
    for i in tqdm(range(len(dataset)), desc="Class Weights"):
        _, mask = dataset[i]
        mask = mask.view(-1)
        counts += torch.bincount(mask, minlength=num_classes).float()
    # 忽略 Void
    counts[ignore_index] = 0.0
    total = counts.sum()
    # 权重 = 总像素数 / (类别数 * 该类像素数)
    weights = total / (num_classes * counts.clamp(min=1))
    weights[ignore_index] = 0.0
    print("Computed class weights:", weights)
    return weights.float()

class_weights = compute_class_weights(train_dataset, NUM_CLASSES, ignore_index=11)
class_weights = class_weights.to(device)

# =========================================================
# 4. 模型
# =========================================================
model = ResNetUNet(num_classes=NUM_CLASSES).to(device)
print("\nModel Loaded: ResNet34-UNet-CBAM")

# =========================================================
# 5. 优化器（分层学习率）
# =========================================================
cbam_params = []
other_params = []
for name, param in model.named_parameters():
    if 'cbam' in name:
        cbam_params.append(param)
    else:
        other_params.append(param)

optimizer = torch.optim.AdamW([
    {'params': other_params, 'lr': BASE_LR},
    {'params': cbam_params, 'lr': CBAM_LR}
], weight_decay=1e-4)

print(f"Optimizer: AdamW | backbone lr={BASE_LR}, cbam lr={CBAM_LR}")

# =========================================================
# 6. 学习率调度器（Warmup + CosineAnnealing）
# =========================================================
def lr_lambda(epoch):
    """前 WARMUP_EPOCHS 线性增长，之后余弦衰减"""
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    else:
        progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scaler = torch.amp.GradScaler('cuda')

# =========================================================
# 7. Loss（加权）
# =========================================================
# 交叉熵（带类别权重）
criterion_ce = torch.nn.CrossEntropyLoss(
    weight=class_weights,
    ignore_index=11
)

def dice_loss(pred, target, num_classes, class_weights):
    """
    加权 Dice Loss，class_weights 与 device 一致
    """
    pred = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

    dice_per_class = (2.0 * intersection + 1e-5) / (union + 1e-5)  # [B, C]
    dice_mean_per_class = dice_per_class.mean(dim=0)                # [C]

    # 加权平均，权重中 Void 为 0，不影响
    weighted_dice = (dice_mean_per_class * class_weights).sum() / class_weights.sum()
    return 1.0 - weighted_dice


# =========================================================
# 8. 训练开始
# =========================================================
best_miou = 0.0

print("\n========================================")
print(" Stage 2 : ResNet34-UNet + CBAM (Weighted Loss + Warmup)")
print("========================================")

print(f"Device       : {device}")
print(f"Train Images : {len(train_dataset)}")
print(f"Val Images   : {len(val_dataset)}")
print(f"Epochs       : {EPOCHS}")
print(f"Batch Size   : {BATCH_SIZE}")
print(f"Base LR      : {BASE_LR}  /  CBAM LR : {CBAM_LR}")


# =========================================================
# 9. Epoch Loop
# =========================================================
for epoch in range(EPOCHS):

    epoch_start_time = time.time()

    # ---------- Train ----------
    model.train()
    total_loss = 0.0

    pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1:03d} Train", leave=False)
    for img, mask in pbar_train:
        img = img.to(device)
        mask = mask.to(device)

        if mask.dim() == 4 and mask.shape[1] == 1:
            mask = mask.squeeze(1)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            out = model(img)
            loss_ce = criterion_ce(out, mask)
            loss_dice = dice_loss(out, mask, NUM_CLASSES, class_weights)
            loss = loss_ce + loss_dice

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar_train.set_postfix({"loss": f"{loss.item():.4f}"})

    train_loss = total_loss / len(train_loader)

    # ---------- Validation ----------
    model.eval()
    epoch_inter = np.zeros(NUM_CLASSES)
    epoch_union = np.zeros(NUM_CLASSES)
    epoch_correct = np.zeros(NUM_CLASSES)
    epoch_pixels = np.zeros(NUM_CLASSES)

    pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1:03d} Val  ", leave=False)
    with torch.no_grad():
        for img, mask in pbar_val:
            img = img.to(device)
            mask = mask.to(device)

            if mask.dim() == 4 and mask.shape[1] == 1:
                mask = mask.squeeze(1)

            out = model(img)
            pred = out.argmax(dim=1).cpu().numpy()
            mask_np = mask.cpu().numpy()

            for cls in range(NUM_CLASSES):
                pred_i = (pred == cls)
                mask_i = (mask_np == cls)
                epoch_inter[cls] += (pred_i & mask_i).sum()
                epoch_union[cls] += (pred_i | mask_i).sum()
                epoch_correct[cls] += (pred_i & mask_i).sum()
                epoch_pixels[cls] += mask_i.sum()

    # ---------- Metrics ----------
    ious = epoch_inter / np.maximum(epoch_union, 1e-10)
    accs = epoch_correct / np.maximum(epoch_pixels, 1e-10)

    ious_11 = ious[:-1]
    pixels_11 = epoch_pixels[:-1]
    correct_11 = epoch_correct[:-1]
    valid_classes_11 = pixels_11 > 0

    current_miou = np.mean(ious_11[valid_classes_11])
    current_pa = np.sum(correct_11) / np.maximum(np.sum(pixels_11), 1e-10)

    epoch_time = time.time() - epoch_start_time

    # ---------- 日志 ----------
    if epoch == 0:
        print("\n")
        print("Epoch  TrainLoss    mIoU      PA      LR(back) LR(cbam)  Time")
        print("-" * 75)

    lr_back = optimizer.param_groups[0]['lr']
    lr_cbam = optimizer.param_groups[1]['lr']

    print(
        f"{epoch+1:03d}    "
        f"{train_loss:.4f}     "
        f"{current_miou*100:5.2f}%   "
        f"{current_pa*100:5.2f}%   "
        f"{lr_back:.6f}  "
        f"{lr_cbam:.6f}  "
        f"{epoch_time:.1f}s"
    )

    # ---------- Save Best ----------
    if current_miou > best_miou:
        best_miou = current_miou
        torch.save(model.state_dict(), "unet_cbam.pth")
        print(f"\n✅ Best mIoU Updated: {best_miou*100:.2f}%")
        print("Saved: unet_cbam.pth")

        print("\n=== Best checkpoint ===")
        print("+------------+-------+-------+")
        print("|    Class   |  IoU  |  Acc  |")
        print("+------------+-------+-------+")
        for cls in range(NUM_CLASSES - 1):
            if epoch_pixels[cls] == 0:
                c_iou = 0.0
                c_acc = 0.0
            else:
                c_iou = ious[cls] * 100
                c_acc = accs[cls] * 100
            c_name = CLASS_NAMES[cls][:10]
            print(f"| {c_name:>10} | {c_iou:5.2f} | {c_acc:5.2f} |")
        print("+------------+-------+-------+\n")

    # 更新学习率
    scheduler.step()


# =========================================================
# 结束
# =========================================================
print("\n========================================")
print("✅ Stage 2 Training Finished!")
print(f"Best mIoU : {best_miou*100:.2f}%")
print("========================================")