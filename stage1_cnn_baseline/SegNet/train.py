import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import SegDataset
from model import SegNet
from tqdm import tqdm
import numpy as np
import time
import os

# ================= 1. 配置参数 =================
device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 12
EPOCHS = 80
BATCH_SIZE = 4
LR = 2e-4

# 精简后的 12 个类别名称（索引 11 为 Void，训练/评测时忽略）
CLASS_NAMES = [
    "Sky", "Building", "Pole", "Road", "Pavement",
    "Tree", "SignSymbol", "Fence", "Car", "Pedestrian",
    "Bicyclist", "Void"
]

# ================= 2. 数据与模型 =================
print("Loading datasets...")
train_dataset = SegDataset("/mnt/workspace/semantic_segmentation/语义分割", train=True)
val_dataset = SegDataset("/mnt/workspace/semantic_segmentation/语义分割", train=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 实例化 SegNet（编码器默认从 ImageNet 预训练 VGG16-BN 初始化）
model = SegNet(num_classes=NUM_CLASSES).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

scaler = torch.amp.GradScaler('cuda')

# ================= 3. 损失函数 =================
def dice_loss(pred, target, num_classes):
    pred = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    dice = (2. * intersection + 1e-5) / (union + 1e-5)
    return 1.0 - dice.mean()

# Void 类(索引 11)不参与 CE 损失
criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=11)

# ================= 4. 训练主循环 =================
best_miou = 0.0
print(f"\n--- 阶段一：SegNet Baseline 训练 ---")
print(f"\nDevice  : {device}")
print(f"Train   : {len(train_dataset)} images  |  Val: {len(val_dataset)} images")
print(f"Model   : SegNet (VGG16-BN encoder)  |  Classes: {NUM_CLASSES}")

for epoch in range(EPOCHS):
    epoch_start_time = time.time()

    # ---------------- 训练阶段 ----------------
    model.train()
    total_loss = 0

    pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1:02d} Train", leave=False)
    for img, mask in pbar_train:
        img, mask = img.to(device), mask.to(device)

        if mask.dim() == 4 and mask.shape[1] == 1:
            mask = mask.squeeze(1)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            out = model(img)
            loss_ce = criterion_ce(out, mask)
            loss_dice = dice_loss(out, mask, NUM_CLASSES)
            loss = loss_ce + loss_dice

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)

    # ---------------- 验证阶段 ----------------
    model.eval()

    epoch_inter = np.zeros(NUM_CLASSES)
    epoch_union = np.zeros(NUM_CLASSES)
    epoch_correct = np.zeros(NUM_CLASSES)
    epoch_pixels = np.zeros(NUM_CLASSES)

    pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1:02d} Val  ", leave=False)
    with torch.no_grad():
        for img, mask in pbar_val:
            img, mask = img.to(device), mask.to(device)
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

    # ---------------- 指标计算与日志打印 ----------------
    ious = epoch_inter / np.maximum(epoch_union, 1e-10)
    accs = epoch_correct / np.maximum(epoch_pixels, 1e-10)

    # 对 IoU 数组和像素统计数组进行切片 [:-1]，丢弃最后一个索引(11, Void类)
    ious_11 = ious[:-1]
    pixels_11 = epoch_pixels[:-1]

    valid_classes_11 = pixels_11 > 0
    current_miou = np.mean(ious_11[valid_classes_11])

    # 同样的，PA 也只计算前 11 类
    correct_11 = epoch_correct[:-1]
    current_pa = np.sum(correct_11) / np.maximum(np.sum(pixels_11), 1e-10)

    epoch_time = time.time() - epoch_start_time

    if epoch == 0:
        print("\nEpoch  TrainLoss    mIoU      PA      Time")
        print("-" * 50)

    print(f"  {epoch+1:02d}    {train_loss:.4f}     {current_miou*100:5.2f}%   {current_pa*100:5.2f}%   {epoch_time:.1f}s")

    # 保存当前最优权重
    if current_miou > best_miou:
        best_miou = current_miou
        torch.save(model.state_dict(), "segnet.pth")
        print(f"         -> best mIoU updated: {best_miou*100:.2f}%  [saved segnet.pth]")

        print(f"\n=== Best checkpoint (epoch {epoch+1}) ===")
        print("+------------+-------+-------+")
        print("|    Class   |  IoU  |  Acc  |")
        print("+------------+-------+-------+")

        for cls in range(NUM_CLASSES-1):
            if epoch_pixels[cls] == 0:
                c_iou, c_acc = 0.0, 0.0
            else:
                c_iou, c_acc = ious[cls] * 100, accs[cls] * 100

            c_name = CLASS_NAMES[cls][:10] if cls < len(CLASS_NAMES) else f"Class_{cls}"
            print(f"| {c_name:>10} | {c_iou:5.2f} | {c_acc:5.2f} |")

        print("+------------+-------+-------+\n")

print(f"\n✅ 阶段一训练彻底完成！SegNet 全局最佳 mIoU 为: {best_miou*100:.2f}%")
