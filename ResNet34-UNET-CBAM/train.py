import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import SegDataset
from model_cbam import ResNetUNetCBAM
from tqdm import tqdm
import numpy as np
import time

# ================= 1. 配置参数 =================
device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 12
EPOCHS = 40
BATCH_SIZE = 4
LR = 2e-4

CLASS_NAMES = [
    "Sky", "Building", "Pole", "Road", "Pavement", 
    "Tree", "SignSymbol", "Fence", "Car", "Pedestrian", 
    "Bicyclist", "Void"
]

# ================= 2. 数据与模型初始化 =================
print("Loading datasets...")
train_dataset = SegDataset("/mnt/workspace/semantic_segmentation/语义分割", train=True)
val_dataset = SegDataset("/mnt/workspace/semantic_segmentation/语义分割", train=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = ResNetUNetCBAM(num_classes=NUM_CLASSES).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scaler = torch.amp.GradScaler('cuda')

# ================= 3. 损失函数 (CE + Dice) =================
def dice_loss(pred, target, num_classes):
    pred = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    dice = (2. * intersection + 1e-5) / (union + 1e-5)
    return 1.0 - dice.mean()

criterion_ce = torch.nn.CrossEntropyLoss()

# ================= 4. 训练主循环 =================
best_miou = 0.0
print(f"\n--- 阶段二：ResNet34-UNet-CBAM 训练启动 ---")
print(f"Device  : {device}")
print(f"Train   : {len(train_dataset)} images | Val: {len(val_dataset)} images")

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
            loss = criterion_ce(out, mask) + dice_loss(out, mask, NUM_CLASSES)

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
                
            pred = model(img).argmax(dim=1).cpu().numpy()
            mask_np = mask.cpu().numpy()
            
            for cls in range(NUM_CLASSES):
                pred_i = (pred == cls)
                mask_i = (mask_np == cls)
                epoch_inter[cls] += (pred_i & mask_i).sum()
                epoch_union[cls] += (pred_i | mask_i).sum()
                epoch_correct[cls] += (pred_i & mask_i).sum()
                epoch_pixels[cls] += mask_i.sum()

    # ---------------- 指标计算与日志 ----------------
    ious = epoch_inter / np.maximum(epoch_union, 1e-10)
    accs = epoch_correct / np.maximum(epoch_pixels, 1e-10)
    
    valid_classes = epoch_pixels > 0
    current_miou = np.mean(ious[valid_classes])
    current_pa = np.sum(epoch_correct) / np.maximum(np.sum(epoch_pixels), 1e-10)
    
    if epoch == 0:
        print("\nEpoch  TrainLoss    mIoU      PA      Time")
        print("-" * 50)
    
    print(f"  {epoch+1:02d}    {train_loss:.4f}     {current_miou*100:5.2f}%   {current_pa*100:5.2f}%   {time.time()-epoch_start_time:.1f}s")

    # 达到最佳指标时保存模型权重
    if current_miou > best_miou:
        best_miou = current_miou
        torch.save(model.state_dict(), "best_unet_cbam.pth")
        print(f"         -> best mIoU updated: {best_miou*100:.2f}%  [saved best_unet_cbam.pth]")
        
        print(f"\n=== Best CBAM checkpoint (epoch {epoch+1}) ===")
        print("+------------+-------+-------+")
        print("|    Class   |  IoU  |  Acc  |")
        print("+------------+-------+-------+")
        for cls in range(NUM_CLASSES):
            c_iou = 0.0 if epoch_pixels[cls] == 0 else ious[cls] * 100
            c_acc = 0.0 if epoch_pixels[cls] == 0 else accs[cls] * 100
            print(f"| {CLASS_NAMES[cls][:10]:>10} | {c_iou:5.2f} | {c_acc:5.2f} |")
        print("+------------+-------+-------+\n")

print(f"\n✅ 阶段二训练彻底完成！CBAM 全局最佳 mIoU 为: {best_miou*100:.2f}%")