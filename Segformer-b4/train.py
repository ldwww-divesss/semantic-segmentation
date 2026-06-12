"""
Stage 3 — SegFormer-B4 Training on CamVid
对照实验: 与 B2 完全相同的训练协议, 仅模型不同

Protocol (matching Stage 2 / B2):
  - AdamW (lr=6e-5, weight_decay=0.01), CosineAnnealingLR, 100 epochs
  - CE + Dice loss
  - Same data augmentation as Stage 2
  - Reports: mIoU, Pixel Acc, per-class IoU, FPS

Usage:
  python train.py --epochs 100 --batch-size 4 --gpus 0
"""

import argparse, os, time, json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import SegDataset
from model import SegFormerB4


# =====================================================
# Constants
# =====================================================
NUM_CLASSES = 12
IGNORE_INDEX = 11

CLASS_NAMES = [
    "Sky", "Building", "Pole", "Road", "Pavement",
    "Tree", "SignSymbol", "Fence", "Car", "Pedestrian",
    "Bicyclist", "Void"
]


# =====================================================
# CE + Dice Loss (matching B2 protocol)
# =====================================================
class CEDiceLoss(nn.Module):
    def __init__(self, ignore_index=IGNORE_INDEX, num_classes=NUM_CLASSES, dice_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        if self.dice_weight <= 0:
            return ce_loss

        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(
            targets.clamp(0, self.num_classes - 1), self.num_classes
        ).permute(0, 3, 1, 2).float()

        mask = (targets != self.ignore_index).unsqueeze(1).float()
        probs = probs * mask
        targets_onehot = targets_onehot * mask

        dims = (0, 2, 3)
        intersection = (probs * targets_onehot).sum(dim=dims)
        union = probs.sum(dim=dims) + targets_onehot.sum(dim=dims)
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1.0 - dice.mean()

        return ce_loss + self.dice_weight * dice_loss


# =====================================================
# Per-class metrics (matching Stage 1/2)
# =====================================================
def compute_metrics(pred, mask):
    ious, accs = np.zeros(NUM_CLASSES), np.zeros(NUM_CLASSES)
    for cls in range(NUM_CLASSES):
        pred_i = (pred == cls)
        mask_i = (mask == cls)
        inter = (pred_i & mask_i).sum()
        union = (pred_i | mask_i).sum()
        cls_pixels = mask_i.sum()
        ious[cls] = inter / union if union > 0 else 0.0
        accs[cls] = inter / cls_pixels if cls_pixels > 0 else 0.0
    return ious, accs


# =====================================================
# FPS Measurement
# =====================================================
@torch.no_grad()
def measure_fps(model, device, input_size=(1, 3, 512, 512), warmup=30, runs=100):
    model.eval()
    dummy = torch.randn(*input_size).to(device)
    for _ in range(warmup):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(runs):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return runs / (time.time() - t0)


# =====================================================
# Train / Val Epochs
# =====================================================
def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Train", leave=False)
    for img, mask in pbar:
        img, mask = img.to(device), mask.to(device)
        if mask.dim() == 4 and mask.shape[1] == 1:
            mask = mask.squeeze(1)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            out = model(img)
            loss = criterion(out, mask)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    return total_loss / len(loader)


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    epoch_inter = np.zeros(NUM_CLASSES)
    epoch_union = np.zeros(NUM_CLASSES)
    epoch_correct = np.zeros(NUM_CLASSES)
    epoch_pixels = np.zeros(NUM_CLASSES)

    pbar = tqdm(loader, desc="Val  ", leave=False)
    for img, mask in pbar:
        img, mask = img.to(device), mask.to(device)
        if mask.dim() == 4 and mask.shape[1] == 1:
            mask = mask.squeeze(1)

        out = model(img)
        loss = criterion(out, mask)
        total_loss += loss.item()

        pred = out.argmax(dim=1).cpu().numpy()
        mask_np = mask.cpu().numpy()

        ious, accs = compute_metrics(pred, mask_np)
        for cls in range(NUM_CLASSES):
            pred_i = (pred == cls)
            mask_i = (mask_np == cls)
            epoch_inter[cls] += (pred_i & mask_i).sum()
            epoch_union[cls] += (pred_i | mask_i).sum()
            epoch_correct[cls] += (pred_i & mask_i).sum()
            epoch_pixels[cls] += mask_i.sum()

    return total_loss / len(loader), epoch_inter, epoch_union, epoch_correct, epoch_pixels


# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser("SegFormer-B4 CamVid Training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--data-root", default="/mnt/workspace/semantic_segmentation/语义分割")
    parser.add_argument("--save-dir", default=".")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)

    print("SegFormer-B4 Training on CamVid")
    print("=" * 60)
    print(f"  Device     : {device}")
    if device.type == "cuda":
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"  AMP        : {use_amp}")

    # Data
    train_ds = SegDataset(args.data_root, train=True)
    val_ds   = SegDataset(args.data_root, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    print(f"  Train/Val  : {len(train_ds)} / {len(val_ds)} images")
    print(f"  Batch Size : {args.batch_size}")

    # Model
    print("\nLoading SegFormer-B4...")
    model = SegFormerB4(num_classes=NUM_CLASSES).to(device)
    n_param = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Params     : {n_param:.1f} M")

    # Loss, Optimizer, Scheduler
    criterion = CEDiceLoss(ignore_index=IGNORE_INDEX, num_classes=NUM_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_miou = 0.0
    no_improve = 0
    history = []

    print(f"\n{'='*60}")
    print(f" Stage 3 : SegFormer-B4 (CE+Dice, CosineLR, wd={args.weight_decay})")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, ep_inter, ep_union, ep_correct, ep_pixels = val_epoch(
            model, val_loader, criterion, device)
        scheduler.step()

        # Compute metrics
        ious = ep_inter / np.maximum(ep_union, 1e-10)
        accs = ep_correct / np.maximum(ep_pixels, 1e-10)
        valid_mask = (np.arange(NUM_CLASSES) != IGNORE_INDEX) & (ep_pixels > 0)
        miou = np.mean(ious[valid_mask])
        pa = np.sum(ep_correct[valid_mask]) / np.maximum(np.sum(ep_pixels[valid_mask]), 1e-10)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "miou": float(miou), "pa": float(pa), "lr": lr, "time": elapsed,
        })

        if epoch == 1:
            print(f"{'Epoch':>5s}  {'TrainLoss':>9s}  {'ValLoss':>8s}  {'mIoU':>6s}  {'PA':>6s}  {'LR':>8s}  {'Time':>5s}")
            print("-" * 65)

        print(f"{epoch:5d}  {train_loss:9.4f}  {val_loss:8.4f}  {miou*100:5.2f}%  {pa*100:5.2f}%  {lr:.2e}  {elapsed:4.1f}s")

        if miou > best_miou:
            best_miou = miou
            no_improve = 0
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "miou": float(miou),
                "pa": float(pa),
                "per_class_iou": ious[:11].tolist(),
                "per_class_acc": accs[:11].tolist(),
                "n_param_M": n_param,
                "args": vars(args),
            }
            torch.save(state, os.path.join(args.save_dir, "segformer_b4_best.pth"))
            print(f"         -> best mIoU: {miou*100:.2f}%  [saved]")

            print(f"\n  {'Class':>12s}  {'IoU':>6s}  {'Acc':>6s}")
            print(f"  {'-'*28}")
            for cls in range(11):
                c_name = CLASS_NAMES[cls][:10]
                c_iou = ious[cls] * 100 if ep_pixels[cls] > 0 else 0.0
                c_acc = accs[cls] * 100 if ep_pixels[cls] > 0 else 0.0
                print(f"  {c_name:>12s}  {c_iou:5.2f}%  {c_acc:5.2f}%")
            print()
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
                break

    # ── Final report ──
    ckpt = torch.load(os.path.join(args.save_dir, "segformer_b4_best.pth"),
                      map_location="cpu", weights_only=False)
    print("\n" + "=" * 65)
    print(f"Best checkpoint (epoch {ckpt['epoch']})")
    print("=" * 65)
    for name, iou in zip(CLASS_NAMES[:11], ckpt["per_class_iou"]):
        print(f"  {name:>12s}  {iou*100:5.2f}%")
    print(f"  {'mIoU':>12s}  {ckpt['miou']*100:5.2f}%")
    print(f"  {'Pixel Acc':>12s}  {ckpt['pa']*100:5.2f}%")

    # FPS
    model.load_state_dict(ckpt["model_state_dict"])
    fps = measure_fps(model, device)
    print(f"  {'FPS':>12s}  {fps:.1f}")
    print(f"  {'Params':>12s}  {n_param:.1f} M")

    # History
    hist_path = os.path.join(args.save_dir, "segformer_b4_history.json")
    with open(hist_path, "w") as f:
        json.dump({"history": history, "best_miou": float(best_miou),
                   "fps": fps, "n_param_M": n_param}, f, indent=2)
    print(f"\nHistory saved to {hist_path}")
    print("Done.")


if __name__ == "__main__":
    main()
