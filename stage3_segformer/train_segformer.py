#!/usr/bin/env python3
"""
Stage 3 — SegFormer-B2 Training on CamVid

Protocol (matching Stage 2):
  - AdamW (lr=6e-5, weight_decay=0.01), Cosine LR, 100 epochs
  - CE + Dice loss
  - Same data augmentation as Stage 1/2
  - Reports: mIoU, Pixel Acc, per-class IoU, FPS

Usage:
  source venv/bin/activate
  python train_segformer.py --epochs 100 --batch-size 8 --gpus 0
"""

import argparse, os, time, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # repo root → common/
from common.dataset import CamVidDataset, get_train_transform, get_val_transform, \
    CAMVID_CLASSES, NUM_CLASSES, IGNORE_INDEX
from common.metrics import SegmentationMetrics


# ──────────────── Loss ────────────────

class CEDiceLoss(nn.Module):
    def __init__(self, ce_weight=None, dice_weight=1.0, ignore_index=255, num_classes=11):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight, ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)

        if self.dice_weight <= 0:
            return ce_loss

        # Dice loss
        probs = torch.softmax(logits, dim=1)
        targets_onehot = nn.functional.one_hot(
            targets.clamp(0, self.num_classes - 1), self.num_classes
        ).permute(0, 3, 1, 2).float()

        mask = (targets != self.ignore_index).unsqueeze(1).float()
        probs = probs * mask
        targets_onehot = targets_onehot * mask

        dims = (0, 2, 3)
        intersection = (probs * targets_onehot).sum(dim=dims)
        union = probs.sum(dim=dims) + targets_onehot.sum(dim=dims)
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1 - dice.mean()

        return ce_loss + self.dice_weight * dice_loss


# ──────────────── Model ────────────────

def build_segformer(model_name="nvidia/segformer-b2-finetuned-ade-512-512"):
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    return model


# ──────────────── FPS Measurement ────────────────

@torch.no_grad()
def measure_fps(model, device, input_size=(1, 3, 360, 480), warmup=30, runs=100):
    model.eval()
    dummy = torch.randn(*input_size).to(device)
    for _ in range(warmup):
        _ = model(dummy).logits
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(runs):
        _ = model(dummy).logits
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    return runs / elapsed


# ──────────────── Train / Val ────────────────

def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            logits = model(imgs).logits
            # SegFormer outputs at 1/4 resolution, upsample to match label
            if logits.shape[-2:] != labels.shape[-2:]:
                logits = nn.functional.interpolate(
                    logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            loss = criterion(logits, labels)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    metrics = SegmentationMetrics()
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs).logits
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
        total_loss += criterion(logits, labels).item()
        metrics.update(logits.argmax(dim=1), labels)
    return total_loss / len(loader), metrics


# ──────────────── Main ────────────────

def main():
    parser = argparse.ArgumentParser('SegFormer-B2 CamVid Training')
    parser.add_argument('--model-name', default='nvidia/segformer-b2-finetuned-ade-512-512')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--gpus', default='0', help='GPU ids, e.g. 0 or 0,1')
    parser.add_argument('--data-root', default='.')
    parser.add_argument('--save-dir', default='checkpoints')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda')
    n_gpus = torch.cuda.device_count()
    print('SegFormer-B2 Training on CamVid')
    print('=' * 60)
    print('GPUs: %d x %s' % (n_gpus, torch.cuda.get_device_name(0)))
    print('AMP: %s' % args.amp)

    # Data
    train_ds = CamVidDataset(
        os.path.join(args.data_root, 'train', 'images'),
        os.path.join(args.data_root, 'train', 'labels'),
        get_train_transform(),
    )
    val_ds = CamVidDataset(
        os.path.join(args.data_root, 'test', 'images'),
        os.path.join(args.data_root, 'test', 'labels'),
        get_val_transform(),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print('Train: %d images | Val: %d images' % (len(train_ds), len(val_ds)))

    # Model
    model = build_segformer(args.model_name)
    n_param = sum(p.numel() for p in model.parameters()) / 1e6
    print('Model: SegFormer-B2 | Params: %.1fM' % n_param)

    # Multi-GPU
    if n_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # Loss, Optimizer, Scheduler
    criterion = CEDiceLoss(ignore_index=IGNORE_INDEX, num_classes=NUM_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    scaler = torch.amp.GradScaler('cuda') if args.amp else None

    ckpt_path = os.path.join(args.save_dir, 'segformer_b2_best.pth')
    best_miou = 0.0
    no_improve = 0
    history = []

    print('\nEpoch  TrainLoss  ValLoss    mIoU      PA     LR     Time')
    print('-' * 65)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, metrics = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        miou = metrics.miou()
        pa = metrics.pixel_accuracy()
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        history.append({
            'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
            'miou': miou, 'pa': pa, 'lr': lr, 'time': elapsed,
        })

        print('%5d  %9.4f  %8.4f  %6.2f%%  %6.2f%%  %.2e  %5.1fs' % (
            epoch, train_loss, val_loss, miou*100, pa*100, lr, elapsed))

        if miou > best_miou:
            best_miou = miou
            no_improve = 0
            state = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'miou': miou,
                'pa': pa,
                'per_class_iou': metrics.iou_per_class().tolist(),
                'n_param_M': n_param,
                'args': vars(args),
            }
            torch.save(state, ckpt_path)
            print('         -> best mIoU: %.2f%% [saved]' % (miou*100))
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print('\nEarly stopping at epoch %d (patience=%d)' % (epoch, args.patience))
                break

    # ── Final report ──
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print('\n' + '=' * 65)
    print('Best checkpoint (epoch %d)' % ckpt['epoch'])
    print('=' * 65)
    for name, iou in zip(CAMVID_CLASSES, ckpt['per_class_iou']):
        s = '%6.2f%%' % (iou*100) if not np.isnan(iou) else '   N/A'
        print('  %-12s %s' % (name, s))
    print('  %-12s %6.2f%%' % ('mIoU', ckpt['miou']*100))
    print('  %-12s %6.2f%%' % ('Pixel Acc', ckpt['pa']*100))

    # FPS
    fps_model = build_segformer(args.model_name)
    fps_model.load_state_dict(ckpt['model_state_dict'])
    fps_model = fps_model.to(device)
    fps = measure_fps(fps_model, device)
    print('  %-12s %.1f' % ('FPS', fps))

    # Save history
    hist_path = os.path.join(args.save_dir, 'segformer_b2_history.json')
    with open(hist_path, 'w') as f:
        json.dump({'history': history, 'best_miou': best_miou,
                   'fps': fps, 'n_param_M': n_param}, f, indent=2)
    print('\nHistory saved to %s' % hist_path)


if __name__ == '__main__':
    main()
