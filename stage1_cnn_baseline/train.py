#!/usr/bin/env python3
"""
Stage 1 Baseline Training — CamVid Semantic Segmentation
Supports U-Net (ResNet34) and DeepLabV3+ (ResNet50) via SMP library.

Usage:
  # CPU quick test (1 epoch, batch=2)
  python train.py --model unet --epochs 1 --batch-size 2 --device cpu

  # Full training on GPU
  python train.py --model unet --epochs 100 --device auto
  python train.py --model deeplabv3plus --epochs 80 --device auto
"""
import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # repo root → common/
from common.dataset import CamVidDataset, get_train_transform, get_val_transform
from common.metrics import SegmentationMetrics, CAMVID_CLASSES


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser('CamVid Stage-1 Baseline')
    p.add_argument('--model',        default='unet',
                   choices=['unet', 'deeplabv3plus'], help='Architecture')
    p.add_argument('--encoder',      default=None,
                   help='Backbone (default: resnet34 for unet, resnet50 for deeplabv3plus)')
    p.add_argument('--epochs',       type=int,   default=100)
    p.add_argument('--batch-size',   type=int,   default=8)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--patience',     type=int,   default=15,
                   help='Early stopping patience (epochs)')
    p.add_argument('--device',       default='auto',
                   choices=['auto', 'cuda', 'mps', 'cpu'])
    p.add_argument('--data-root',    default='.',
                   help='Directory containing train/ and test/ subdirs')
    p.add_argument('--save-dir',     default='checkpoints')
    p.add_argument('--num-workers',  type=int,   default=0,
                   help='DataLoader workers (0 = main process, safe for macOS)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(name: str, encoder: str | None) -> nn.Module:
    if name == 'unet':
        return smp.Unet(
            encoder_name=encoder or 'resnet34',
            encoder_weights='imagenet',
            in_channels=3, classes=11,
        )
    elif name == 'deeplabv3plus':
        return smp.DeepLabV3Plus(
            encoder_name=encoder or 'resnet50',
            encoder_weights='imagenet',
            in_channels=3, classes=11,
        )
    raise ValueError(f'Unknown model: {name}')


# ---------------------------------------------------------------------------
# Train / Val loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
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
        logits = model(imgs)
        total_loss += criterion(logits, labels).item()
        metrics.update(logits.argmax(dim=1), labels)

    return total_loss / len(loader), metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def select_device(arg: str) -> torch.device:
    if arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(arg)


def main():
    args   = parse_args()
    device = select_device(args.device)
    print(f'Device  : {device}')

    # ── Data ──────────────────────────────────────────────────────────────
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
    pin = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin)
    print(f'Train   : {len(train_ds)} images  |  Val: {len(val_ds)} images')

    # ── Model ─────────────────────────────────────────────────────────────
    model   = build_model(args.model, args.encoder).to(device)
    n_param = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Model   : {args.model}  |  Params: {n_param:.1f}M')

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path  = os.path.join(args.save_dir, f'{args.model}_best.pth')

    best_miou  = 0.0
    no_improve = 0

    print(f'\n{"Epoch":>5}  {"TrainLoss":>9}  {"ValLoss":>8}  {"mIoU":>7}  {"PA":>7}  {"Time":>6}')
    print('-' * 58)

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss           = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, metrics    = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        miou = metrics.miou()
        pa   = metrics.pixel_accuracy()
        elapsed = time.time() - t0

        print(f'{epoch:5d}  {train_loss:9.4f}  {val_loss:8.4f}  '
              f'{miou*100:6.2f}%  {pa*100:6.2f}%  {elapsed:5.1f}s')

        if miou > best_miou:
            best_miou  = miou
            no_improve = 0
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'miou':             miou,
                'pa':               pa,
                'per_class_iou':    metrics.iou_per_class().tolist(),
                'args':             vars(args),
            }, ckpt_path)
            print(f'         -> best mIoU updated: {miou*100:.2f}%  [saved {ckpt_path}]')
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f'\nEarly stopping at epoch {epoch} '
                      f'(no improvement for {args.patience} epochs).')
                break

    # ── Final per-class report ────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(f'\n=== Best checkpoint (epoch {ckpt["epoch"]}) ===')
    for name, iou in zip(CAMVID_CLASSES, ckpt['per_class_iou']):
        import numpy as np
        iou_str = f'{iou*100:6.2f}%' if not np.isnan(iou) else '   N/A '
        print(f'  {name:<12} {iou_str}')
    print(f'  {"mIoU":<12} {ckpt["miou"]*100:6.2f}%')
    print(f'  {"Pixel Acc":<12} {ckpt["pa"]*100:6.2f}%')


if __name__ == '__main__':
    main()
