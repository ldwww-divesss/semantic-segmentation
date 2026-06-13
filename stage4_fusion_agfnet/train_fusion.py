#!/usr/bin/env python3
"""
Stage 4 — CNN-Transformer Fusion with Adaptive Gated Fusion Module (AGFM)

Dual-path architecture:
  - CNN branch: ResNet34 + FPN (local features)
  - Transformer branch: MiT-B2 from SegFormer (global context)
  - AGFM: Adaptive Gated Fusion Module (learned per-pixel weighting)

Ablation variants:
  --fusion agfm         : AGFM gate fusion (proposed, default)
  --fusion none_concat  : Naive concatenation fusion
  --fusion none_add     : Simple addition fusion

Usage:
  source venv/bin/activate
  python train_fusion.py --fusion agfm --epochs 100 --batch-size 6
"""

import argparse, os, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from transformers import SegformerModel

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # repo root → common/
from common.dataset import CamVidDataset, get_train_transform, get_val_transform, \
    NUM_CLASSES, IGNORE_INDEX, CAMVID_CLASSES
from common.metrics import SegmentationMetrics


# ──────────────── AGFM Module ────────────────

class AGFM(nn.Module):
    """Adaptive Gated Fusion Module.
    F_fused = alpha * F_cnn + (1 - alpha) * F_trans
    alpha predicted by lightweight 2-layer 1x1 conv gate.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, 1),
        )

    def forward(self, f_cnn, f_trans):
        alpha = torch.sigmoid(self.gate(torch.cat([f_cnn, f_trans], dim=1)))
        return alpha * f_cnn + (1 - alpha) * f_trans


# ──────────────── CNN Branch (ResNet34 + FPN) ────────────────

class CNNBranch(nn.Module):
    """ResNet34 encoder + FPN, outputs feature at 1/4 resolution."""
    def __init__(self, out_channels=128):
        super().__init__()
        base = smp.Unet(
            encoder_name='resnet34', encoder_weights='imagenet',
            in_channels=3, classes=NUM_CLASSES,
        )
        self.encoder = base.encoder
        # SMP ResNet34 encoder outputs 6 stages:
        # [0](3ch), [1](64,h/2), [2](64,h/4), [3](128,h/8), [4](256,h/16), [5](512,h/32)
        self.lateral5 = nn.Conv2d(512, out_channels, 1)
        self.lateral4 = nn.Conv2d(256, out_channels, 1)
        self.lateral3 = nn.Conv2d(128, out_channels, 1)
        self.lateral2 = nn.Conv2d(64, out_channels, 1)
        self.smooth = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Freeze early layers
        for name, param in self.encoder.named_parameters():
            if name.startswith('layer0'):
                param.requires_grad = False

    def forward(self, x):
        feats = self.encoder(x)
        p5 = self.lateral5(feats[5])
        p4 = self.lateral4(feats[4]) + F.interpolate(p5, size=feats[4].shape[-2:], mode='bilinear', align_corners=False)
        p3 = self.lateral3(feats[3]) + F.interpolate(p4, size=feats[3].shape[-2:], mode='bilinear', align_corners=False)
        p2 = self.lateral2(feats[2]) + F.interpolate(p3, size=feats[2].shape[-2:], mode='bilinear', align_corners=False)
        return self.smooth(p2)


# ──────────────── Transformer Branch (MiT-B2) ────────────────

class TransBranch(nn.Module):
    """MiT-B2 encoder from SegFormer, outputs 1/4 resolution features.
    Uses hidden_states[0] which is at 1/4 res (H/4, W/4) with 64 channels for B2.
    """
    def __init__(self, out_channels=128, use_stage=0):
        super().__init__()
        self.use_stage = use_stage
        self.mit = SegformerModel.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512"
        )
        mit_ch = self.mit.config.hidden_sizes[use_stage]  # stage0=64 for B2
        self.proj = nn.Sequential(
            nn.Conv2d(mit_ch, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Freeze early patch embeddings
        for name, param in self.mit.named_parameters():
            if 'patch_embeddings.0' in name:
                param.requires_grad = False

    def forward(self, x):
        outputs = self.mit(x, output_hidden_states=True)
        feat = outputs.hidden_states[self.use_stage]  # (B, 64, H/4, W/4) for stage0
        return self.proj(feat)


# ──────────────── Fusion Model ────────────────

class CnnTransFusion(nn.Module):
    def __init__(self, fusion_type='agfm', out_channels=128):
        super().__init__()
        self.fusion_type = fusion_type
        self.cnn_branch = CNNBranch(out_channels)
        self.trans_branch = TransBranch(out_channels)

        if fusion_type == 'agfm':
            self.agfm = AGFM(out_channels)
            fuse_ch = out_channels
        elif fusion_type == 'none_concat':
            fuse_ch = out_channels * 2
        elif fusion_type == 'none_add':
            fuse_ch = out_channels
        else:
            raise ValueError(f"Unknown fusion: {fusion_type}")

        self.decoder = nn.Sequential(
            nn.Conv2d(fuse_ch, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(out_channels, NUM_CLASSES, 1)

    def forward(self, x):
        H, W = x.shape[2:]
        f_cnn = self.cnn_branch(x)
        f_trans = self.trans_branch(x)

        if self.fusion_type == 'agfm':
            fused = self.agfm(f_cnn, f_trans)
        elif self.fusion_type == 'none_concat':
            fused = torch.cat([f_cnn, f_trans], dim=1)
        else:
            fused = f_cnn + f_trans

        decoded = self.decoder(fused)
        logits = self.head(decoded)
        return F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

    def get_gate_map(self, x):
        """Return AGFM gate alpha for visualization."""
        f_cnn = self.cnn_branch(x)
        f_trans = self.trans_branch(x)
        return torch.sigmoid(self.agfm.gate(torch.cat([f_cnn, f_trans], dim=1)))


# ──────────────── Loss ────────────────

class CEDiceFocalLoss(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX,
                 dice_w=1.0, focal_w=0.5, focal_gamma=2.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_w = dice_w
        self.focal_w = focal_w
        self.focal_gamma = focal_gamma
        self.nc = num_classes
        self.ig = ignore_index

    def forward(self, logits, targets):
        loss = self.ce(logits, targets)
        # Dice
        probs = torch.softmax(logits, dim=1)
        tgt_oh = F.one_hot(targets.clamp(0, self.nc-1), self.nc).permute(0,3,1,2).float()
        mask = (targets != self.ig).unsqueeze(1).float()
        inter = (probs * mask * tgt_oh * mask).sum(dim=(0,2,3))
        union = (probs * mask).sum(dim=(0,2,3)) + (tgt_oh * mask).sum(dim=(0,2,3))
        dice = (2*inter+1e-6) / (union+1e-6)
        loss = loss + self.dice_w * (1 - dice.mean())
        # Focal
        if self.focal_w > 0:
            ce_r = F.cross_entropy(logits, targets, ignore_index=self.ig, reduction='none')
            pt = torch.exp(-ce_r)
            focal = ((1-pt)**self.focal_gamma) * ce_r
            loss = loss + self.focal_w * focal[targets!=self.ig].mean()
        return loss


# ──────────────── Train / Val ────────────────

def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            loss = criterion(model(imgs), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
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


@torch.no_grad()
def measure_fps(model, device, warmup=20, runs=80):
    model.eval()
    dummy = torch.randn(1, 3, 360, 480).to(device)
    for _ in range(warmup):
        _ = model(dummy)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(runs):
        _ = model(dummy)
    torch.cuda.synchronize()
    return runs / (time.time() - t0)


# ──────────────── Main ────────────────

def main():
    parser = argparse.ArgumentParser('CNN-Transformer Fusion (Stage 4)')
    parser.add_argument('--fusion', default='agfm',
                        choices=['agfm', 'none_concat', 'none_add'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data-root', default='.')
    parser.add_argument('--save-dir', default='checkpoints')
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')

    print('CNN-Transformer Fusion Training')
    print('=' * 60)
    print('Fusion: %s | GPU: %s' % (args.fusion, torch.cuda.get_device_name(0)))

    train_ds = CamVidDataset(
        os.path.join(args.data_root, 'train', 'images'),
        os.path.join(args.data_root, 'train', 'labels'),
        get_train_transform())
    val_ds = CamVidDataset(
        os.path.join(args.data_root, 'test', 'images'),
        os.path.join(args.data_root, 'test', 'labels'),
        get_val_transform())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print('Train: %d | Val: %d' % (len(train_ds), len(val_ds)))

    model = CnnTransFusion(fusion_type=args.fusion).to(device)
    n_param = sum(p.numel() for p in model.parameters()) / 1e6
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print('Params: %.1fM total, %.1fM trainable' % (n_param, n_train))

    criterion = CEDiceFocalLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    scaler = torch.amp.GradScaler('cuda')

    tag = 'fusion_%s' % args.fusion
    ckpt_path = os.path.join(args.save_dir, '%s_best.pth' % tag)
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
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                        'miou': miou, 'pa': pa, 'lr': lr, 'time': elapsed})

        print('%5d  %9.4f  %8.4f  %6.2f%%  %6.2f%%  %.2e  %5.1fs' % (
            epoch, train_loss, val_loss, miou*100, pa*100, lr, elapsed))

        if miou > best_miou:
            best_miou = miou
            no_improve = 0
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'miou': miou, 'pa': pa,
                'per_class_iou': metrics.iou_per_class().tolist(),
                'n_param_M': n_param, 'fusion_type': args.fusion,
            }, ckpt_path)
            print('         -> best mIoU: %.2f%% [saved]' % (miou*100))
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print('\nEarly stopping at epoch %d' % epoch)
                break

    # Final
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    fps = measure_fps(model, device)
    print('\n' + '=' * 65)
    print('Best: %s (epoch %d)' % (tag, ckpt['epoch']))
    print('=' * 65)
    for name, iou in zip(CAMVID_CLASSES, ckpt['per_class_iou']):
        s = '%6.2f%%' % (iou*100) if not np.isnan(iou) else '   N/A'
        print('  %-12s %s' % (name, s))
    print('  %-12s %6.2f%%' % ('mIoU', ckpt['miou']*100))
    print('  %-12s %6.2f%%' % ('Pixel Acc', ckpt['pa']*100))
    print('  %-12s %.1f' % ('FPS', fps))
    print('  %-12s %.1fM' % ('Params', n_param))

    with open(os.path.join(args.save_dir, '%s_history.json' % tag), 'w') as f:
        json.dump({'history': history, 'best_miou': best_miou,
                   'fps': fps, 'n_param_M': n_param, 'fusion_type': args.fusion}, f, indent=2)
    print('\nDone.')


if __name__ == '__main__':
    main()
