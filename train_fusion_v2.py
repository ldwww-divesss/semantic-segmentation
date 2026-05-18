#!/usr/bin/env python3
"""
Stage 4 v2 — CNN-Transformer Fusion with multi-scale AGFM

Key improvements over v1:
  - Multi-scale fusion: uses ALL MiT-B2 stages, not just stage0
  - Lower learning rate (6e-5, matching SegFormer protocol)
  - More trainable Transformer parameters
  - Optional TTA at eval time
"""

import argparse, os, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from transformers import SegformerModel

from dataset import CamVidDataset, get_train_transform, get_val_transform, \
    NUM_CLASSES, IGNORE_INDEX, CAMVID_CLASSES
from metrics import SegmentationMetrics


class AGFM(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 1, 1),
        )
    def forward(self, f_cnn, f_trans):
        alpha = torch.sigmoid(self.gate(torch.cat([f_cnn, f_trans], dim=1)))
        return alpha * f_cnn + (1 - alpha) * f_trans


class MultiScaleTransBranch(nn.Module):
    """Uses ALL MiT-B2 stages, fuses multi-scale Transformer features."""
    def __init__(self, out_channels=128):
        super().__init__()
        self.mit = SegformerModel.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512"
        )
        # B2 hidden_sizes: [64, 128, 320, 512]
        hs = self.mit.config.hidden_sizes  # [64, 128, 320, 512]
        # Project each stage to out_channels and upsample to 1/4 res
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, out_channels, 1), nn.BatchNorm2d(out_channels), nn.ReLU(True))
            for c in hs
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        H4, W4 = x.shape[2] // 4, x.shape[3] // 4
        outs = self.mit(x, output_hidden_states=True).hidden_states
        feats = []
        for i, proj in enumerate(self.projs):
            f = proj(outs[i])
            if f.shape[-2:] != (H4, W4):
                f = F.interpolate(f, size=(H4, W4), mode='bilinear', align_corners=False)
            feats.append(f)
        return self.fuse(torch.cat(feats, dim=1))  # (B, out_ch, H/4, W/4)


class CNNBranch(nn.Module):
    def __init__(self, out_channels=128):
        super().__init__()
        base = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=NUM_CLASSES)
        self.encoder = base.encoder
        self.lateral5 = nn.Conv2d(512, out_channels, 1)
        self.lateral4 = nn.Conv2d(256, out_channels, 1)
        self.lateral3 = nn.Conv2d(128, out_channels, 1)
        self.lateral2 = nn.Conv2d(64, out_channels, 1)
        self.smooth = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        feats = self.encoder(x)
        p5 = self.lateral5(feats[5])
        p4 = self.lateral4(feats[4]) + F.interpolate(p5, size=feats[4].shape[-2:], mode='bilinear', align_corners=False)
        p3 = self.lateral3(feats[3]) + F.interpolate(p4, size=feats[3].shape[-2:], mode='bilinear', align_corners=False)
        p2 = self.lateral2(feats[2]) + F.interpolate(p3, size=feats[2].shape[-2:], mode='bilinear', align_corners=False)
        return self.smooth(p2)


class CnnTransFusionV2(nn.Module):
    def __init__(self, fusion_type='agfm', out_channels=128):
        super().__init__()
        self.fusion_type = fusion_type
        self.cnn = CNNBranch(out_channels)
        self.trans = MultiScaleTransBranch(out_channels)

        if fusion_type == 'agfm':
            self.agfm = AGFM(out_channels)
            fuse_ch = out_channels
        elif fusion_type == 'none_concat':
            fuse_ch = out_channels * 2
        else:
            fuse_ch = out_channels

        self.decoder = nn.Sequential(
            nn.Conv2d(fuse_ch, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(True),
            nn.Dropout2d(0.1),
        )
        self.head = nn.Conv2d(out_channels, NUM_CLASSES, 1)

    def forward(self, x):
        H, W = x.shape[2:]
        f_cnn = self.cnn(x)
        f_trans = self.trans(x)
        if self.fusion_type == 'agfm':
            fused = self.agfm(f_cnn, f_trans)
        elif self.fusion_type == 'none_concat':
            fused = torch.cat([f_cnn, f_trans], dim=1)
        else:
            fused = f_cnn + f_trans
        logits = self.head(self.decoder(fused))
        return F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

    def get_gate_map(self, x):
        f_cnn = self.cnn(x)
        f_trans = self.trans(x)
        return torch.sigmoid(self.agfm.gate(torch.cat([f_cnn, f_trans], dim=1)))


class CEDiceFocalLoss(nn.Module):
    def __init__(self, nc=NUM_CLASSES, ig=IGNORE_INDEX, dw=1.0, fw=0.5, fg=2.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ig)
        self.dw, self.fw, self.fg, self.nc, self.ig = dw, fw, fg, nc, ig

    def forward(self, logits, targets):
        loss = self.ce(logits, targets)
        probs = torch.softmax(logits, 1)
        tgt_oh = F.one_hot(targets.clamp(0, self.nc-1), self.nc).permute(0,3,1,2).float()
        m = (targets != self.ig).unsqueeze(1).float()
        inter = (probs*m * tgt_oh*m).sum(dim=(0,2,3))
        union = (probs*m).sum(dim=(0,2,3)) + (tgt_oh*m).sum(dim=(0,2,3))
        loss = loss + self.dw * (1 - ((2*inter+1e-6)/(union+1e-6)).mean())
        if self.fw > 0:
            ce_r = F.cross_entropy(logits, targets, ignore_index=self.ig, reduction='none')
            pt = torch.exp(-ce_r)
            loss = loss + self.fw * (((1-pt)**self.fg)*ce_r)[targets!=self.ig].mean()
        return loss


def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total = 0.0
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            loss = criterion(model(imgs), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def val_epoch(model, loader, criterion, device, tta=False):
    model.eval()
    total = 0.0
    metrics = SegmentationMetrics()
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        if tta:
            logits = model(imgs)
            logits = (logits + torch.flip(model(torch.flip(imgs, [-1])), [-1])) / 2
        else:
            logits = model(imgs)
        total += criterion(logits, labels).item()
        metrics.update(logits.argmax(dim=1), labels)
    return total / len(loader), metrics


@torch.no_grad()
def measure_fps(model, device, warmup=20, runs=80):
    model.eval()
    d = torch.randn(1, 3, 360, 480).to(device)
    for _ in range(warmup): _ = model(d)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(runs): _ = model(d)
    torch.cuda.synchronize()
    return runs / (time.time() - t0)


def main():
    parser = argparse.ArgumentParser('CNN-Trans Fusion v2')
    parser.add_argument('--fusion', default='agfm', choices=['agfm', 'none_concat', 'none_add'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data-root', default='.')
    parser.add_argument('--save-dir', default='checkpoints')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--tta', action='store_true', help='Use TTA at final eval')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')
    print('CNN-Trans Fusion v2 | %s | GPU %s' % (args.fusion, torch.cuda.get_device_name(0)))

    train_ds = CamVidDataset(os.path.join(args.data_root,'train','images'),
                             os.path.join(args.data_root,'train','labels'), get_train_transform())
    val_ds = CamVidDataset(os.path.join(args.data_root,'test','images'),
                           os.path.join(args.data_root,'test','labels'), get_val_transform())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print('Train: %d | Val: %d' % (len(train_ds), len(val_ds)))

    model = CnnTransFusionV2(args.fusion).to(device)
    n_param = sum(p.numel() for p in model.parameters())/1e6
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
    print('Params: %.1fM (%.1fM trainable)' % (n_param, n_train))

    criterion = CEDiceFocalLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    scaler = torch.amp.GradScaler('cuda')

    tag = 'fusion_v2_%s' % args.fusion
    ckpt_path = os.path.join(args.save_dir, tag + '_best.pth')
    best_miou = 0.0
    no_improve = 0
    history = []

    print('\nEpoch  TrainLoss  ValLoss    mIoU      PA     LR     Time')
    print('-' * 65)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tl = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        vl, metrics = val_epoch(model, val_loader, criterion, device, tta=args.tta)
        scheduler.step()
        miou = metrics.miou()
        pa = metrics.pixel_accuracy()
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0
        history.append({'epoch':epoch,'train_loss':tl,'val_loss':vl,'miou':miou,'pa':pa,'lr':lr,'time':elapsed})
        print('%5d  %9.4f  %8.4f  %6.2f%%  %6.2f%%  %.2e  %5.1fs' % (epoch,tl,vl,miou*100,pa*100,lr,elapsed))
        if miou > best_miou:
            best_miou = miou; no_improve = 0
            torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'miou':miou,'pa':pa,
                        'per_class_iou':metrics.iou_per_class().tolist(),'n_param_M':n_param,
                        'fusion_type':args.fusion}, ckpt_path)
            print('         -> best mIoU: %.2f%% [saved]' % (miou*100))
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print('\nEarly stopping at epoch %d' % epoch); break

    # Final with TTA
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if args.tta:
        _, metrics_tta = val_epoch(model, val_loader, criterion, device, tta=True)
        miou_tta = metrics_tta.miou()
        print('\n  TTA mIoU: %.2f%% (non-TTA: %.2f%%)' % (miou_tta*100, ckpt['miou']*100))

    fps = measure_fps(model, device)
    print('\n' + '='*65)
    print('Best: %s (epoch %d)' % (tag, ckpt['epoch']))
    print('='*65)
    for name, iou in zip(CAMVID_CLASSES, ckpt['per_class_iou']):
        print('  %-12s %s' % (name, '%6.2f%%' % (iou*100) if not np.isnan(iou) else '   N/A'))
    print('  %-12s %6.2f%%' % ('mIoU', ckpt['miou']*100))
    print('  %-12s %6.2f%%' % ('PA', ckpt['pa']*100))
    print('  %-12s %.1f' % ('FPS', fps))
    print('  %-12s %.1fM' % ('Params', n_param))

    with open(os.path.join(args.save_dir, tag+'_history.json'), 'w') as f:
        json.dump({'history':history,'best_miou':best_miou,'fps':fps,'n_param_M':n_param,
                   'fusion_type':args.fusion}, f, indent=2)
    print('\nDone.')


if __name__ == '__main__':
    main()
