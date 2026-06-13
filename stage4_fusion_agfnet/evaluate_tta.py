#!/usr/bin/env python3
"""
Standalone evaluation with optional test-time augmentation (TTA)
for the Stage IV fusion models (AGFNet and static-fusion variants).

TTA modes:
  none     single forward pass (reproduces the training-time evaluation)
  flip     horizontal flip, 2 views
  ms       multi-scale {0.75, 1.0, 1.25}, 3 views
  ms-flip  multi-scale x horizontal flip, 6 views

Softmax probabilities of all views are averaged at the original
resolution before the argmax decision.

Usage:
  python evaluate_tta.py --checkpoint checkpoints/fusion_v2_agfm_best.pth \
      --fusion agfm --tta ms-flip
"""

import argparse, json, os, time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # repo root → common/
from common.dataset import CamVidDataset, get_val_transform, CAMVID_CLASSES
from common.metrics import SegmentationMetrics
from train_fusion_v2 import CnnTransFusionV2


def pick_device(name: str) -> torch.device:
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


@torch.no_grad()
def predict_probs(model, img, scales, hflip):
    """Average softmax probabilities over TTA views.

    img: (1, 3, H, W) normalized tensor on the target device.
    Returns: (1, C, H, W) averaged probabilities.
    """
    _, _, H, W = img.shape
    probs = None
    n = 0
    for s in scales:
        if s == 1.0:
            x = img
        else:
            h = max(32, int(round(H * s / 32)) * 32)
            w = max(32, int(round(W * s / 32)) * 32)
            x = F.interpolate(img, size=(h, w), mode='bilinear',
                              align_corners=False)
        views = [x]
        if hflip:
            views.append(torch.flip(x, dims=[-1]))
        for i, v in enumerate(views):
            logits = model(v)
            if i == 1:  # un-flip
                logits = torch.flip(logits, dims=[-1])
            if logits.shape[-2:] != (H, W):
                logits = F.interpolate(logits, size=(H, W), mode='bilinear',
                                       align_corners=False)
            p = torch.softmax(logits, dim=1)
            probs = p if probs is None else probs + p
            n += 1
    return probs / n


def main():
    ap = argparse.ArgumentParser('Stage IV evaluation with TTA')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--fusion', default='agfm',
                    choices=['agfm', 'none_concat', 'none_add'])
    ap.add_argument('--tta', default='none',
                    choices=['none', 'flip', 'ms', 'ms-flip'])
    ap.add_argument('--scales', type=float, nargs='+',
                    default=[0.75, 1.0, 1.25])
    ap.add_argument('--data-root', default='.')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--save-json', default=None,
                    help='Write metrics to this JSON file')
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f'Device: {device} | TTA: {args.tta}')

    if args.tta == 'none':
        scales, hflip = [1.0], False
    elif args.tta == 'flip':
        scales, hflip = [1.0], True
    elif args.tta == 'ms':
        scales, hflip = args.scales, False
    else:
        scales, hflip = args.scales, True
    n_views = len(scales) * (2 if hflip else 1)
    print(f'Scales: {scales} | hflip: {hflip} | views: {n_views}')

    ds = CamVidDataset(os.path.join(args.data_root, 'test', 'images'),
                       os.path.join(args.data_root, 'test', 'labels'),
                       get_val_transform())
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    print(f'Test images: {len(ds)}')

    model = CnnTransFusionV2(args.fusion).to(device)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    ref = ckpt.get('miou')
    if ref is not None:
        print(f'Checkpoint epoch {ckpt.get("epoch")} | '
              f'stored mIoU {ref*100:.2f}%')

    metrics = SegmentationMetrics()
    t0 = time.time()
    for i, (img, label, fname) in enumerate(loader):
        img = img.to(device)
        probs = predict_probs(model, img, scales, hflip)
        pred = probs.argmax(dim=1).cpu()
        metrics.update(pred, label)
        if (i + 1) % 20 == 0:
            print(f'  {i+1}/{len(ds)}  ({time.time()-t0:.0f}s)')
    elapsed = time.time() - t0

    print(f'\n=== Results (TTA: {args.tta}) | {elapsed:.0f}s ===')
    metrics.print_report()

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or '.', exist_ok=True)
        iou = metrics.iou_per_class()
        out = {
            'checkpoint': args.checkpoint,
            'fusion': args.fusion,
            'tta': args.tta,
            'scales': scales,
            'hflip': hflip,
            'n_views': n_views,
            'miou': metrics.miou(),
            'pixel_accuracy': metrics.pixel_accuracy(),
            'per_class_iou': {c: (None if np.isnan(v) else float(v))
                              for c, v in zip(CAMVID_CLASSES, iou)},
            'eval_seconds': elapsed,
            'device': str(device),
        }
        with open(args.save_json, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'Saved: {args.save_json}')


if __name__ == '__main__':
    main()
