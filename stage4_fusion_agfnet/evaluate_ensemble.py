#!/usr/bin/env python3
"""
Probability-level ensemble of the Stage IV fusion variants
(AGFM + Concat + Add), each with horizontal-flip TTA.

Softmax probabilities of all member models and views are averaged
before the argmax decision.
"""

import argparse, json, os, time

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # repo root → common/
from common.dataset import CamVidDataset, get_val_transform, CAMVID_CLASSES
from common.metrics import SegmentationMetrics
from train_fusion_v2 import CnnTransFusionV2
from evaluate_tta import pick_device, predict_probs


def main():
    ap = argparse.ArgumentParser('Stage IV fusion-variant ensemble')
    ap.add_argument('--members', nargs='+', default=[
        'agfm:checkpoints/fusion_v2_agfm_best.pth',
        'none_concat:checkpoints/fusion_v2_none_concat_best.pth',
        'none_add:checkpoints/fusion_v2_none_add_best.pth',
    ], help='fusion_type:checkpoint_path entries')
    ap.add_argument('--hflip', action='store_true', default=True)
    ap.add_argument('--data-root', default='.')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--save-json', default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f'Device: {device} | members: {len(args.members)} | '
          f'hflip: {args.hflip}')

    ds = CamVidDataset(os.path.join(args.data_root, 'test', 'images'),
                       os.path.join(args.data_root, 'test', 'labels'),
                       get_val_transform())
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    models = []
    for entry in args.members:
        fusion, path = entry.split(':', 1)
        m = CnnTransFusionV2(fusion).to(device)
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        m.load_state_dict(ckpt['model_state_dict'])
        m.eval()
        print(f'  loaded {fusion:<12} {path} '
              f'(stored mIoU {ckpt.get("miou", float("nan"))*100:.2f}%)')
        models.append(m)

    metrics = SegmentationMetrics()
    t0 = time.time()
    with torch.no_grad():
        for i, (img, label, _) in enumerate(loader):
            img = img.to(device)
            probs = None
            for m in models:
                p = predict_probs(m, img, scales=[1.0], hflip=args.hflip)
                probs = p if probs is None else probs + p
            pred = (probs / len(models)).argmax(dim=1).cpu()
            metrics.update(pred, label)
            if (i + 1) % 25 == 0:
                print(f'  {i+1}/{len(ds)}  ({time.time()-t0:.0f}s)')

    print(f'\n=== Ensemble results | {time.time()-t0:.0f}s ===')
    metrics.print_report()

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or '.', exist_ok=True)
        iou = metrics.iou_per_class()
        json.dump({
            'members': args.members,
            'hflip': args.hflip,
            'miou': metrics.miou(),
            'pixel_accuracy': metrics.pixel_accuracy(),
            'per_class_iou': {c: (None if np.isnan(v) else float(v))
                              for c, v in zip(CAMVID_CLASSES, iou)},
            'device': str(device),
        }, open(args.save_json, 'w'), indent=2)
        print(f'Saved: {args.save_json}')


if __name__ == '__main__':
    main()
