from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from models import deeplab_from_train_args
from utils.data import NUM_CLASSES, build_dataloaders

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Visualize the final CBAM spatial attention map.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=MODULE_ROOT.parent)
    p.add_argument("--split-protocol", choices=["internal-val", "official-test"], default="internal-val")
    p.add_argument("--split-dir", type=Path)
    p.add_argument("--num-samples", type=int, default=6)
    p.add_argument("--output-dir", type=Path, default=MODULE_ROOT / "visualizations")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


class SpatialMapHook:
    def __init__(self, spatial_module):
        self.module = spatial_module
        self.value = None
        self.handle = spatial_module.register_forward_pre_hook(self._capture)

    def _capture(self, module, inputs):
        feature = inputs[0]
        pooled = torch.cat([feature.mean(1, keepdim=True), feature.max(1, keepdim=True).values], dim=1)
        self.value = torch.sigmoid(module.conv(pooled)).detach().cpu()[0, 0].numpy()

    def close(self):
        self.handle.remove()


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    train_args = checkpoint.get("args", {})
    model = deeplab_from_train_args(train_args, num_classes=NUM_CLASSES, pretrained_backbone=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    spatial = getattr(model.backbone.cbam4, "spatial_att", None)
    if spatial is None:
        raise ValueError("Checkpoint does not contain a CBAM spatial branch.")
    return model, spatial


def denormalize(tensor):
    rgb = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip((rgb * STD + MEAN) * 255, 0, 255).astype(np.uint8)


def render(rgb, attention):
    h, w = rgb.shape[:2]
    attention = cv2.resize(attention.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    attention = cv2.normalize(attention, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat = cv2.applyColorMap(attention, cv2.COLORMAP_JET)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(bgr, 0.5, heat, 0.5, 0)
    return np.hstack([bgr, heat, overlay])


@torch.no_grad()
def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, spatial = load_model(args.checkpoint, device)
    _, loader = build_dataloaders(
        args.data_root, split_protocol=args.split_protocol, split_dir=args.split_dir,
        batch_size=1, val_batch_size=1, num_workers=args.workers,
    )
    count = min(args.num_samples, len(loader.dataset))
    indices = set(np.random.default_rng(args.seed).choice(len(loader.dataset), count, replace=False).tolist())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hook = SpatialMapHook(spatial)
    saved = []
    try:
        for index, batch in enumerate(loader):
            if index not in indices:
                continue
            images = batch["image"].to(device)
            model(images)
            stem = batch["stem"][0]
            output = args.output_dir / f"{stem}.png"
            cv2.imwrite(str(output), render(denormalize(images[0]), hook.value))
            saved.append(stem)
    finally:
        hook.close()
    metadata = {
        "checkpoint": str(args.checkpoint), "split_protocol": args.split_protocol,
        "sampled_stems": saved,
    }
    (args.output_dir / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved {len(saved)} visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
