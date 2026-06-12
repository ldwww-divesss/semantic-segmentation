from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from models import deeplab_from_train_args
from utils.data import CLASSES, IGNORE_INDEX, NUM_CLASSES, build_dataloaders
from utils.metrics import SegMetric
from utils.tta import SegmentationTTA


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Evaluate a stage-two checkpoint with standard inference and TTA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=MODULE_ROOT.parent)
    p.add_argument("--split-protocol", choices=["internal-val", "official-test"], default="internal-val")
    p.add_argument("--split-dir", type=Path)
    p.add_argument("--scales", type=float, nargs="+", default=[0.75, 1.0, 1.25])
    p.add_argument("--flip", action="store_true", default=True)
    p.add_argument("--no-flip", dest="flip", action="store_false")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-base", action="store_true")
    p.add_argument("--output", type=Path)
    return p.parse_args(argv)


def load_model_from_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu")
    train_args = checkpoint.get("args", {})
    model = deeplab_from_train_args(train_args, num_classes=NUM_CLASSES, pretrained_backbone=False)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


@torch.no_grad()
def evaluate_standard(model, loader, device):
    metric = SegMetric(NUM_CLASSES, IGNORE_INDEX)
    for batch in loader:
        images = batch["image"].to(device)
        metric.update(model(images).argmax(1), batch["mask"].to(device))
    return metric.compute()


@torch.no_grad()
def evaluate_with_tta(tta, loader, device):
    metric = SegMetric(NUM_CLASSES, IGNORE_INDEX)
    for batch in loader:
        images = batch["image"].to(device)
        metric.update(tta(images).argmax(1), batch["mask"].to(device))
    return metric.compute()


def _clean(results):
    return {key: value.tolist() if hasattr(value, "tolist") else value
            for key, value in results.items() if key != "confusion"}


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    _, loader = build_dataloaders(
        args.data_root, split_protocol=args.split_protocol, split_dir=args.split_dir,
        batch_size=1, val_batch_size=1, num_workers=args.workers,
    )
    output = {
        "meta": {
            "checkpoint": str(args.checkpoint), "split_protocol": args.split_protocol,
            "evaluation_images": len(loader.dataset), "scales": args.scales,
            "flip": args.flip, "best_miou_in_checkpoint": checkpoint.get("best_miou"),
        }
    }
    if not args.no_base:
        started = time.time()
        base = evaluate_standard(model, loader, device)
        output["base_results"] = _clean(base)
        output["meta"]["base_seconds"] = time.time() - started
        print(f"standard mIoU={base['miou'] * 100:.2f}% PA={base['pa'] * 100:.2f}%")
    started = time.time()
    tta = SegmentationTTA(model, tuple(args.scales), args.flip)
    result = evaluate_with_tta(tta, loader, device)
    output["tta_results"] = _clean(result)
    output["meta"]["tta_seconds"] = time.time() - started
    if "base_results" in output:
        output["delta_miou_percent"] = 100 * (result["miou"] - output["base_results"]["miou"])
    print(f"TTA mIoU={result['miou'] * 100:.2f}% PA={result['pa'] * 100:.2f}%")
    for name, iou in zip(CLASSES, result["iou_per_class"]):
        print(f"  {name:<12} {iou * 100:6.2f}%")
    save_path = args.output or args.checkpoint.with_name("tta_results.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"saved={save_path}")


if __name__ == "__main__":
    main()
