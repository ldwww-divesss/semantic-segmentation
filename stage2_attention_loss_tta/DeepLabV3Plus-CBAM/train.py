from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, PolynomialLR

MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from losses import DiceLoss, FocalLoss, WeightedCrossEntropyLoss
from models import DeepLabV3PlusCBAM
from utils.data import CLASSES, IGNORE_INDEX, NUM_CLASSES, build_dataloaders
from utils.metrics import SegMetric


def _config_defaults(path: Path | None) -> dict:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config must contain a JSON object.")
    for key in ("data_root", "work_dir", "split_dir", "resume"):
        value = raw.get(key)
        if value and not Path(value).is_absolute():
            raw[key] = (MODULE_ROOT / value).resolve()
    return raw


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    known, _ = pre.parse_known_args(argv)
    defaults = _config_defaults(known.config)

    p = argparse.ArgumentParser(
        description="Train stage-two DeepLabV3+ attention ablations on CamVid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, help="JSON preset; explicit CLI options override it")
    p.add_argument("--data-root", type=Path, default=MODULE_ROOT.parent)
    p.add_argument("--work-dir", type=Path, default=MODULE_ROOT / "checkpoints" / "default")
    p.add_argument("--split-protocol", choices=["internal-val", "official-test"], default="internal-val")
    p.add_argument("--split-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--crop-size", type=int, nargs=2, metavar=("H", "W"), default=(360, 480))
    p.add_argument("--loss", choices=["ce", "dice", "focal", "ce+dice", "ce+focal"], default="ce+dice")
    p.add_argument("--attention", choices=["cbam", "eca", "coord", "none"], default="cbam")
    p.add_argument("--cbam", choices=["full", "channel", "spatial", "none"], default="full")
    p.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    p.add_argument("--scheduler", choices=["cosine", "poly"], default="cosine")
    p.add_argument("--no-pretrain", action="store_true")
    p.add_argument("--resume", type=Path)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    valid_dests = {action.dest for action in p._actions}
    p.set_defaults(**{key: value for key, value in defaults.items() if key in valid_dests})
    return p.parse_args(argv)


def build_loss(name: str, class_weights: torch.Tensor, device: torch.device) -> nn.Module:
    weights = class_weights.to(device)
    ce = WeightedCrossEntropyLoss(weights, ignore_index=IGNORE_INDEX)
    dice = DiceLoss(ignore_index=IGNORE_INDEX)
    focal = FocalLoss(gamma=2.0, alpha=weights, ignore_index=IGNORE_INDEX)
    if name == "ce":
        return ce
    if name == "dice":
        return dice
    if name == "focal":
        return focal
    first, second = (ce, dice) if name == "ce+dice" else (ce, focal)

    class CombinedLoss(nn.Module):
        def forward(self, logits, targets):
            return 0.5 * first(logits, targets) + 0.5 * second(logits, targets)

    return CombinedLoss()


def build_model(attention: str, cbam_variant: str, pretrained: bool) -> DeepLabV3PlusCBAM:
    return DeepLabV3PlusCBAM(
        num_classes=NUM_CLASSES,
        attention_type=attention,
        cbam_use_channel=cbam_variant in ("full", "channel"),
        cbam_use_spatial=cbam_variant in ("full", "spatial"),
        pretrained_backbone=pretrained,
    )


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type=device.type):
                loss = loss_fn(model(images), targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = loss_fn(model(images), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, device) -> dict:
    model.eval()
    metric = SegMetric(NUM_CLASSES, ignore_index=IGNORE_INDEX)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        metric.update(model(images).argmax(dim=1), targets)
    return metric.compute()


def _checkpoint_state(epoch, model, optimizer, scheduler, best_miou, val_results, args) -> dict:
    serial_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    return {
        "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "best_miou": best_miou,
        "val_results": val_results, "args": serial_args,
    }


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None) -> tuple[int, float, dict]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint.get("epoch", -1) + 1, checkpoint.get("best_miou", 0.0), checkpoint


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )

    train_loader, val_loader = build_dataloaders(
        args.data_root, split_protocol=args.split_protocol, split_dir=args.split_dir,
        batch_size=args.batch_size, val_batch_size=1, num_workers=args.workers,
        crop_size=tuple(args.crop_size),
    )
    print(f"device={device} protocol={args.split_protocol} train={len(train_loader.dataset)} eval={len(val_loader.dataset)}")
    class_weights = train_loader.dataset.compute_class_weights("inv_freq")
    model = build_model(args.attention, args.cbam, not args.no_pretrain).to(device)
    loss_fn = build_loss(args.loss, class_weights, device)
    groups = model.param_groups(lr=args.lr, backbone_lr_scale=0.1)
    optimizer = AdamW(groups, weight_decay=args.wd) if args.optimizer == "adamw" else SGD(
        groups, momentum=0.9, weight_decay=args.wd, nesterov=True
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6) if args.scheduler == "cosine" else PolynomialLR(
        optimizer, total_iters=args.epochs, power=0.9
    )
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None
    start_epoch, best_miou = 0, 0.0
    if args.resume:
        start_epoch, best_miou, _ = load_checkpoint(args.resume, model, optimizer, scheduler)

    log_path = args.work_dir / "train_log.csv"
    new_log = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_log:
            writer.writerow(["epoch", "train_loss", "miou", "pa", *[f"iou_{name}" for name in CLASSES]])
        for epoch in range(start_epoch, args.epochs):
            started = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
            scheduler.step()
            results = validate(model, val_loader, device)
            miou = float(results["miou"])
            is_best = miou > best_miou
            if is_best:
                best_miou = miou
            state = _checkpoint_state(epoch, model, optimizer, scheduler, best_miou, results, args)
            torch.save(state, args.work_dir / "last.pth")
            if is_best:
                torch.save(state, args.work_dir / "best.pth")
            writer.writerow([epoch + 1, train_loss, miou, results["pa"], *results["iou_per_class"]])
            handle.flush()
            print(
                f"epoch={epoch + 1}/{args.epochs} loss={train_loss:.4f} "
                f"mIoU={miou * 100:.2f}% PA={results['pa'] * 100:.2f}% "
                f"best={best_miou * 100:.2f}% time={time.time() - started:.1f}s"
            )


if __name__ == "__main__":
    main()
