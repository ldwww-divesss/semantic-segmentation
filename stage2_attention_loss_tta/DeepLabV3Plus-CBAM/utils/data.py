from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

CLASSES = (
    "Sky", "Building", "Pole", "Road", "Pavement", "Tree",
    "SignSymbol", "Fence", "Car", "Pedestrian", "Bicyclist",
)
NUM_CLASSES = len(CLASSES)
VOID_ID = 11
IGNORE_INDEX = VOID_ID
PALETTE = [
    (128, 128, 128), (128, 0, 0), (192, 192, 128), (128, 64, 128),
    (0, 0, 192), (128, 128, 0), (192, 128, 128), (64, 64, 128),
    (64, 0, 128), (64, 64, 0), (0, 128, 192),
]

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
MODULE_ROOT = Path(__file__).resolve().parents[1]
BUILTIN_SPLIT_DIR = MODULE_ROOT / "configs" / "splits"


@dataclass(frozen=True)
class DatasetLayout:
    name: str
    train_images: Path
    train_labels: Path
    test_images: Path | None = None
    test_labels: Path | None = None
    internal_val_images: Path | None = None
    internal_val_labels: Path | None = None


def discover_layout(root: str | Path) -> DatasetLayout:
    root = Path(root).expanduser().resolve()
    if (root / "train" / "images").is_dir() and (root / "train" / "labels").is_dir():
        return DatasetLayout(
            "train-test", root / "train" / "images", root / "train" / "labels",
            root / "test" / "images", root / "test" / "labels",
        )
    if (root / "images" / "train").is_dir() and (root / "annotations" / "train").is_dir():
        return DatasetLayout(
            "images-annotations-splits",
            root / "images" / "train", root / "annotations" / "train",
            root / "images" / "test", root / "annotations" / "test",
            root / "images" / "val", root / "annotations" / "val",
        )
    raise FileNotFoundError(
        f"Unsupported CamVid layout at {root}. Expected train/images + train/labels "
        "or images/train + annotations/train."
    )


def _read_stems(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _pairs(image_dir: Path, label_dir: Path, stems: list[str] | None = None) -> list[tuple[Path, Path]]:
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Missing image/label directories: {image_dir}, {label_dir}")
    stems = stems or sorted(path.stem for path in image_dir.glob("*.png"))
    samples = [(image_dir / f"{stem}.png", label_dir / f"{stem}.png") for stem in stems]
    missing = [str(path) for pair in samples for path in pair if not path.is_file()]
    if missing:
        raise FileNotFoundError("Split references missing files:\n" + "\n".join(missing[:5]))
    return samples


def resolve_samples(
    root: str | Path,
    split_protocol: str,
    *,
    split_dir: str | Path | None = None,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], DatasetLayout]:
    layout = discover_layout(root)
    split_dir = Path(split_dir) if split_dir else BUILTIN_SPLIT_DIR
    if split_protocol == "internal-val":
        train_stems = _read_stems(split_dir / "train.txt")
        val_stems = _read_stems(split_dir / "val.txt")
        if layout.internal_val_images is not None:
            train = _pairs(layout.train_images, layout.train_labels)
            val = _pairs(layout.internal_val_images, layout.internal_val_labels)
            if {p.stem for p, _ in train} != set(train_stems) or {p.stem for p, _ in val} != set(val_stems):
                raise ValueError("Dataset train/val directories do not match the bundled 294/73 manifests.")
        else:
            train = _pairs(layout.train_images, layout.train_labels, train_stems)
            val = _pairs(layout.train_images, layout.train_labels, val_stems)
        return train, val, layout
    if split_protocol == "official-test":
        if layout.test_images is None or layout.test_labels is None:
            raise FileNotFoundError("official-test requires test images and labels.")
        train = _pairs(layout.train_images, layout.train_labels)
        if layout.internal_val_images is not None:
            train += _pairs(layout.internal_val_images, layout.internal_val_labels)
        return train, _pairs(layout.test_images, layout.test_labels), layout
    raise ValueError(f"Unknown split protocol: {split_protocol}")


def build_transforms(train: bool, size: tuple[int, int]):
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError as exc:
        raise ImportError("Install albumentations to use the default transforms.") from exc
    h, w = size
    if train:
        pad_kwargs = dict(min_height=h, min_width=w, border_mode=0)
        major_version = int(A.__version__.split(".", 1)[0])
        if major_version >= 2:
            pad = A.PadIfNeeded(**pad_kwargs, fill=0, fill_mask=VOID_ID)
        else:
            pad = A.PadIfNeeded(**pad_kwargs, value=0, mask_value=VOID_ID)
        return A.Compose([
            A.RandomScale(scale_limit=(-0.5, 0.5), p=1.0), pad,
            A.RandomCrop(height=h, width=w), A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
            A.Normalize(mean=_MEAN, std=_STD), ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(height=h, width=w), A.Normalize(mean=_MEAN, std=_STD), ToTensorV2(),
    ])


class CamVidDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, Path]], transforms: Callable | None = None) -> None:
        self.samples = samples
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_path, mask_path = self.samples[index]
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        if self.transforms is not None:
            item = self.transforms(image=image, mask=mask)
            image_tensor, mask_tensor = item["image"].float(), item["mask"].long()
        else:
            image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(mask.copy()).long()
        mask_tensor[(mask_tensor < 0) | (mask_tensor > VOID_ID)] = IGNORE_INDEX
        return {"image": image_tensor, "mask": mask_tensor, "stem": image_path.stem}

    def compute_class_weights(self, method: str = "inv_freq") -> torch.Tensor:
        counts = np.zeros(NUM_CLASSES, dtype=np.float64)
        for _, mask_path in self.samples:
            mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
            counts += np.bincount(mask[mask < NUM_CLASSES], minlength=NUM_CLASSES)
        freq = counts / max(counts.sum(), 1.0)
        if method == "inv_freq":
            nonzero = freq[freq > 0]
            median = float(np.median(nonzero)) if nonzero.size else 1.0
            weights = np.where(freq > 0, median / (freq + 1e-12), 0.0)
        elif method == "inv_log":
            weights = 1.0 / np.log(1.02 + freq)
        else:
            raise ValueError(f"Unknown class-weight method: {method}")
        return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(
    data_root: str | Path,
    *,
    split_protocol: str = "internal-val",
    split_dir: str | Path | None = None,
    batch_size: int = 4,
    val_batch_size: int = 1,
    num_workers: int = 4,
    crop_size: tuple[int, int] = (360, 480),
    train_transforms=None,
    val_transforms=None,
) -> tuple[DataLoader, DataLoader]:
    train_samples, val_samples, _ = resolve_samples(data_root, split_protocol, split_dir=split_dir)
    train_ds = CamVidDataset(train_samples, train_transforms or build_transforms(True, crop_size))
    val_ds = CamVidDataset(val_samples, val_transforms or build_transforms(False, crop_size))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=len(train_ds) >= batch_size,
    )
    val_loader = DataLoader(
        val_ds, batch_size=val_batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
