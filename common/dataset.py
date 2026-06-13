import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

CAMVID_CLASSES = [
    'Sky', 'Building', 'Pole', 'Road', 'Pavement',
    'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist',
]
NUM_CLASSES = 11
IGNORE_INDEX = 255

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class CamVidDataset(Dataset):
    """CamVid dataset with index-based labels (0-10 = classes, 11 = void → 255)."""

    def __init__(self, img_dir: str, label_dir: str, transform=None):
        self.img_dir   = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.filenames = sorted(f for f in os.listdir(img_dir) if f.endswith('.png'))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img   = np.array(Image.open(os.path.join(self.img_dir,   fname)).convert('RGB'))
        label = np.array(Image.open(os.path.join(self.label_dir, fname)))  # uint8, 0-11

        label = label.copy()
        label[label == 11] = IGNORE_INDEX  # void → ignore

        if self.transform:
            out   = self.transform(image=img, mask=label)
            img   = out['image']           # float32 tensor (3, H, W)
            label = out['mask'].long()     # int64 tensor (H, W)

        return img, label, fname


def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.3, border_mode=cv2.BORDER_REFLECT_101),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transform():
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
