import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random


class SegDataset(Dataset):
    """CamVid 数据集加载 (增强版, 同 Stage 2)

    root 目录结构:
        root/train/images/   root/train/labels/
        root/test/images/    root/test/labels/
    标签: 单通道索引 PNG, 值 0-11 (11=Void)
    """

    def __init__(self, root, train=True):
        self.train = train
        sub = "train" if train else "test"
        self.img_dir = os.path.join(root, sub, "images")
        self.mask_dir = os.path.join(root, sub, "labels")
        self.files = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name))

        # 1. Resize to 512
        img = img.resize((512, 512), Image.BILINEAR)
        mask = mask.resize((512, 512), Image.NEAREST)

        # 2. Augmentation (train only)
        if self.train:
            # Horizontal flip
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

            # Rotation +-10 deg
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)
                img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

            # Random scale crop
            if random.random() > 0.5:
                scale = random.uniform(0.5, 2.0)
                new_w = int(512 * scale)
                new_h = int(512 * scale)
                img = TF.resize(img, (new_h, new_w), interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.resize(mask, (new_h, new_w), interpolation=TF.InterpolationMode.NEAREST)
                if new_h >= 512 and new_w >= 512:
                    top = random.randint(0, new_h - 512)
                    left = random.randint(0, new_w - 512)
                    img = TF.crop(img, top, left, 512, 512)
                    mask = TF.crop(mask, top, left, 512, 512)
                else:
                    img = TF.resize(img, (512, 512), interpolation=TF.InterpolationMode.BILINEAR)
                    mask = TF.resize(mask, (512, 512), interpolation=TF.InterpolationMode.NEAREST)

            # Color jitter
            if random.random() > 0.5:
                brightness = random.uniform(0.7, 1.3)
                contrast = random.uniform(0.7, 1.3)
                saturation = random.uniform(0.7, 1.3)
                hue = random.uniform(-0.1, 0.1)
                img = TF.adjust_brightness(img, brightness)
                img = TF.adjust_contrast(img, contrast)
                img = TF.adjust_saturation(img, saturation)
                img = TF.adjust_hue(img, hue)

        # 3. ToTensor + Normalize (ImageNet)
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])

        mask = np.array(mask)
        return img, torch.tensor(mask, dtype=torch.long)
