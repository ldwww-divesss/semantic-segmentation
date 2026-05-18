import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random

class SegDataset(Dataset):
    def __init__(self, root, train=True):
        self.train = train
        self.img_dir = os.path.join(root, "train/images" if train else "test/images")
        self.mask_dir = os.path.join(root, "train/labels" if train else "test/labels")
        self.files = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name))

        # 统一缩放尺寸 (Mask 必须使用最近邻插值 NEAREST 以保持类别标签不被破坏)
        img = img.resize((512, 512), Image.BILINEAR)
        mask = mask.resize((512, 512), Image.NEAREST) 

        # 训练集启用联合数据增强：随机水平翻转
        if self.train and random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # 转换为 Tensor 并执行 ImageNet 标准化
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        mask = np.array(mask)
        

        
        return img, torch.tensor(mask, dtype=torch.long)