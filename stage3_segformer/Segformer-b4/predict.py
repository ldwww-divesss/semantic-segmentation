"""
Stage 3 预测脚本:
  1. 单模型: Input | GT | SegFormer-B4 (存为 result_segformer.png)
  2. 三模型对比: Input | GT | UNet | UNet-CBAM | SegFormer-B4 (4x5 grid, 存为 result_compare_segformer.png)
"""
import sys
import os
import importlib.util

import torch
import numpy as np
import matplotlib.pyplot as plt

# ---- 阻断 Stage1/2 ResNet34 预训练下载 (仅影响 stage1/stage2 模型) ----
import torch.hub
_original_download = torch.hub.download_url_to_file
def _skip(url, dst, *a, **kw):
    raise RuntimeError(f"Download blocked: {url}")
torch.hub.download_url_to_file = _skip

import torchvision.models
_orig_resnet34 = torchvision.models.resnet34
def _no_dl(*a, **kw):
    kw["weights"] = None
    return _orig_resnet34(*a, **kw)
torchvision.models.resnet34 = _no_dl


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 12

# CamVid academic color palette
CAMVID_COLORS = np.array([
    [128, 128, 128],  # 0: Sky
    [128, 0, 0],      # 1: Building
    [192, 192, 128],  # 2: Pole
    [128, 64, 128],   # 3: Road
    [0, 0, 192],      # 4: Pavement
    [128, 128, 0],    # 5: Tree
    [192, 128, 128],  # 6: SignSymbol
    [64, 64, 128],    # 7: Fence
    [64, 0, 128],     # 8: Car
    [64, 64, 0],      # 9: Pedestrian
    [0, 128, 192],    # 10: Bicyclist
    [0, 0, 0],        # 11: Void
], dtype=np.uint8)

CLASS_NAMES = [
    "Sky", "Building", "Pole", "Road", "Pavement",
    "Tree", "SignSymbol", "Fence", "Car", "Pedestrian",
    "Bicyclist", "Void"
]


def decode_segmap(mask_idx):
    r = np.zeros_like(mask_idx, dtype=np.uint8)
    g = np.zeros_like(mask_idx, dtype=np.uint8)
    b = np.zeros_like(mask_idx, dtype=np.uint8)
    for l in range(NUM_CLASSES):
        idx = mask_idx == l
        r[idx] = CAMVID_COLORS[l, 0]
        g[idx] = CAMVID_COLORS[l, 1]
        b[idx] = CAMVID_COLORS[l, 2]
    return np.stack([r, g, b], axis=2)


def unnormalize(img_tensor):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    return np.clip(img, 0, 1)


# ---- 加载所有模型 ----
print(f"[Device] {DEVICE}")

# Stage 1
_stage1_mod = _load_module("s1_model", "../stage1/model.py")
UNet = _stage1_mod.ResNetUNet
# Stage 2
_stage2_mod = _load_module("s2_model", "../stage2/model.py")
UNetCBAM = _stage2_mod.ResNetUNet
# Stage 3
from model import SegFormerB4

# Dataset (use stage3's own)
from dataset import SegDataset

# Data path: try remote first, fallback to local
for root in ["/mnt/workspace/semantic_segmentation/语义分割", "../data"]:
    if os.path.isdir(os.path.join(root, "test/images")):
        DATA_ROOT = root
        break
else:
    raise FileNotFoundError("Cannot find test dataset")

print(f"[Data] {DATA_ROOT}")
dataset = SegDataset(DATA_ROOT, train=False)

print("[Model] Loading weights...")
model1 = UNet(num_classes=NUM_CLASSES).to(DEVICE)
model1.load_state_dict(torch.load("../stage1/unet.pth", map_location=DEVICE, weights_only=True))
model1.eval()

model2 = UNetCBAM(num_classes=NUM_CLASSES).to(DEVICE)
model2.load_state_dict(torch.load("../stage2/unet_cbam.pth", map_location=DEVICE, weights_only=True))
model2.eval()

model3 = SegFormerB4(num_classes=NUM_CLASSES).to(DEVICE)
model3.load_state_dict(torch.load("segformer_b4_best.pth", map_location=DEVICE, weights_only=False)["model_state_dict"])
model3.eval()


# ---- 1. 单模型预测 (3 panel) ----
print("\n[1/2] Single model prediction...")
idx = 0
img, mask = dataset[idx]

with torch.no_grad():
    pred3 = model3(img.unsqueeze(0).to(DEVICE)).argmax(1).squeeze().cpu().numpy()

mask_np = mask.squeeze().cpu().numpy() if hasattr(mask, "cpu") else mask
if mask_np.ndim == 3:
    mask_np = mask_np.squeeze()

img_rgb = unnormalize(img)
gt_rgb = decode_segmap(mask_np)
pred3_rgb = decode_segmap(pred3)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, data, title in [
    (axes[0], img_rgb, "Input"),
    (axes[1], gt_rgb, "Ground Truth"),
    (axes[2], pred3_rgb, "SegFormer-B4"),
]:
    ax.imshow(data)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
plt.tight_layout()
plt.savefig("result_segformer.png", bbox_inches="tight", dpi=200)
print("  Saved result_segformer.png")


# ---- 2. 三模型对比 (4x5 grid) ----
print("\n[2/2] 3-model comparison...")
indices = [0, 1, 2, 3]
rows_data = []

with torch.no_grad():
    for i in indices:
        img, mask = dataset[i]
        batch = img.unsqueeze(0).to(DEVICE)

        p1 = model1(batch).argmax(1).squeeze().cpu().numpy()
        p2 = model2(batch).argmax(1).squeeze().cpu().numpy()
        p3 = model3(batch).argmax(1).squeeze().cpu().numpy()

        mask_np = mask.squeeze().cpu().numpy() if hasattr(mask, "cpu") else mask
        if mask_np.ndim == 3:
            mask_np = mask_np.squeeze()

        rows_data.append((
            unnormalize(img),
            decode_segmap(mask_np),
            decode_segmap(p1),
            decode_segmap(p2),
            decode_segmap(p3),
        ))

fig, axes = plt.subplots(4, 5, figsize=(20, 16))
col_titles = ["Input", "Ground Truth", "ResNet34-UNet", "ResNet-UNet-CBAM", "SegFormer-B4"]

for row_idx in range(4):
    for col_idx in range(5):
        axes[row_idx, col_idx].imshow(rows_data[row_idx][col_idx])
        axes[row_idx, col_idx].axis("off")
        if row_idx == 0:
            axes[0, col_idx].set_title(col_titles[col_idx], fontsize=12, fontweight="bold", pad=8)
    axes[row_idx, 0].set_ylabel(f"#{indices[row_idx]+1}", fontsize=12,
                                 fontweight="bold", rotation=0, labelpad=15, va="center")

# Legend
legend_els = []
for cls_id in range(11):
    color = CAMVID_COLORS[cls_id] / 255.0
    legend_els.append(plt.Line2D([0], [0], marker="s", color="w",
                                  markerfacecolor=color, markersize=10,
                                  label=CLASS_NAMES[cls_id]))
fig.legend(handles=legend_els, loc="lower center", ncol=6, fontsize=8, frameon=False)

plt.subplots_adjust(wspace=0.02, hspace=0.02, bottom=0.07)
plt.suptitle("CamVid: CNN vs Transformer", fontsize=16, fontweight="bold", y=0.98)
plt.savefig("result_compare_segformer.png", bbox_inches="tight", dpi=200)
print("  Saved result_compare_segformer.png")

print("\n[Done]")
