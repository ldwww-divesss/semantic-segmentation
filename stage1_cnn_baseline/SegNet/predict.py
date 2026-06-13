import torch
import matplotlib.pyplot as plt
import numpy as np
from dataset import SegDataset
from model import SegNet

device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 12

# ==========================================
# 定义 CamVid 标准学术调色盘
# ==========================================
CAMVID_COLORS = np.array([
    [128, 128, 128], # 0: Sky (灰)
    [128, 0, 0],     # 1: Building (暗红)
    [192, 192, 128], # 2: Pole / Column_Pole (土黄)
    [128, 64, 128],  # 3: Road (紫)
    [0, 0, 192],     # 4: Pavement / Sidewalk (深蓝)
    [128, 128, 0],   # 5: Tree (墨绿)
    [192, 128, 128], # 6: SignSymbol (浅紫红)
    [64, 64, 128],   # 7: Fence (蓝灰)
    [64, 0, 128],    # 8: Car (深紫)
    [64, 64, 0],     # 9: Pedestrian (棕黄)
    [0, 128, 192],   # 10: Bicyclist (亮蓝)
    [0, 0, 0]        # 11: Void (黑) - 未标注或忽略区域
], dtype=np.uint8)

def decode_segmap(image_index):
    """将单通道的分割索引图转化为三通道的彩色 RGB 图"""
    r = np.zeros_like(image_index, dtype=np.uint8)
    g = np.zeros_like(image_index, dtype=np.uint8)
    b = np.zeros_like(image_index, dtype=np.uint8)

    for l in range(NUM_CLASSES):
        idx = image_index == l
        r[idx] = CAMVID_COLORS[l, 0]
        g[idx] = CAMVID_COLORS[l, 1]
        b[idx] = CAMVID_COLORS[l, 2]

    return np.stack([r, g, b], axis=2)


print("正在加载测试集...")
dataset = SegDataset("/mnt/workspace/semantic_segmentation/语义分割", train=False)

print("正在加载 SegNet 模型...")
# 预测阶段不需要再下载预训练权重，pretrained=False
model = SegNet(num_classes=NUM_CLASSES, pretrained=False).to(device)
model.load_state_dict(torch.load("segnet.pth", map_location=device))
title_name = "SegNet Baseline"
model.eval()

# 从测试集中获取一张图像和标签（可把 0 改成其他数字测试不同的图片）
img, mask = dataset[0]

print("正在进行预测...")
with torch.no_grad():
    pred = model(img.unsqueeze(0).to(device))
    pred = pred.argmax(1).squeeze().cpu().numpy()

if isinstance(mask, torch.Tensor):
    mask = mask.squeeze().cpu().numpy()

# 反标准化图像以便展示
img_show = img.permute(1, 2, 0).cpu().numpy()
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img_show = std * img_show + mean
img_show = np.clip(img_show, 0, 1)

pred_color = decode_segmap(pred)
mask_color = decode_segmap(mask)

# 绘图设置
plt.figure(figsize=(18, 6))

plt.subplot(131)
plt.imshow(img_show)
plt.title("Image (Un-normalized)", fontsize=14, pad=10)
plt.axis('off')

plt.subplot(132)
plt.imshow(mask_color)
plt.title("Ground Truth (Label)", fontsize=14, pad=10)
plt.axis('off')

plt.subplot(133)
plt.imshow(pred_color)
plt.title(f"Prediction ({title_name})", fontsize=14, pad=10)
plt.axis('off')

plt.tight_layout()

save_path = "result_segnet.png"
plt.savefig(save_path, bbox_inches='tight', dpi=300)
print(f"✅ 预测完成！结果已成功保存为 {save_path}。")
