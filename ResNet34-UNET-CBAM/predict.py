import torch
import matplotlib.pyplot as plt
import numpy as np
from dataset import SegDataset

# 【新增】导入所有的网络模型类
# from model import ResNetUNet
# from model import ResNetUNetCBAM
# from model_aspp import ResNetUNetASPP
from model import ResNetUNet

device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 12

# ==========================================
# 【核心控制开关】在这里修改你要预测的阶段！
# 可选值: "Baseline", "CBAM", "ASPP"
# ==========================================
STAGE = "CBAM"

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
    """
    将单通道的分割索引图转化为三通道的彩色 RGB 图
    """
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

print(f"正在加载 {STAGE} 模型...")
# 【关键修改】根据选定的 STAGE 自动加载对应的模型和权重
if STAGE == "Baseline":
    model = ResNetUNet(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load("unet.pth", map_location=device))
    title_name = "ResNet-UNet Baseline"
elif STAGE == "CBAM":
    model = ResNetUNet(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load("unet_cbam.pth", map_location=device))
    title_name = "ResNet-UNet + CBAM"
elif STAGE == "ASPP":
    model = ResNetUNetASPP(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load("best_unet_aspp.pth", map_location=device))
    title_name = "ResNet-UNet + ASPP"
else:
    raise ValueError("未知的 STAGE，请检查拼写！")

model.eval()

# 从验证集中获取一张图像和标签（你可以把 0 改成其他数字测试不同的图片）
img, mask = dataset[0]

print("正在进行预测...")
with torch.no_grad():
    # img 增加 batch 维度变为 [1, C, H, W] 并传入模型
    pred = model(img.unsqueeze(0).to(device))
    # 获取预测的类别索引，并转为 numpy 数组
    pred = pred.argmax(1).squeeze().cpu().numpy()

# 确保 mask 的维度正确，并转为 numpy
if isinstance(mask, torch.Tensor):
    mask = mask.squeeze().cpu().numpy() 

# 反标准化图像以便展示
img_show = img.permute(1, 2, 0).cpu().numpy()
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img_show = std * img_show + mean
img_show = np.clip(img_show, 0, 1) # 强制截断到 0~1 之间

# 将索引图转化为标准学术 RGB 图
pred_color = decode_segmap(pred)
mask_color = decode_segmap(mask)

# 绘图设置
plt.figure(figsize=(18, 6)) # 稍微加宽画布以容纳更清晰的细节

plt.subplot(131)
plt.imshow(img_show)
plt.title("Image (Un-normalized)", fontsize=14, pad=10)
plt.axis('off')

plt.subplot(132)
plt.imshow(mask_color)
plt.title("Ground Truth (Label)", fontsize=14, pad=10)
plt.axis('off')

plt.subplot(133)
# 【动态修改】标题会自动变成你当前的阶段
plt.imshow(pred_color)
plt.title(f"Prediction ({title_name})", fontsize=14, pad=10)
plt.axis('off')

plt.tight_layout()

# 【动态修改】保存的图片名字也会自动变成 result_cbam.png, result_aspp.png 等
save_path = f"result_{STAGE.lower()}.png"
plt.savefig(save_path, bbox_inches='tight', dpi=300)
print(f"✅ 预测完成！结果已成功保存为 {save_path}，请在左侧文件树中双击查看。")