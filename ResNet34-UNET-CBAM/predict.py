# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# from dataset import SegDataset
# from model_cbam import ResNetUNetCBAM

# device = "cuda" if torch.cuda.is_available() else "cpu"
# NUM_CLASSES = 12

# print("加载测试集...")
# dataset = SegDataset("/mnt/workspace/semantic_segmentation/语义分割", train=False)

# print("加载 CBAM 增强版模型权重...")
# model = ResNetUNetCBAM(num_classes=NUM_CLASSES).to(device)
# model.load_state_dict(torch.load("best_unet_cbam.pth", map_location=device))
# model.eval()

# # 可随意更改 dataset[0] 中的数字，测试不同的验证集图片
# img, mask = dataset[0]

# with torch.no_grad():
#     pred = model(img.unsqueeze(0).to(device))
#     pred = pred.argmax(1).squeeze().cpu().numpy()

# mask = mask.squeeze().cpu().numpy() if isinstance(mask, torch.Tensor) else mask

# # 图像反标准化，恢复正常彩色显示
# img_show = img.permute(1, 2, 0).cpu().numpy()
# mean = np.array([0.485, 0.456, 0.406])
# std = np.array([0.229, 0.224, 0.225])
# img_show = std * img_show + mean
# img_show = np.clip(img_show, 0, 1)

# # 生成并排对比图
# plt.figure(figsize=(15, 5))

# plt.subplot(131)
# plt.imshow(img_show)
# plt.title("Original Image")
# plt.axis('off')

# plt.subplot(132)
# plt.imshow(mask, cmap='nipy_spectral', vmin=0, vmax=11)
# plt.title("Ground Truth")
# plt.axis('off')

# plt.subplot(133)
# plt.imshow(pred, cmap='nipy_spectral', vmin=0, vmax=11)
# plt.title("Prediction (CBAM Enhanced)")
# plt.axis('off')

# plt.tight_layout()
# save_path = "result_cbam.png"
# plt.savefig(save_path, bbox_inches='tight', dpi=300)
# print(f"✅ 阶段二可视化结果已保存至: {save_path}，请用于报告提交。")


import torch
import matplotlib.pyplot as plt
import numpy as np
from dataset import SegDataset
from model_cbam import ResNetUNetCBAM


device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 12

print("加载测试集...")
dataset = SegDataset("/mnt/workspace/semantic_segmentation/语义分割", train=False)

print("加载 CBAM 增强版模型权重...")
model = ResNetUNetCBAM(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load("best_unet_cbam.pth", map_location=device))
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

# 【关键修改 3】反标准化图像以便展示
# 因为我们在 dataset.py 里对图像做了 Normalize，这里要还原，否则颜色会发黑/发绿
img_show = img.permute(1, 2, 0).cpu().numpy()
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img_show = std * img_show + mean
img_show = np.clip(img_show, 0, 1) # 强制截断到 0~1 之间，防止 Matplotlib 报错

# 绘图设置
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(img_show)
plt.title("Image (Un-normalized)")
plt.axis('off')

plt.subplot(132)
plt.imshow(mask)
plt.title("Ground Truth (Label)")
plt.axis('off')

plt.subplot(133)
plt.imshow(pred)
plt.title("Prediction (ResNet-UNet)")
plt.axis('off')



plt.tight_layout()
save_path = "result_cbam.png"
plt.savefig(save_path, bbox_inches='tight', dpi=300)
print(f"✅ 阶段二可视化结果已保存至: {save_path}，请用于报告提交。")