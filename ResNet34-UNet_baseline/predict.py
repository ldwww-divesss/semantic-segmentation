import torch
import matplotlib.pyplot as plt
import numpy as np
from dataset import SegDataset

# 【关键修改 1】导入新的网络模型类 ResNetUNet
from model import ResNetUNet 

device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 12

print("正在加载测试集...")
dataset = SegDataset("/mnt/workspace/semantic_segmentation/语义分割", train=False)

# 【关键修改 2】使用新的 ResNetUNet，并加载 best_unet.pth
print("正在加载模型...")
model = ResNetUNet(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load("unet.pth", map_location=device))
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

# 【关键修改 4】适配你的云端环境，直接存为图片
save_path = "result.png"
plt.savefig(save_path, bbox_inches='tight', dpi=300)
print(f"✅ 预测完成！结果已成功保存为 {save_path}，请在左侧文件树中双击查看。")