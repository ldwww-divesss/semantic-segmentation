# CamVid 城市街景语义分割

## 数据集

本项目使用 **CamVid**（Cambridge-driving Labeled Video Database），11 类城市街景语义分割。

```
训练集：367 张  |  测试集：101 张  |  分辨率：360×480
```
数据结构：
```
semantic-segmentation/
├── train/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## 环境配置

```bash
conda activate venv
pip install -r requirements.txt
```

主要依赖：`torch >= 2.0`、`segmentation-models-pytorch`、`albumentations`

## 标注说明

labels 为灰度图，像素值即类别 ID：

| ID | 类别 | ID | 类别 |
|----|------|----|------|
| 0 | Sky（天空） | 6 | SignSymbol（标牌） |
| 1 | Building（建筑） | 7 | Fence（围栏） |
| 2 | Pole（灯柱） | 8 | Car（车辆） |
| 3 | Road（道路） | 9 | Pedestrian（行人） |
| 4 | Pavement（人行道） | 10 | Bicyclist（骑行者） |
| 5 | Tree（树木） | 11 | Void → 忽略 |

## 使用方法

### 训练

```bash
# 阶段一 Baseline：U-Net
python train.py --model unet --epochs 100 --batch-size 8 --device auto

# 阶段一 Baseline：DeepLabV3+
python train.py --model deeplabv3plus --epochs 80 --batch-size 8 --device auto

# CPU 快速验证（2 epoch）
python train.py --model unet --epochs 2 --batch-size 2 --device cpu
```

`--device auto` 自动选择 CUDA → MPS → CPU。

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `unet` | 架构：`unet` / `deeplabv3plus` |
| `--encoder` | 自动 | Backbone（unet→resnet34，deeplabv3plus→resnet50） |
| `--epochs` | 100 | 训练轮数 |
| `--batch-size` | 8 | Batch size |
| `--lr` | 1e-4 | 学习率（AdamW） |
| `--patience` | 15 | Early stopping patience |
| `--device` | auto | cuda / mps / cpu / auto |
| `--data-root` | `.` | 数据集根目录 |
| `--save-dir` | `checkpoints/` | 模型保存目录 |

最优模型自动保存至 `checkpoints/<model>_best.pth`，训练结束输出 per-class IoU。

## 实验进展

| 阶段 | 内容 | 状态 |
|------|------|------|
| 阶段一 | U-Net + DeepLabV3+ Baseline | 进行中 |
| 阶段二 | CBAM 注意力 + 损失函数优化 | 待开始 |
| 阶段三 | SegFormer 跨框架对比 | 待开始 |
| 阶段四 | CNN-Transformer 融合创新 | 待开始 |

## 评价指标

- **mIoU**：11 类平均交并比（排除 void 类）
- **Pixel Accuracy**：像素准确率

## 项目结构

```
├── dataset.py      # CamVid 数据集加载与数据增强
├── metrics.py      # mIoU、Pixel Accuracy 计算（混淆矩阵）
├── train.py        # 训练主脚本
├── requirements.txt
└── README.md
```

## ResNet34-UNet_baseline
Device  : cuda
Train   : 367 images     |  Val: 101 images
Model   : ResNet34-UNet  |  Classes: 12          

=== Best checkpoint (epoch 37) ===
+------------+-------+-------+
|    Class   |  IoU  |  Acc  |
+------------+-------+-------+
|        Sky | 94.20 | 96.64 |
|   Building | 86.47 | 89.35 |
|       Pole | 18.07 | 24.94 |
|       Road | 97.18 | 98.32 |
|   Pavement | 88.68 | 95.92 |
|       Tree | 90.82 | 97.66 |
| SignSymbol | 57.50 | 68.90 |
|      Fence | 72.45 | 88.33 |
|        Car | 86.26 | 90.28 |
| Pedestrian | 55.55 | 77.02 |
|  Bicyclist | 80.05 | 84.65 |
|       Void | 30.25 | 61.49 |
+------------+-------+-------+

Epoch  |TrainLoss  |  mIoU  |    PA   |   Time                                                                                                
  37   | 0.3603    | 71.46% |  93.33% |  8.3s    

|Method	               | mIoU   |	PA |  FPS	  |Params
Stage 1 UNet (ResNet34)| 71.46% |93.33%|	168.25|	24.40M

## ResNet34-UNET-CBAM
Device  : cuda
Train   : 367 images | Val: 101 images    

Epoch  |TrainLoss  |  mIoU  |    PA   |   Time                                                                                                
  25   | 0.4279    | 71.60% |  93.50% |  9.6s    

=== Best checkpoint (epoch 25) ===
+------------+-------+-------+
|    Class   |  IoU  |  Acc  |
+------------+-------+-------+
|        Sky | 94.31 | 97.03 |
|   Building | 87.39 | 90.80 |
|       Pole | 16.23 | 20.29 |
|       Road | 97.00 | 98.07 |
|   Pavement | 88.18 | 96.10 |
|       Tree | 91.35 | 96.97 |
| SignSymbol | 59.82 | 67.77 |
|      Fence | 71.86 | 84.46 |
|        Car | 85.80 | 93.84 |
| Pedestrian | 57.32 | 74.01 |
|  Bicyclist | 79.88 | 86.56 |
|       Void | 30.07 | 61.46 |
+------------+-------+-------+


|Method	               | mIoU   |	PA |  FPS	  |Params
Stage 2 UNet-CBAM      | 71.60% |93.50%|	91.73|	24.44M



