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
