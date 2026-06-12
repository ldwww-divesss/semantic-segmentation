# CamVid 城市街景语义分割（四阶段研究 + AGFNet）

四阶段实验研究:CNN Baseline → CNN 改进(CBAM/损失/TTA)→ SegFormer 跨框架对比 → CNN-Transformer 融合(AGFNet)。
最终结果:AGFNet **80.35% mIoU / 96.14% PA**;三模型集成 + flip TTA **80.83% mIoU / 96.27% PA**。

## 数据集

本项目使用 **CamVid**(Cambridge-driving Labeled Video Database),11 类城市街景语义分割。

```
训练集:367 张  |  测试集:101 张  |  分辨率:360×480
```
数据结构:
```
semantic_segmentation/
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

主要依赖:`torch >= 2.0`、`segmentation-models-pytorch`、`transformers==4.49.0`(MiT-B2,新版本会重命名 SegFormer 权重键导致 checkpoint 加载失败)、`albumentations`。

## 标注说明

labels 为灰度图,像素值即类别 ID:

| ID | 类别 | ID | 类别 |
|----|------|----|------|
| 0 | Sky(天空) | 6 | SignSymbol(标牌) |
| 1 | Building(建筑) | 7 | Fence(围栏) |
| 2 | Pole(灯柱) | 8 | Car(车辆) |
| 3 | Road(道路) | 9 | Pedestrian(行人) |
| 4 | Pavement(人行道) | 10 | Bicyclist(骑行者) |
| 5 | Tree(树木) | 11 | Void → 忽略 |

## 训练

### 阶段一:CNN Baseline

```bash
# U-Net (ResNet34)
python train.py --model unet --epochs 100 --batch-size 8 --device auto

# DeepLabV3+ (ResNet50)
python train.py --model deeplabv3plus --epochs 80 --batch-size 8 --device auto

# CPU 快速验证(2 epoch)
python train.py --model unet --epochs 2 --batch-size 2 --device cpu
```

`--device auto` 自动选择 CUDA → MPS → CPU。最优模型自动保存至 `checkpoints/<model>_best.pth`。

### 阶段三:SegFormer

```bash
python train_segformer.py --variant b2 --epochs 100
```

### 阶段四:CNN-Transformer 融合(AGFNet)

```bash
# v1:单尺度 Transformer 特征
python train_fusion.py

# v2:多尺度 MiT-B2 + FPN,三种融合方式
python train_fusion_v2.py --fusion agfm         # 自适应门控融合(最终模型)
python train_fusion_v2.py --fusion none_concat  # 拼接融合(消融)
python train_fusion_v2.py --fusion none_add     # 相加融合(消融)
```

## 评测(TTA 与集成)

`evaluate_tta.py` 对单个 Stage IV checkpoint 做带 TTA 的独立评测,`evaluate_ensemble.py` 对三个融合变体做概率级集成:

```bash
# 单模型,无 TTA(复现训练时评测)
python evaluate_tta.py --checkpoint checkpoints/fusion_v2_agfm_best.pth \
    --fusion agfm --tta none --save-json results_eval/agfm_notta.json

# 单模型 + 水平翻转 TTA(2 视图)
python evaluate_tta.py --checkpoint checkpoints/fusion_v2_agfm_best.pth \
    --fusion agfm --tta flip --save-json results_eval/agfm_flip.json

# 单模型 + 多尺度翻转 TTA(6 视图,实测掉点,见论文分析)
python evaluate_tta.py --checkpoint checkpoints/fusion_v2_agfm_best.pth \
    --fusion agfm --tta ms-flip --scales 0.75 1.0 1.25

# 三模型集成(AGFM + Concat + Add)+ 翻转 TTA
python evaluate_ensemble.py --save-json results_eval/ensemble_flip.json
```

### 评测结果(同一评测流水线,Apple MPS)

| 配置 | 视图数 | mIoU | PA |
|------|--------|------|----|
| AGFNet 单次前向 | 1 | 80.06 | 96.08 |
| + flip TTA | 2 | 80.35 | 96.15 |
| + ms-flip TTA (0.75/1.0/1.25) | 6 | 79.65 | 95.66 |
| + ms-flip TTA (0.9/1.0/1.1) | 6 | 79.29 | 95.54 |
| **三模型集成 + flip** | 6 | **80.83** | **96.27** |

注:训练时(RTX 3090)单次前向为 80.35,与 MPS 重测的 80.06 之间约 0.29pp 为跨设备数值差异,集中在小类;表内各行均在同一硬件、同一流水线下测得,增量可直接比较。多尺度 TTA 掉点的原因分析见论文 §4.6.6 Inference-Time Enhancements(AGFNet 已通过 FPN 与四级 MiT 层级在内部聚合多分辨率信息)。

## 项目结构

```
├── dataset.py              # CamVid 数据集加载与数据增强
├── metrics.py              # mIoU、Pixel Accuracy 计算(混淆矩阵)
├── train.py                # 阶段一:CNN Baseline 训练
├── train_segformer.py      # 阶段三:SegFormer 训练
├── train_fusion.py         # 阶段四 v1:单尺度融合
├── train_fusion_v2.py      # 阶段四 v2:AGFNet(多尺度 + AGFM/Concat/Add)
├── evaluate_tta.py         # 单模型评测(none/flip/ms/ms-flip TTA)
├── evaluate_ensemble.py    # 三模型概率级集成评测
├── results_eval/           # 评测结果 JSON
├── logs/                   # 训练日志
├── checkpoints/            # 模型权重(.gitignore)
├── ResNet34-UNet_baseline/ # 阶段一:U-Net 独立实现(队友版本,详见下文)
├── ResNet34-UNET-CBAM/     # 阶段二:U-Net + CBAM 跨骨干对照实验
├── Segformer-b4/           # 阶段三:SegFormer-B4 实验
├── requirements.txt
└── README.md
```

## 评价指标

- **mIoU**:11 类平均交并比(排除 void 类)
- **Pixel Accuracy**:像素准确率
- 推理速度:RTX 3090、360×480、batch=1

---

# 各阶段子目录文档

## ResNet34-UNet Baseline（阶段一）

### Per‑Class IoU & Accuracy

| Class       | IoU (%) | Acc (%) |
|-------------|---------|---------|
| Sky         | 94.20   | 96.64   |
| Building    | 86.47   | 89.35   |
| Pole        | 18.07   | 24.94   |
| Road        | 97.18   | 98.32   |
| Pavement    | 88.68   | 95.92   |
| Tree        | 90.82   | 97.66   |
| SignSymbol  | 57.50   | 68.90   |
| Fence       | 72.45   | 88.33   |
| Car         | 86.26   | 90.28   |
| Pedestrian  | 55.55   | 77.02   |
| Bicyclist   | 80.05   | 84.65   |
| Void        | 30.25   | 61.49   |

> 最佳 checkpoint (epoch 37): **mIoU = 71.46%**，**PA = 93.33%**

### 模型性能对比

| Method                  | mIoU   | PA     | FPS    | Params   |
|-------------------------|--------|--------|--------|----------|
| Stage 1 UNet (ResNet34) | 71.46% | 93.33% | 168.25 | 24.40M   |

###  网络架构

```
输入 [3, 512, 512]
  │
  ▼ enc1: ResNet.conv1+bn1+relu → [64, 256, 256]  stride=2
  ▼ pool: ResNet.maxpool         → [64, 128, 128]
  ▼ enc2: ResNet.layer1 (3×BasicBlock) → [64, 128, 128]   ← Skip e1
  ▼ enc3: ResNet.layer2 (4×BasicBlock) → [128, 64, 64]    ← Skip e2
  ▼ enc4: ResNet.layer3 (6×BasicBlock) → [256, 32, 32]    ← Skip e3
  ▼ enc5: ResNet.layer4 (3×BasicBlock) → [512, 16, 16]    ← Skip e4
  │
  ▼ dec4: ConvTranspose2d(512→256) + Cat(e4) → DoubleConv → [256, 32, 32]
  ▼ dec3: ConvTranspose2d(256→128) + Cat(e3) → DoubleConv → [128, 64, 64]
  ▼ dec2: ConvTranspose2d(128→64)  + Cat(e2) → DoubleConv → [64, 128, 128]
  ▼ dec1: ConvTranspose2d(64→64)   + Cat(e1) → DoubleConv → [64, 256, 256]
  │
  ▼ Bilinear Upsample 2× → [64, 512, 512]
  ▼ Conv2d(64, 12)        → [12, 512, 512]
```

**架构特点**：
- 编码器：ResNet34 ImageNet 预训练，5 阶段逐级下采样（H→H/32）
- 解码器：4 个 DecoderBlock，每个由 ConvTranspose2d 上采样 + Skip Connection Concat + DoubleConv 组成
- 跳跃连接：通过对称编码器-解码器架构与跳跃连接，恢复高频空间细节
- 数据集：CamVid（11 个语义类别，367 张训练图像）
- 损失函数：CrossEntropyLoss (ignore_index=11) + DiceLoss (1:1)

---

## ResNet34-UNet + CBAM（阶段二）

### Per‑Class IoU & Accuracy

| Class       | IoU (%) | Acc (%) |
|-------------|---------|---------|
| Sky         | 94.31   | 97.03   |
| Building    | 87.39   | 90.80   |
| Pole        | 16.23   | 20.29   |
| Road        | 97.00   | 98.07   |
| Pavement    | 88.18   | 96.10   |
| Tree        | 91.35   | 96.97   |
| SignSymbol  | 59.82   | 67.77   |
| Fence       | 71.86   | 84.46   |
| Car         | 85.80   | 93.84   |
| Pedestrian  | 57.32   | 74.01   |
| Bicyclist   | 79.88   | 86.56   |
| Void        | 30.07   | 61.46   |

> 最佳 checkpoint (epoch 25): **mIoU = 71.60%**，**PA = 93.50%**

### 模型性能对比

| Method            | mIoU   | PA     | FPS   | Params   |
|-------------------|--------|--------|-------|----------|
| Stage 2 UNet‑CBAM | 71.60% | 93.50% | 91.73 | 24.44M   |

### 改进方案

**CBAM 注意力模块**嵌入解码器深层：

```
DecoderBlockCBAM:
  Up → Cat(skip) → DoubleConv → CBAM → output
                                 │
                    ┌────────────┴────────────┐
                    │  ChannelAttention       │
                    │  AvgPool+MaxPool → MLP  │
                    │  → Sigmoid → [C,1,1]    │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │  SpatialAttention       │
                    │  ChannelAvgMax → 7×7Conv│
                    │  → Sigmoid → [1,H,W]    │
                    └─────────────────────────┘
```

**CBAM 仅在深层解码块 (dec3, dec4) 嵌入**：平衡高级语义过滤与低级空间保留，使网络能保留精确边界定位所需的细粒度细节。深层通道数多（128/256），语义抽象级别高，注意力能有效重标定重要性。浅层 dec1/dec2 保留原始卷积以保护纹理细节。
### mIoU 停滞 ≠ 没有改进：注意力预算的结构性转移

76.01% → 76.03%，mIoU 变化仅 **+0.02 pp**，但这不是"没涨"——这是模型内部发生了一次**注意力预算的结构性再分配**。

**小类集体受益**：SignSymbol (+5.12 pp)、Bicyclist (+4.37 pp)、Fence (+3.07 pp)、Pedestrian (+2.35 pp)。这四个类别平均提升 **+3.73 pp**。

**大类付出代价**：Pavement (-4.19 pp)、Car (-3.49 pp)、Road (-1.28 pp)。这三个类别平均下降 **-2.99 pp**。

增益和损失几乎精确对冲，导致 mIoU 原地踏步。这种"零和转移"的机制：

1. **类别权重改变了梯度景观**：逆频率权重使 Pole 的每个像素贡献约 3× 梯度，Road 约 0.3×。模型被强制分配更多优化资源给少数类。
2. **CBAM 放大了权重效应**：通道注意力学习重标定特征通道。在加权损失的压力下，注意力自然偏向稀有类相关的语义通道，抑制了主导类的通道。
3. **物理约束**：CamVid 仅 367 张训练图，Road 占 ~30% 像素。从 Road "偷" 1.3 pp 的模型容量，能"喂"给 4 个小类各 2-5 pp——因为小类的像素总数极低，少量的正确预测就能大幅提升 IoU。

**关键结论**：**CBAM 没有改变 mIoU，但重新分配了模型的注意力预算** 

