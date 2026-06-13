# CamVid 城市街景语义分割（四阶段研究 + AGFNet）

四阶段实验研究:CNN Baseline → CNN 改进(CBAM/损失/TTA)→ SegFormer 跨框架对比 → CNN-Transformer 融合(AGFNet)。
最终结果:三模型集成 + flip TTA **80.83% mIoU / 96.27% PA**。

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

### 阶段二:DeepLabV3+ + 注意力/损失/TTA

阶段二 DeepLabV3+ 完整实验链已独立整理到 `DeepLabV3Plus-CBAM/`。默认使用内置 294/73 validation 协议；已有结果不得视为 101 张 official test 结果。

```bash
python DeepLabV3Plus-CBAM/train.py \
    --config DeepLabV3Plus-CBAM/configs/cbam_spatial.json \
    --data-root .

python DeepLabV3Plus-CBAM/evaluate_tta.py \
    --checkpoint DeepLabV3Plus-CBAM/checkpoints/cbam_spatial/best.pth \
    --data-root . --split-protocol internal-val --flip
```

统一参数包括 `--attention cbam|eca|coord|none`、`--cbam full|channel|spatial|none`、`--loss ce|dice|focal|ce+dice|ce+focal` 和 `--split-protocol internal-val|official-test`。完整说明见 `DeepLabV3Plus-CBAM/README.md`。

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

**最终结果:三模型集成 + flip TTA 达到 80.83% mIoU / 96.27% PA。** 各中间配置的完整指标见 `results_eval/` 下的 JSON;多尺度 TTA 掉点的分析见论文 §4.6.6。

## 项目结构

```
├── dataset.py              # CamVid 数据集加载与数据增强
├── metrics.py              # mIoU、Pixel Accuracy 计算(混淆矩阵)
├── train.py                # 阶段一:CNN Baseline 训练
├── DeepLabV3Plus-CBAM/     # 阶段二:DeepLabV3+ 注意力、损失、TTA 独立实验链
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

