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
python stage1_cnn_baseline/train.py --model unet --epochs 100 --batch-size 8 --device auto

# DeepLabV3+ (ResNet50)
python stage1_cnn_baseline/train.py --model deeplabv3plus --epochs 80 --batch-size 8 --device auto

# CPU 快速验证(2 epoch)
python stage1_cnn_baseline/train.py --model unet --epochs 2 --batch-size 2 --device cpu
```

`--device auto` 自动选择 CUDA → MPS → CPU。最优模型自动保存至 `checkpoints/<model>_best.pth`。

阶段一共三个 baseline:**DeepLabV3+** 与 **U-Net** 由上面的主线 `train.py` 通过 `--model` 切换;**ResNet34-UNet** 和 **SegNet** 各为独立实现,放在对应子文件夹,接口与运行方式一致(自带 `dataset.py`/`metrics.py`,512×512,12 类、Void=11 忽略):

```bash
# ResNet34-UNet baseline(队友独立实现)
python stage1_cnn_baseline/ResNet34-UNet/train.py

# SegNet baseline(VGG16-BN 编码器 + 最大池化索引反池化解码器)
python stage1_cnn_baseline/SegNet/train.py
```

> 这两个独立实现脚本默认从各自目录内的相对路径加载数据,数据根在脚本顶部的 `SegDataset(...)` 处配置。

### 阶段二:DeepLabV3+ + 注意力/损失/TTA

阶段二 DeepLabV3+ 完整实验链已独立整理到 `stage2_attention_loss_tta/DeepLabV3Plus-CBAM/`。默认使用内置 294/73 validation 协议；已有结果不得视为 101 张 official test 结果。

```bash
python stage2_attention_loss_tta/DeepLabV3Plus-CBAM/train.py \
    --config stage2_attention_loss_tta/DeepLabV3Plus-CBAM/configs/cbam_spatial.json \
    --data-root .

python stage2_attention_loss_tta/DeepLabV3Plus-CBAM/evaluate_tta.py \
    --checkpoint stage2_attention_loss_tta/DeepLabV3Plus-CBAM/checkpoints/cbam_spatial/best.pth \
    --data-root . --split-protocol internal-val --flip
```

统一参数包括 `--attention cbam|eca|coord|none`、`--cbam full|channel|spatial|none`、`--loss ce|dice|focal|ce+dice|ce+focal` 和 `--split-protocol internal-val|official-test`。完整说明见 `stage2_attention_loss_tta/DeepLabV3Plus-CBAM/README.md`。U-Net + CBAM 的跨骨干对照实验见 `stage2_attention_loss_tta/ResNet34-UNet-CBAM/`。

### 阶段三:SegFormer

```bash
python stage3_segformer/train_segformer.py --variant b2 --epochs 100
```

SegFormer-B4 的独立实验见 `stage3_segformer/Segformer-b4/`。

### 阶段四:CNN-Transformer 融合(AGFNet)

```bash
# v1:单尺度 Transformer 特征
python stage4_fusion_agfnet/train_fusion.py

# v2:多尺度 MiT-B2 + FPN,三种融合方式
python stage4_fusion_agfnet/train_fusion_v2.py --fusion agfm         # 自适应门控融合(最终模型)
python stage4_fusion_agfnet/train_fusion_v2.py --fusion none_concat  # 拼接融合(消融)
python stage4_fusion_agfnet/train_fusion_v2.py --fusion none_add     # 相加融合(消融)
```

## 评测(TTA 与集成)

`evaluate_tta.py` 对单个 Stage IV checkpoint 做带 TTA 的独立评测,`evaluate_ensemble.py` 对三个融合变体做概率级集成:

```bash
# 单模型,无 TTA(复现训练时评测)
python stage4_fusion_agfnet/evaluate_tta.py --checkpoint checkpoints/fusion_v2_agfm_best.pth \
    --fusion agfm --tta none --save-json stage4_fusion_agfnet/results_eval/agfm_notta.json

# 单模型 + 水平翻转 TTA(2 视图)
python stage4_fusion_agfnet/evaluate_tta.py --checkpoint checkpoints/fusion_v2_agfm_best.pth \
    --fusion agfm --tta flip --save-json stage4_fusion_agfnet/results_eval/agfm_flip.json

# 单模型 + 多尺度翻转 TTA(6 视图,实测掉点,见论文分析)
python stage4_fusion_agfnet/evaluate_tta.py --checkpoint checkpoints/fusion_v2_agfm_best.pth \
    --fusion agfm --tta ms-flip --scales 0.75 1.0 1.25

# 三模型集成(AGFM + Concat + Add)+ 翻转 TTA
python stage4_fusion_agfnet/evaluate_ensemble.py --save-json stage4_fusion_agfnet/results_eval/ensemble_flip.json
```

**最终结果:三模型集成 + flip TTA 达到 80.83% mIoU / 96.27% PA。** 各中间配置的完整指标见 `stage4_fusion_agfnet/results_eval/` 下的 JSON;多尺度 TTA 掉点的分析见论文 §4.6.6。

## 项目结构

仓库按四个研究阶段组织,主线脚本共享 `common/` 下的数据集与指标模块,各阶段另含队友的独立实现子文件夹:

```
├── common/                          # 主线共享模块
│   ├── dataset.py                   #   CamVid 数据集加载与数据增强
│   └── metrics.py                   #   mIoU、Pixel Accuracy 计算(混淆矩阵)
│
├── stage1_cnn_baseline/             # 阶段一:CNN Baseline(三个 baseline)
│   ├── train.py                     #   主线:U-Net / DeepLabV3+(--model 切换)
│   ├── ResNet34-UNet/               #   ResNet34-UNet 独立实现(队友版本)
│   └── SegNet/                      #   SegNet 独立实现(VGG16-BN + max-unpool)
│
├── stage2_attention_loss_tta/       # 阶段二:注意力 / 损失 / TTA
│   ├── DeepLabV3Plus-CBAM/          #   DeepLabV3+ 注意力·损失·TTA 实验链
│   └── ResNet34-UNet-CBAM/          #   U-Net + CBAM 跨骨干对照实验
│
├── stage3_segformer/                # 阶段三:SegFormer 跨框架对比
│   ├── train_segformer.py           #   SegFormer-B2 主线训练
│   ├── Segformer-b4/                #   SegFormer-B4 独立实验
│   └── logs/                        #   训练日志
│
├── stage4_fusion_agfnet/            # 阶段四:CNN-Transformer 融合(AGFNet)
│   ├── train_fusion.py              #   v1:单尺度融合
│   ├── train_fusion_v2.py           #   v2:AGFNet(多尺度 + AGFM/Concat/Add)
│   ├── evaluate_tta.py              #   单模型评测(none/flip/ms/ms-flip TTA)
│   ├── evaluate_ensemble.py         #   三模型概率级集成评测
│   ├── results_eval/                #   评测结果 JSON
│   └── logs/                        #   训练日志
│
├── requirements.txt
└── README.md
```

> 主线训练/评测脚本通过文件内的 `sys.path` 引导自动定位仓库根,导入 `common/`,从任意目录均可运行(如 `python stage1_cnn_baseline/train.py ...`)。模型权重默认保存到运行目录下的 `checkpoints/`(已 `.gitignore`)。

## 评价指标

- **mIoU**:11 类平均交并比(排除 void 类)
- **Pixel Accuracy**:像素准确率
- 推理速度:RTX 3090、360×480、batch=1

---

