# DeepLabV3+ Stage II: Attention, Losses, and TTA

This directory is the standalone Stage II experiment chain. It does not depend on the Stage I, III, or IV implementations in the repository root.

## Supported experiments

- DeepLabV3+ with a dilated ResNet50 backbone (output stride 8)
- Attention: `cbam`, `eca`, `coord`, or `none`
- CBAM ablations: `full`, `channel`, `spatial`, or `none`
- Losses: `ce`, `dice`, `focal`, `ce+dice`, or `ce+focal`
- Checkpoint resume, standard evaluation, multi-scale/flip TTA, and CBAM spatial-map visualization

`ce` uses median-frequency class weights, matching the original Stage II experiment implementation. Void pixels use label ID `11` and are excluded consistently from every loss and metric.

## Data layouts

Both layouts are detected automatically.

```text
# Repository layout
<root>/
  train/images/*.png
  train/labels/*.png
  test/images/*.png
  test/labels/*.png

# Original Stage II layout
<root>/
  images/train/*.png
  images/val/*.png
  images/test/*.png
  annotations/train/*.png
  annotations/val/*.png
  annotations/test/*.png
```

The split protocol is explicit:

- `internal-val` (default): the bundled manifests reproduce the 294 training / 73 validation split from the original 367 training images.
- `official-test`: trains on all 367 training images and evaluates on the separate 101-image test set.

The existing results under `results/` are **294/73 validation results**, not official 101-image test results.

## Train

From the repository root:

```bash
python stage2_attention_loss_tta/DeepLabV3Plus-CBAM/train.py \
  --config stage2_attention_loss_tta/DeepLabV3Plus-CBAM/configs/cbam_spatial.json \
  --data-root .
```

Explicit CLI options override JSON values:

```bash
python stage2_attention_loss_tta/DeepLabV3Plus-CBAM/train.py \
  --config stage2_attention_loss_tta/DeepLabV3Plus-CBAM/configs/default.json \
  --data-root . \
  --attention eca \
  --cbam none \
  --loss ce+dice \
  --split-protocol official-test
```

Portable preset launcher:

```bash
stage2_attention_loss_tta/DeepLabV3Plus-CBAM/scripts/train_preset.sh cbam_full --data-root .
```

Checkpoints contain `model`, optimizer, scheduler, serialized training arguments, current validation metrics, and the best mIoU. `last.pth` is written every epoch; `best.pth` is written only on improvement.

## Evaluate and visualize

```bash
python stage2_attention_loss_tta/DeepLabV3Plus-CBAM/evaluate_tta.py \
  --checkpoint stage2_attention_loss_tta/DeepLabV3Plus-CBAM/checkpoints/cbam_spatial/best.pth \
  --data-root . \
  --split-protocol internal-val \
  --scales 0.75 1.0 1.25 --flip

python stage2_attention_loss_tta/DeepLabV3Plus-CBAM/visualize_attention.py \
  --checkpoint stage2_attention_loss_tta/DeepLabV3Plus-CBAM/checkpoints/cbam_spatial/best.pth \
  --data-root . --num-samples 6
```

The visualization script requires a CBAM checkpoint with a spatial branch.

## Reproduced CBAM ablation

Protocol: bundled 294/73 split, 11 classes, void ID 11 ignored, no TTA.

| Variant | mIoU | PA | FWIoU |
|---|---:|---:|---:|
| Channel only | 74.57% | 94.03% | 89.66% |
| Spatial only | **74.97%** | 94.09% | 89.76% |
| Full CBAM | 74.91% | **94.11%** | **89.77%** |

Machine-readable aggregate, per-class, latency, and training configuration files are in `results/`. Model weights and raw cluster logs are intentionally excluded.

## Dependencies

The repository `requirements.txt` covers this module: PyTorch, torchvision, albumentations, NumPy, Pillow, matplotlib, and OpenCV. ImageNet pretraining may download torchvision ResNet50 weights; use `--no-pretrain` for offline runs and tests.
