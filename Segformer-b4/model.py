"""
SegFormer-B4: 双后端支持
  1. HuggingFace transformers (优先, 有预训练权重)
  2. 纯 PyTorch 实现 (fallback, 零依赖, 随机初始化)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_hf_segformer(num_classes):
    """从本地加载预训练权重"""
    from transformers import SegformerForSemanticSegmentation
    import os

    # 相对于 model.py 所在目录的本地权重路径
    local_dir = os.path.join(os.path.dirname(__file__), "segformer_b4_local")

    model = SegformerForSemanticSegmentation.from_pretrained(
        local_dir,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    return model


def _build_pure_segformer(num_classes):
    """纯 PyTorch SegFormer, 无预训练"""
    return _PureSegFormerB4(num_classes)


# ---- 纯 PyTorch 实现 (fallback) ----
ENC_DIMS = [64, 128, 320, 512]
DEPTHS  = [3, 8, 27, 3]
HEADS   = [1, 2, 5, 8]
SR_RATIOS = [8, 4, 2, 1]
DEC_DIM = 768


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=64, patch_size=7, stride=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, stride, patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, E, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return x


class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr_conv = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)
        else:
            self.sr_conv = None

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_conv is not None:
            x_2d = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_2d = self.sr_conv(x_2d)          # [B, C, H/sr, W/sr]
            _, _, H2, W2 = x_2d.shape
            x_2d = x_2d.flatten(2).transpose(1, 2)  # [B, N', C]
            x = self.sr_norm(x_2d)              # LayerNorm over C
        else:
            x = x  # sr_ratio=1, no reduction

        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class MixFFN(nn.Module):
    def __init__(self, dim, ffn_ratio=4):
        super().__init__()
        hidden = dim * ffn_ratio
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.fc2(self.dwconv(self.act(self.fc1(x))))
        return x.flatten(2).transpose(1, 2)


class MiTBlock(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MiTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed1 = OverlapPatchEmbed(3, ENC_DIMS[0], 7, 4)
        self.block1 = nn.Sequential(*[MiTBlock(ENC_DIMS[0], HEADS[0], SR_RATIOS[0]) for _ in range(DEPTHS[0])])
        self.patch_embed2 = OverlapPatchEmbed(ENC_DIMS[0], ENC_DIMS[1], 3, 2)
        self.block2 = nn.Sequential(*[MiTBlock(ENC_DIMS[1], HEADS[1], SR_RATIOS[1]) for _ in range(DEPTHS[1])])
        self.patch_embed3 = OverlapPatchEmbed(ENC_DIMS[1], ENC_DIMS[2], 3, 2)
        self.block3 = nn.Sequential(*[MiTBlock(ENC_DIMS[2], HEADS[2], SR_RATIOS[2]) for _ in range(DEPTHS[2])])
        self.patch_embed4 = OverlapPatchEmbed(ENC_DIMS[2], ENC_DIMS[3], 3, 2)
        self.block4 = nn.Sequential(*[MiTBlock(ENC_DIMS[3], HEADS[3], SR_RATIOS[3]) for _ in range(DEPTHS[3])])

    def forward(self, x):
        feats = []
        for pe, blk in [(self.patch_embed1, self.block1), (self.patch_embed2, self.block2),
                        (self.patch_embed3, self.block3), (self.patch_embed4, self.block4)]:
            x = pe(x)
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = blk(x)
            x = x.transpose(1, 2).reshape(B, C, H, W)
            feats.append(x)
        return feats


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Conv2d(in_dim, out_dim, 1)
    def forward(self, x):
        return self.fc(x)


class SegFormerDecoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.mlp_stages = nn.ModuleList([MLP(d, DEC_DIM) for d in ENC_DIMS])
        self.fuse = nn.Sequential(
            MLP(DEC_DIM * 4, DEC_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(DEC_DIM, num_classes, 1),
        )

    def forward(self, feats):
        h, w = feats[0].shape[2], feats[0].shape[3]
        out = []
        for f, mlp in zip(feats, self.mlp_stages):
            out.append(F.interpolate(mlp(f), size=(h, w), mode="bilinear", align_corners=False))
        return self.fuse(torch.cat(out, dim=1))


class _PureSegFormerB4(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.encoder = MiTEncoder()
        self.decoder = SegFormerDecoder(num_classes)

    def forward(self, x):
        feats = self.encoder(x)
        logits = self.decoder(feats)
        return F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)


# ---- 统一接口 ----
class SegFormerB4(nn.Module):
    """SegFormer-B4 for CamVid 12-class segmentation.

    后端优先级:
      1. HuggingFace `nvidia/segformer-b4-finetuned-ade-512-512` (推荐)
      2. 纯 PyTorch 实现 (无预训练, 367 张图上 64M 模型容易过拟合)
    """

    def __init__(self, num_classes=12, backend="hf"):
        super().__init__()
        self.backend = backend

        if backend == "hf":
            print("[Model] Trying HuggingFace SegFormer...", flush=True)
            try:
                self.model = _build_hf_segformer(num_classes)
                self._forward = self._forward_hf
                print("[Model] HuggingFace SegFormer loaded.", flush=True)
            except Exception as e:
                print(f"[Model] HF failed ({e}), falling back to pure PyTorch.", flush=True)
                self.model = _build_pure_segformer(num_classes)
                self._forward = self._forward_pure
        else:
            print("[Model] Using pure PyTorch SegFormer (no pretrained weights).", flush=True)
            self.model = _build_pure_segformer(num_classes)
            self._forward = self._forward_pure

    def _forward_hf(self, x):
        out = self.model(pixel_values=x)
        logits = out.logits
        return F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)

    def _forward_pure(self, x):
        return self.model(x)

    def forward(self, x):
        return self._forward(x)
