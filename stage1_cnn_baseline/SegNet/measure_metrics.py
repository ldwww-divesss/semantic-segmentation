import torch
import time

from model import SegNet


def measure_model(model_name, model, device="cuda", img_size=(1, 3, 512, 512)):
    print(f"[{model_name}] 测试中...")
    model = model.to(device)
    model.eval()

    # ===============================
    # 1. 计算 Params (参数量)
    # ===============================
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_m = total_params / 1e6
    print(f"  -> Params : {params_m:.2f} M")

    # ===============================
    # 2. 计算 FPS (推理速度)
    # ===============================
    dummy_input = torch.randn(img_size).to(device)

    # 预热阶段 (Warm-up)：排除 GPU 上下文初始化开销
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)

    iterations = 200
    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)

    if device == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    fps = iterations / total_time
    print(f"  -> FPS    : {fps:.2f} frames/sec")
    print("-" * 40)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"当前测试设备: {device}\n" + "=" * 40)

    # 阶段一 SegNet Baseline（测速不需要预训练权重）
    measure_model("SegNet (Baseline)", SegNet(num_classes=12, pretrained=False), device)
