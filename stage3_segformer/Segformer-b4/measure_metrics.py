"""
Stage 3: SegFormer-B4 Params / FLOPs / FPS
"""
import time
import torch
from model import SegFormerB4


def measure_params(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6


def measure_flops(model, device, input_size=(1, 3, 512, 512)):
    try:
        from thop import profile
        print("      (thop profiling, may take ~30s)...", end=" ", flush=True)
        dummy = torch.randn(input_size).to(device)
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        print("done.", flush=True)
        return flops / 1e9
    except ImportError:
        print("(thop not installed, skip)", flush=True)
        return None


def measure_fps(model, device, input_size=(1, 3, 512, 512), warmup=30, iters=100):
    dummy = torch.randn(input_size).to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - start
    return iters / elapsed


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("[1/2] Building SegFormer-B4...")
    model = SegFormerB4(num_classes=12).to(device)
    model.eval()
    print("      Done.\n")

    print("[2/2] Measuring...")
    params = measure_params(model)
    print(f"  Params : {params:.2f} M")

    flops = measure_flops(model, device)
    if flops is not None:
        print(f"  FLOPs  : {flops:.2f} G")
    else:
        print(f"  FLOPs  : N/A")

    fps = measure_fps(model, device)
    print(f"  FPS    : {fps:.2f}")

    print(f"\n{'='*50}")
    print(f"  Model           : SegFormer-B4")
    print(f"  Params (M)      : {params:.2f}")
    print(f"  FLOPs  (G)      : {flops:.2f}" if flops else "  FLOPs  (G)      : N/A")
    print(f"  FPS             : {fps:.2f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
