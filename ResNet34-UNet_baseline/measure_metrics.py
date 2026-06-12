import torch
import time

# 导入你的四个模型
from model import ResNetUNet
# from model_cbam import ResNetUNetCBAM
# from model_vit import ViTSeg
# from model_transunet import TransUNet

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
    # 构造一张随机生成的假图片用于测试
    dummy_input = torch.randn(img_size).to(device)

    # 预热阶段 (Warm-up):
    # GPU 刚启动时会有额外的上下文初始化开销，必须先空跑几轮排除干扰
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)

    # 正式测速阶段
    iterations = 200
    # 同步 GPU，确保预热任务已彻底清空
    torch.cuda.synchronize() 
    start_time = time.time()

    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)

    # 同步 GPU，确保推理任务全部执行完毕再计算时间
    torch.cuda.synchronize() 
    end_time = time.time()

    total_time = end_time - start_time
    fps = iterations / total_time
    print(f"  -> FPS    : {fps:.2f} frames/sec")
    print("-" * 40)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"当前测试设备: {device}\n" + "="*40)
    
    # 1. 测试阶段一 Baseline
    measure_model("ResNet34-UNet (Baseline)", ResNetUNet(num_classes=12), device)
    
    # 2. 测试阶段二 CBAM
    # measure_model("ResNet34-UNet-CBAM", ResNetUNetCBAM(num_classes=12), device)
    
#     # 3. 测试阶段三 ViT
#     measure_model("ViTSeg (纯 Transformer)", ViTSeg(num_classes=12), device)
    
#     # 4. 测试阶段四 TransUNet
#     measure_model("TransUNet (融合模型)", TransUNet(num_classes=12), device)