import torch
import os


def analyze_model_file(model_path):
    # 1. 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 文件不存在: {model_path}")
        return

    # 2. 获取文件大小
    file_size_bytes = os.path.getsize(model_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"📂 分析文件: {model_path}")
    print(f"📏 文件总大小: {file_size_mb:.2f} MB")
    print("-" * 30)

    # 3. 加载模型数据
    # map_location='cpu' 确保在CPU上也能加载，防止GPU环境报错
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    # 4. 判断加载的数据类型
    print(f"📦 数据类型: {type(checkpoint)}")

    # --- 情况 A: 如果是 state_dict (只有权重) ---
    if isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
        print("✅ 这是一个 state_dict (仅包含权重参数)")
        print(f"📝 包含参数层数: {len(checkpoint)}")

        total_params = 0
        print("\n详细参数列表:")
        for name, param in checkpoint.items():
            # 计算该层参数量
            num_params = param.numel()
            # 计算占用字节 (Float32 = 4字节)
            byte_size = num_params * 4
            total_params += num_params

            print(f"  - {name}: 形状 {list(param.shape)} -> {num_params:,} 个参数 ({byte_size / 1024:.1f} KB)")

        print("-" * 30)
        print(f"📊 纯权重理论大小: {total_params * 4 / (1024 * 1024):.2f} MB")

    # --- 情况 B: 如果是完整模型 (包含结构+环境) ---
    else:
        print("⚠️ 这是一个完整模型对象 (包含结构、权重和环境信息)")
        print("由于包含了 Python 类定义和环境元数据，无法直接列出具体字节分布。")
        print("这也是导致文件体积巨大 (999MB) 的原因。")

        # 如果是完整模型，我们可以尝试提取它的 state_dict 来看看权重有多大
        if hasattr(checkpoint, 'state_dict'):
            state_dict = checkpoint.state_dict()
            total_params = sum(p.numel() for p in state_dict.values())
            print(f"\n💡 即使在这个大文件中，真正的权重数据仅约为: {total_params * 4 / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    # 在这里修改你的模型路径
    model_path = 'yolov1_best.pth'
    analyze_model_file(model_path)