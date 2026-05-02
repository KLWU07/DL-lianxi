import os
import shutil
import random
from tqdm import tqdm


def split_dataset():
    # ================= 配置区域 =================
    # 你的原始数据路径 (请修改这里)
    RAW_IMG_DIR = 'data/images'  # 存放所有原始图片的文件夹
    RAW_LABEL_DIR = 'data/labels'  # 存放所有原始标签的文件夹

    # 划分比例
    TRAIN_RATIO = 0.8  # 训练集占 80%

    # 输出目录
    OUTPUT_DIR = 'dataset'
    # =============================================

    # 1. 获取所有图片文件
    img_files = [f for f in os.listdir(RAW_IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    # 2. 随机打乱顺序
    random.seed(42)  # 固定随机种子，保证结果可复现
    random.shuffle(img_files)

    # 3. 计算划分数量
    total_count = len(img_files)
    train_count = int(total_count * TRAIN_RATIO)

    train_files = img_files[:train_count]
    val_files = img_files[train_count:]

    print(f"总共 {total_count} 张图片")
    print(f"训练集: {len(train_files)} 张")
    print(f"验证集: {len(val_files)} 张")

    # 4. 定义目标目录结构
    # YOLOv1 标准结构: dataset/images/train, dataset/labels/train, etc.
    dirs = [
        f"{OUTPUT_DIR}/images/train", f"{OUTPUT_DIR}/images/val",
        f"{OUTPUT_DIR}/labels/train", f"{OUTPUT_DIR}/labels/val"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # 5. 复制文件函数
    def copy_files(file_list, split_name):
        for img_name in tqdm(file_list, desc=f"Processing {split_name}"):
            # 复制图片
            src_img = os.path.join(RAW_IMG_DIR, img_name)
            dst_img = os.path.join(OUTPUT_DIR, "images", split_name, img_name)
            shutil.copy(src_img, dst_img)

            # 复制对应的标签 (假设文件名相同，后缀为 .txt)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            src_label = os.path.join(RAW_LABEL_DIR, label_name)

            # 只有当标签文件存在时才复制（防止报错）
            if os.path.exists(src_label):
                dst_label = os.path.join(OUTPUT_DIR, "labels", split_name, label_name)
                shutil.copy(src_label, dst_label)
            else:
                print(f"警告: 找不到标签 {label_name}")

    # 6. 执行复制
    copy_files(train_files, "train")
    copy_files(val_files, "val")

    print("✅ 数据集划分完成！")


if __name__ == "__main__":
    split_dataset()