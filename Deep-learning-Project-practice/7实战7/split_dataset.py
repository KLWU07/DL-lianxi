#!/usr/bin/env python3
# 带数据一致性检查功能：获取所有图片的ID。为每个ID查找对应的标注文件，即检查 Annotations_txt/2007_000032.txt 是否存在。报告缺失情况，告诉你哪些图片没有标注，帮你提前发现并修复数据问题。
"""
数据集划分脚本：按8:1:1比例随机划分训练集、验证集、测试集
保存为ImageSets/Main/train.txt, val.txt, test.txt
"""

import os
import random
import argparse
from pathlib import Path
from typing import List, Tuple


def split_dataset(
        data_root: str,
        jpeg_dir: str = "JPEGImages",
        output_dir: str = "ImageSets/Main",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        shuffle: bool = True
) -> None:
    """
    随机划分数据集并保存划分结果

    参数:
        data_root: VOC数据集根目录
        jpeg_dir: 图片文件夹名称
        output_dir: 输出文件夹名称
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子（确保可复现）
        shuffle: 是否打乱数据
    """
    # 验证比例总和为1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"比例总和应为1.0，当前为{total_ratio}")

    # 设置随机种子
    random.seed(seed)

    # 构建路径
    jpeg_path = Path(data_root) / jpeg_dir
    output_path = Path(data_root) / output_dir

    print("=" * 60)
    print("数据集划分工具")
    print("=" * 60)

    # 1. 获取所有图片文件
    if not jpeg_path.exists():
        raise FileNotFoundError(f"图片文件夹不存在: {jpeg_path}")

    # 支持多种图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(jpeg_path.glob(f'*{ext}'))
        image_files.extend(jpeg_path.glob(f'*{ext.upper()}'))

    if not image_files:
        raise FileNotFoundError(f"在 {jpeg_path} 中未找到任何图片文件")

    # 提取纯文件名（不带扩展名）
    image_ids = []
    for img_file in image_files:
        # 保留原始文件名，去除扩展名
        image_ids.append(img_file.stem)

    print(f"找到 {len(image_ids)} 张图片")

    # 2. 去重并排序（确保可复现）
    image_ids = list(set(image_ids))  # 去重
    image_ids.sort()  # 排序，确保每次相同的顺序

    # 3. 打乱顺序
    if shuffle:
        print(f"使用随机种子 {seed} 打乱数据顺序...")
        random.shuffle(image_ids)

    # 4. 计算划分点
    total_count = len(image_ids)
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)

    # 处理可能的舍入误差，确保测试集包含所有剩余图片
    test_count = total_count - train_count - val_count

    # 5. 执行划分
    train_ids = image_ids[:train_count]
    val_ids = image_ids[train_count:train_count + val_count]
    test_ids = image_ids[train_count + val_count:]

    print("\n划分结果统计:")
    print("-" * 40)
    print(f"训练集: {len(train_ids)} 张 ({len(train_ids) / total_count * 100:.1f}%)")
    print(f"验证集: {len(val_ids)} 张 ({len(val_ids) / total_count * 100:.1f}%)")
    print(f"测试集: {len(test_ids)} 张 ({len(test_ids) / total_count * 100:.1f}%)")
    print(f"总计: {total_count} 张")

    # 6. 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 7. 保存划分文件
    train_file = output_path / "train.txt"
    val_file = output_path / "val.txt"
    test_file = output_path / "test.txt"

    with open(train_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_ids))

    with open(val_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_ids))

    with open(test_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_ids))

    print("\n划分文件已保存:")
    print(f"  {train_file}")
    print(f"  {val_file}")
    print(f"  {test_file}")

    # 8. 保存划分详情（可选）
    detail_file = output_path / "split_details.txt"
    with open(detail_file, 'w', encoding='utf-8') as f:
        f.write("数据集划分详情\n")
        f.write("=" * 40 + "\n")
        f.write(f"数据根目录: {data_root}\n")
        f.write(f"随机种子: {seed}\n")
        f.write(f"总图片数: {total_count}\n")
        f.write(f"训练集: {len(train_ids)} ({train_ratio * 100:.1f}%)\n")
        f.write(f"验证集: {len(val_ids)} ({val_ratio * 100:.1f}%)\n")
        f.write(f"测试集: {len(test_ids)} ({test_ratio * 100:.1f}%)\n")
        f.write("\n划分比例: 训练集:验证集:测试集 = ")
        f.write(f"{train_ratio}:{val_ratio}:{test_ratio}\n")

    print(f"划分详情: {detail_file}")
    print("=" * 60)


def check_annotation_compatibility(
        data_root: str,
        jpeg_dir: str = "JPEGImages",
        annot_dir: str = "Annotations_txt"
) -> None:
    """
    检查图片文件和标注文件的对应关系

    参数:
        data_root: 数据集根目录
        jpeg_dir: 图片文件夹
        annot_dir: 标注文件夹
    """
    print("\n检查标注文件兼容性...")

    jpeg_path = Path(data_root) / jpeg_dir
    annot_path = Path(data_root) / annot_dir

    # 获取图片文件
    image_files = list(jpeg_path.glob('*.jpg'))
    image_ids = [f.stem for f in image_files]

    missing_annotations = []

    for img_id in image_ids:
        annot_file = annot_path / f"{img_id}.txt"
        if not annot_file.exists():
            missing_annotations.append(img_id)

    if missing_annotations:
        print(f"警告: {len(missing_annotations)} 张图片缺少对应的标注文件")
        if len(missing_annotations) <= 10:
            print("缺失标注的图片ID:")
            for img_id in missing_annotations:
                print(f"  - {img_id}")
        else:
            print("前10个缺失标注的图片ID:")
            for img_id in missing_annotations[:10]:
                print(f"  - {img_id}")
            print(f"  ... 共 {len(missing_annotations)} 个")
    else:
        print("✓ 所有图片都有对应的标注文件")


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description="数据集划分工具 (8:1:1比例)")
    parser.add_argument(
        "--data_root",
        type=str,
        default="./VOC2012",
        help="VOC数据集根目录 (默认: ./VOC2012)"
    )
    parser.add_argument(
        "--jpeg_dir",
        type=str,
        default="JPEGImages",
        help="图片文件夹名称 (默认: JPEGImages)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ImageSets/Main",
        help="输出文件夹名称 (默认: ImageSets/Main)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="训练集比例 (默认: 0.8)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="验证集比例 (默认: 0.1)"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="测试集比例 (默认: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)"
    )
    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="不打乱数据顺序"
    )
    parser.add_argument(
        "--check_annotations",
        action="store_true",
        help="检查标注文件兼容性"
    )

    args = parser.parse_args()

    try:
        # 执行划分
        split_dataset(
            data_root=args.data_root,
            jpeg_dir=args.jpeg_dir,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            shuffle=not args.no_shuffle
        )

        # 如果需要，检查标注文件
        if args.check_annotations:
            check_annotation_compatibility(
                data_root=args.data_root,
                jpeg_dir=args.jpeg_dir,
                annot_dir="Annotations_txt"  # 你的标注文件夹
            )

    except Exception as e:
        print(f"错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

# python split_dataset.py \
#     --data_root ./VOC2012 \
#     --train_ratio 0.8 \
#     --val_ratio 0.1 \
#     --test_ratio 0.1 \
#     --seed 12345 \
#     --check_annotations+