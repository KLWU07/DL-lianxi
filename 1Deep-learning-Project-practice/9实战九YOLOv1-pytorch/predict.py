import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from yolo import YOLOv1  # 导入你的模型类

# ================= 配置区域 =================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'yolov1_best.pth'  # 你的模型权重路径
INPUT_DIR = 'input'  # 输入图片文件夹
OUTPUT_DIR = 'output'  # 输出图片文件夹
IMG_SIZE = 448  # 必须与训练时一致！
CONF_THRESHOLD = 0.2  # 【修改点1】阈值降低到 0.1，先看看能不能检出
IOU_THRESHOLD = 0.4  # NMS 阈值


# =============================================

def preprocess_image(image_path, img_size):
    """读取图片并预处理"""
    image = cv2.imread(image_path)
    if image is None:
        return None, None

    # 保存原始尺寸用于还原坐标
    original_h, original_w = image.shape[:2]

    # 缩放图片 (关键步骤：必须强制 Resize 到 448x448)
    image = cv2.resize(image, (img_size, img_size))
    # BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 归一化
    image = image.astype('float32') / 255.0
    # HWC -> CHW
    image = image.transpose((2, 0, 1))
    # 增加 Batch 维度
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

    return image, (original_h, original_w)


def non_max_suppression(prediction, conf_thres=0.5, iou_thres=0.4):
    """
    自适应 NMS 实现
    """
    boxes = []

    # 获取最后一个维度的实际大小 (例如 11 或 12)
    box_dim = prediction.shape[-1]

    # 强制假设 B=2
    B = 2
    stride = box_dim // B

    max_conf_in_batch = 0  # 用于调试：记录这张图的最大置信度

    # 遍历 7x7 网格
    for i in range(7):
        for j in range(7):
            # 遍历 2 个 anchor box
            for b in range(B):
                # 动态计算偏移量
                offset = b * stride

                # 确保不越界
                if offset + 4 >= box_dim:
                    continue

                # 提取数据
                x = prediction[0, i, j, offset + 0].item()
                y = prediction[0, i, j, offset + 1].item()
                w = prediction[0, i, j, offset + 2].item()
                h = prediction[0, i, j, offset + 3].item()
                conf = prediction[0, i, j, offset + 4].item()

                # 类别概率
                cls_conf = 1.0
                if offset + 5 < box_dim:
                    cls_conf = prediction[0, i, j, offset + 5].item()

                final_conf = conf * cls_conf

                # 记录最大置信度用于调试
                if final_conf > max_conf_in_batch:
                    max_conf_in_batch = final_conf

                if final_conf > conf_thres:
                    # 还原绝对坐标 (0-1)
                    grid_x = j
                    grid_y = i

                    # 绝对中心 = (grid + offset) / 7
                    center_x = (grid_x + x) / 7.0
                    center_y = (grid_y + y) / 7.0

                    # 宽高
                    box_w = w
                    box_h = h

                    # 转换为左上角和右下角坐标 (0-1)
                    x1 = center_x - box_w / 2.0
                    y1 = center_y - box_h / 2.0
                    x2 = center_x + box_w / 2.0
                    y2 = center_y + box_h / 2.0

                    # 裁剪到 0-1 范围
                    x1 = max(0, min(1, x1))
                    y1 = max(0, min(1, y1))
                    x2 = max(0, min(1, x2))
                    y2 = max(0, min(1, y2))

                    boxes.append([x1, y1, x2, y2, final_conf])

    # 简单的 NMS
    if not boxes:
        return [], max_conf_in_batch  # 返回最大置信度

    boxes.sort(key=lambda x: x[4], reverse=True)

    keep_boxes = []
    while boxes:
        best_box = boxes.pop(0)
        keep_boxes.append(best_box)

        boxes = [b for b in boxes if calculate_iou(best_box, b) < iou_thres]

    return keep_boxes, max_conf_in_batch


def calculate_iou(box1, box2):
    # box: [x1, y1, x2, y2, conf]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def main():
    # 1. 加载模型
    print(f"🚀 正在加载模型: {MODEL_PATH}")
    model = YOLOv1(num_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. 准备输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. 遍历文件夹
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"⚠️ 在 '{INPUT_DIR}' 中没有找到图片！")
        return

    print(f"📂 找到 {len(image_files)} 张图片，开始预测...")
    print(f"⚙️ 当前阈值设置: {CONF_THRESHOLD}")
    print("-" * 30)

    with torch.no_grad():
        for img_name in image_files:
            img_path = os.path.join(INPUT_DIR, img_name)

            # 预处理
            input_tensor, (orig_h, orig_w) = preprocess_image(img_path, IMG_SIZE)
            if input_tensor is None:
                continue

            # 推理
            prediction = model(input_tensor)

            # NMS 后处理
            # 【修改点2】接收返回的 max_conf 用于打印
            boxes, max_conf = non_max_suppression(prediction, CONF_THRESHOLD, IOU_THRESHOLD)

            # 打印调试信息
            print(f"🖼️ 图片: {img_name:<20} | 最大置信度: {max_conf:.4f} | 检出框数: {len(boxes)}")

            # 读取原图用于绘图
            image = cv2.imread(img_path)

            # 绘制边界框
            for box in boxes:
                # 还原坐标到原图尺寸
                x1 = int(box[0] * orig_w)
                y1 = int(box[1] * orig_h)
                x2 = int(box[2] * orig_w)
                y2 = int(box[3] * orig_h)
                conf = box[4]

                # 画框 (绿色)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 写文字
                label = f"Person: {conf:.2f}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 保存结果
            save_path = os.path.join(OUTPUT_DIR, img_name)
            cv2.imwrite(save_path, image)

    print("-" * 30)
    print(f"🎉 所有图片处理完毕，结果保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()