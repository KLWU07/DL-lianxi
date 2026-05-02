import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np


class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, S=7, img_size=448, augment=True):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.S = S
        self.img_size = img_size
        self.augment = augment
        # 获取文件列表
        self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # 1. 读取图片
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = cv2.imread(img_path)
        h, w = img.shape[:2]  # 获取原始宽高

        # 2. 读取标签 (先读取为原始归一化坐标)
        label_path = os.path.join(self.label_dir, self.img_files[idx].replace('.jpg', '.txt').replace('.png', '.txt'))
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        x_center, y_center, w_box, h_box = map(float, parts[1:5])
                        boxes.append([cls, x_center, y_center, w_box, h_box])

        # ==========================================
        # 3. Letterbox 处理 (核心修改部分)
        # ==========================================
        # 计算缩放比例 (保持长宽比)
        scale = min(self.img_size / w, self.img_size / h)

        # 计算缩放后的新宽高
        new_w, new_h = int(w * scale), int(h * scale)

        # 执行缩放
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 创建 448x448 的灰色背景 (114 是 YOLO 默认填充色，也可以设为 0 黑色)
        padded_img = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)

        # --- 修复点：使用 // 进行整除，确保结果是整数 ---
        pad_x = (self.img_size - new_w) // 2
        pad_y = (self.img_size - new_h) // 2

        # 将缩放后的图片粘贴到背景上
        padded_img[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img

        # ==========================================
        # 4. 同步修正标签坐标
        # ==========================================
        for box in boxes:
            # box: [cls, x_center, y_center, w, h]

            # 1. 缩放坐标：先乘以缩放比例，映射到新尺寸
            box[1] = box[1] * w * scale
            box[2] = box[2] * h * scale

            # 2. 偏移坐标：加上黑边的偏移量
            box[1] += pad_x
            box[2] += pad_y

            # 3. 重新归一化：除以最终画布尺寸 (448)
            box[1] /= self.img_size
            box[2] /= self.img_size

            # 4. 宽高也需要重新归一化 (宽高不受偏移影响，只受缩放影响)
            box[3] = box[3] * w * scale / self.img_size
            box[4] = box[4] * h * scale / self.img_size

        # 5. 数据增强 (水平翻转)
        if self.augment and np.random.rand() > 0.5:
            padded_img = cv2.flip(padded_img, 1)
            # 翻转时同步修正坐标
            for box in boxes:
                box[1] = 1.0 - box[1]

        # 6. 转换为 Tensor
        img_tensor = torch.from_numpy(padded_img).permute(2, 0, 1).float() / 255.0

        # 7. 构建 YOLO Target
        num_classes = 1
        target = torch.zeros((self.S, self.S, 5 + num_classes))

        for box in boxes:
            cls, x_center, y_center, w_box, h_box = box
            cls = int(cls)

            grid_x = int(x_center * self.S)
            grid_y = int(y_center * self.S)

            if grid_x < self.S and grid_y < self.S:
                if target[grid_y, grid_x, 0] == 0:
                    target[grid_y, grid_x, 0] = 1
                    # 计算相对于格子的偏移量
                    x_offset = x_center * self.S - grid_x
                    y_offset = y_center * self.S - grid_y

                    target[grid_y, grid_x, 1:5] = torch.tensor([x_offset, y_offset, w_box, h_box])
                    target[grid_y, grid_x, 5 + cls] = 1

        return img_tensor, target