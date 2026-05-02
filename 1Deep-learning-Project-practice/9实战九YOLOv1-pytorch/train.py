import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from yolo import YOLOv1
from dataset import YOLODataset


# ==========================================
# 1. 定义 YOLO Loss 类 (保持不变)
# ==========================================
class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, num_classes=1, lambda_coord=5, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.S = S
        self.B = B
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, preds, targets):
        batch_size = preds.size(0)
        total_loss = 0

        for b in range(batch_size):
            pred = preds[b]
            target = targets[b]

            for i in range(self.S):
                for j in range(self.S):
                    cell_target = target[i, j]
                    has_obj = cell_target[0] > 0.5

                    pred_class = pred[i, j, self.B * 5:]
                    target_class = int(cell_target[5])

                    if has_obj:
                        bbox_idx = 0
                        pred_x = pred[i, j, bbox_idx * 5]
                        pred_y = pred[i, j, bbox_idx * 5 + 1]
                        pred_w = pred[i, j, bbox_idx * 5 + 2]
                        pred_h = pred[i, j, bbox_idx * 5 + 3]
                        pred_conf = pred[i, j, bbox_idx * 5 + 4]

                        target_x = cell_target[1]
                        target_y = cell_target[2]
                        target_w = cell_target[3]
                        target_h = cell_target[4]

                        coord_loss = (pred_x - target_x) ** 2 + (pred_y - target_y) ** 2
                        coord_loss += (pred_w - target_w) ** 2 + (pred_h - target_h) ** 2
                        coord_loss *= self.lambda_coord

                        conf_loss = (pred_conf - 1.0) ** 2

                        class_loss = 0
                        for c in range(self.num_classes):
                            label = 1.0 if c == target_class else 0.0
                            class_loss += (pred_class[c] - label) ** 2

                        total_loss += coord_loss + conf_loss + class_loss

                    else:
                        for b_idx in range(self.B):
                            pred_conf = pred[i, j, b_idx * 5 + 4]
                            total_loss += (pred_conf ** 2) * self.lambda_noobj

        return total_loss / batch_size


# ==========================================
# 2. 训练主函数 (重点修改了保存逻辑)
# ==========================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    model = YOLOv1(num_classes=1, S=7, B=2).to(device)
    criterion = YOLOLoss(S=7, B=2, num_classes=1)
    # 学习率调小，防止 Loss 变成 0
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    train_dataset = YOLODataset('./dataset/images/train', './dataset/labels/train', S=7, augment=True)
    val_dataset = YOLODataset('./dataset/images/val', './dataset/labels/val', S=7, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    # --- 新增：用于记录最优模型的变量 ---
    best_val_loss = float('inf')  # 初始化为无穷大
    best_epoch = 0

    print("🔥 开始训练...")
    # 增加轮数，因为学习率变小了
    for epoch in range(100):
        # --- 训练阶段 ---
        model.train()
        total_train_loss = 0

        for batch_idx, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # --- 验证阶段 ---
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                preds = model(imgs)
                loss = criterion(preds, targets)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch [{epoch + 1}/100] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- 核心修改：保存最优模型逻辑 ---
        # 如果当前验证集 Loss 比历史最优还低，就保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1

            # 使用 state_dict() 保存，文件体积小 (约 200MB)
            # 只保存权重参数，不保存模型结构和 Python 环境
            torch.save(model.state_dict(), "yolov1_best.pth")
            print(f"✨ 发现更好的模型！验证集 Loss 降至 {avg_val_loss:.4f}，已保存为 yolov1_best.pth")

    print(f"🏁 训练结束！最优模型出现在第 {best_epoch} 轮，最低验证集 Loss 为 {best_val_loss:.4f}")


if __name__ == "__main__":
    train()