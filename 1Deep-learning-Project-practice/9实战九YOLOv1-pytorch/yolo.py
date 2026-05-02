import torch
import torch.nn as nn


class YOLOv1(nn.Module):
    def __init__(self, num_classes=1, S=7, B=2):
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.num_classes = num_classes

        # 特征提取网络
        # 注意：这里去掉了 AdaptiveAvgPool2d，因为我们要在最后时刻才进行压缩
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3), nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            nn.Conv2d(64, 192, 3, 1, 1), nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            nn.Conv2d(192, 128, 1), nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 3, 1, 1), nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 3, 1, 1), nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, 1, 1), nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            nn.Conv2d(512, 256, 1), nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, 1, 1), nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 1), nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, 1, 1), nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 1), nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, 1, 1), nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 1), nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, 1, 1), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 512, 1), nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, 3, 1, 1), nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 512, 1), nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, 3, 1, 1), nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, 3, 1, 1), nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, 3, 2, 1), nn.LeakyReLU(0.1),
        )

        # --- 修改重点 1：全连接层输入维度 ---
        # 原代码是 nn.Linear(1024 * 7 * 7, 4096) -> 导致 900MB+
        # 修改后是 nn.Linear(1024, 4096) -> 配合下面的池化，体积将降至约 20MB
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 4096),  # 输入变成了 1024 (1x1x1024)
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (B * 5 + num_classes))
        )

    def forward(self, x):
        x = self.features(x)

        # --- 修改重点 2：在 forward 中加入池化 ---
        # 将特征图从 [Batch, 1024, 7, 7] 压缩为 [Batch, 1024, 1, 1]
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))

        x = self.fc(x)

        # 重塑形状为 [Batch, 7, 7, 30]
        out = x.view(x.size(0), self.S, self.S, self.B * 5 + self.num_classes)

        # --- 激活函数处理 ---
        # 1. 坐标 (x, y) -> Sigmoid (0~1)
        out[..., :self.B * 4] = torch.sigmoid(out[..., :self.B * 4])
        # 2. 置信度 -> Sigmoid (0~1)
        out[..., self.B * 4: self.B * 5] = torch.sigmoid(out[..., self.B * 4: self.B * 5])
        # 3. 类别 -> Sigmoid (0~1)
        out[..., self.B * 5:] = torch.sigmoid(out[..., self.B * 5:])

        return out