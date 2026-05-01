# 版本1：基础版， —— 支持 train / val / test 的原版 R-CNN 修正实现。
# 后续只会在基础班上更改，让ROI更丰富（原始代码中Selective Search只运行一次并缓存，导致每个epoch都使用相同的ROI，失去了数据增强的多样性，并且可能错过一些区域。
# 改进方案：预计算更多ROI（例如，使用快速模式和质量模式，或者多次运行），然后在训练时随机选择一部分，以增加多样性。）

## 原始R-CNN论文中的做法，每张图片都需要：
# 1. 运行 Selective Search → 生成约2000个候选区域，默认保存RIO一次到缓存，后续重新训练固定。
# 2. 这些候选区域在整个训练过程中都保持不变
# 3. 每个epoch使用相同的2000个区域
import os, cv2, joblib, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T, torchvision.models as M
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import torch.cuda.amp as amp
import time

VOC_CLASSES = ['background'] + \
              'aeroplane bicycle bird boat bottle bus car cat chair cow diningtable ' \
              'dog horse motorbike person pottedplant sheep sofa train tvmonitor'.split()
NUM_CLS = len(VOC_CLASSES)


# ==================== 配置参数 ====================
class Config:
    # GPU配置
    use_multi_gpu = False  # 是否使用多GPU训练
    gpu_ids = [0]  # 使用的GPU ID列表
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 数据配置
    cache_dir = './cache'
    max_rois = 2000  # 每张图片最大ROI数量
    use_cache = True  # 是否缓存ROI

    # 训练参数
    num_epochs = 10
    learning_rate = 1e-3
    weight_decay = 5e-4
    momentum = 0.9

    # 数据加载参数
    num_workers = 6  # 根据您的CPU核心数调整
    pin_memory = True
    prefetch_factor = 2

    # 模型参数
    backbone = 'alexnet'  # 'alexnet' 或 'resnet34'

    # 混合精度训练
    use_amp = True

    # 梯度累积
    accumulation_steps = 1

    # ROI处理批大小
    roi_batch_size = 512


config = Config()


# -------------------- 1. 数据集类 --------------------
class VOCDataset(Dataset):
    def __init__(self, img_dir, annot_dir, split_txt, transform=None):
        with open(split_txt) as f:
            ids = [x.strip() for x in f if x.strip()]
        self.ids = ids
        self.img_dir = img_dir
        self.annot_dir = annot_dir
        self.transform = transform

        # 创建缓存目录
        os.makedirs(config.cache_dir, exist_ok=True)

        # 加载或计算ROI缓存
        self.rois_cache = {}
        if config.use_cache:
            cache_file = os.path.join(config.cache_dir, f"rois_{os.path.basename(split_txt).split('.')[0]}.pkl")
            if os.path.exists(cache_file):
                print(f"加载缓存的ROIs: {cache_file}")
                self.rois_cache = joblib.load(cache_file)
            else:
                print(f"预计算ROIs并缓存到 {cache_file}...")
                self._precompute_rois(cache_file)

    def _precompute_rois(self, cache_file):
        """预计算Selective Search结果"""
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

        for img_id in tqdm(self.ids, desc="预计算ROIs"):
            img_path = os.path.join(self.img_dir, img_id + '.jpg')
            img_cv = cv2.imread(img_path)
            if img_cv is None:
                continue

            H, W = img_cv.shape[:2]
            ss.setBaseImage(img_cv)
            ss.switchToSelectiveSearchFast()  # 使用快速模式

            try:
                rects = ss.process()
            except:
                continue

            rois = []
            for x, y, w, h in rects[:config.max_rois]:
                if w < 20 or h < 20:
                    continue
                x2, y2 = x + w, y + h
                x, x2 = max(0, x), min(W, x2)
                y, y2 = max(0, y), min(H, y2)
                if x2 > x and y2 > y:  # 确保有效区域
                    rois.append([x, y, x2, y2])

            self.rois_cache[img_id] = np.array(rois, dtype=np.float32)

        joblib.dump(self.rois_cache, cache_file)
        print(f"ROIs缓存已保存到: {cache_file}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        # 加载图像
        img_path = os.path.join(self.img_dir, img_id + '.jpg')
        img = Image.open(img_path).convert('RGB')
        img_cv = cv2.imread(img_path)
        H, W = img_cv.shape[:2]

        # 加载标注
        gt_boxes, gt_labels = [], []
        annot_path = os.path.join(self.annot_dir, img_id + '.txt')
        if os.path.exists(annot_path):
            with open(annot_path) as f:
                for line in f:
                    x1, y1, x2, y2, c = line.strip().split()
                    x1, y1, x2, y2, c = map(float, (x1, y1, x2, y2, c))
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
                    gt_boxes.append([x1, y1, x2, y2])
                    gt_labels.append(int(c))

        gt_boxes = np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.zeros((0, 4), dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64) if gt_labels else np.zeros(0, dtype=np.int64)

        # 获取ROIs
        rois = self.rois_cache.get(img_id, np.zeros((0, 4), dtype=np.float32))

        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = T.ToTensor()(img)

        return img_tensor, rois, gt_boxes, gt_labels, img_id


# -------------------- 2. 模型 --------------------
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        if config.backbone == 'resnet34':
            # 使用ResNet34
            resnet = M.resnet34(weights=M.ResNet34_Weights.IMAGENET1K_V1)
            self.features = nn.Sequential(*list(resnet.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d((6, 6))
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True)
            )
        else:
            # 原始AlexNet
            # alex = M.alexnet(weights=M.AlexNet_Weights.IMAGENET1K_V1)
            alex = M.alexnet(pretrained=True)   #旧版API
            self.features = alex.features
            self.fc = nn.Sequential(*list(alex.classifier.children())[:-1])

        # 冻结卷积层
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        if hasattr(self, 'pool'):
            x = self.pool(x)
        return self.fc(x.flatten(1))


class RCNN(nn.Module):
    def __init__(self, feat_extractor):
        super().__init__()
        self.feat = feat_extractor
        self.cls = nn.Linear(4096, NUM_CLS)
        self.reg = nn.Linear(4096, NUM_CLS * 4)

    def forward(self, x):
        feat = self.feat(x)
        return self.cls(feat), self.reg(feat)


# -------------------- 3. ROI处理器 --------------------
class ROIBatchProcessor:
    """批量处理ROI裁剪"""

    def __init__(self, device, crop_size=227):
        self.device = device
        self.crop_size = crop_size
        self.normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.to_tensor = T.ToTensor()

    def extract_crops(self, img_np, rois):
        """批量提取ROI crops"""
        crops = []

        for i in range(0, len(rois), config.roi_batch_size):
            batch_rois = rois[i:i + config.roi_batch_size]
            batch_crops = []

            for roi in batch_rois:
                x1, y1, x2, y2 = map(int, roi)
                # 边界检查
                if x2 <= x1 or y2 <= y1:
                    crop = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
                else:
                    crop = img_np[y1:y2, x1:x2]
                    if crop.size == 0:
                        crop = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
                    else:
                        crop = cv2.resize(crop, (self.crop_size, self.crop_size))

                # 转换为tensor并归一化
                crop_tensor = self.normalize(self.to_tensor(crop))
                batch_crops.append(crop_tensor)

            if batch_crops:
                crops.append(torch.stack(batch_crops).to(self.device))

        return torch.cat(crops, dim=0) if crops else torch.tensor([], device=self.device)


# -------------------- 4. 训练辅助函数 --------------------
def assign_labels(rois, gt_boxes, gt_labels, pos_th=0.5, neg_th=0.3):
    """返回 labels, bbox_targets, valid_mask"""
    N = len(rois)
    labels = np.zeros(N, dtype=np.int64)
    bbox_targets = np.zeros((N, 4), dtype=np.float32)

    if len(gt_boxes) == 0:
        return labels, bbox_targets, np.ones(N, dtype=bool)

    ious = np.array([[compute_iou(r, g) for g in gt_boxes] for r in rois])
    max_ious = ious.max(1)
    max_idx = ious.argmax(1)

    # 正/负样本
    pos_mask = max_ious >= pos_th
    neg_mask = max_ious < neg_th
    labels[pos_mask] = gt_labels[max_idx[pos_mask]]

    # 回归目标
    for i in np.where(pos_mask)[0]:
        r, g = rois[i], gt_boxes[max_idx[i]]
        x1, y1, x2, y2 = r
        w, h = x2 - x1, y2 - y1
        gx1, gy1, gx2, gy2 = g
        gw, gh = gx2 - gx1, gy2 - gy1

        tx = (gx1 - x1) / w
        ty = (gy1 - y1) / h
        tw = np.log(gw / w)
        th = np.log(gh / h)
        bbox_targets[i] = [tx, ty, tw, th]

    valid = pos_mask | neg_mask
    return labels[valid], bbox_targets[valid], valid


def compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    sa = (a[2] - a[0]) * (a[3] - a[1])
    sb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (sa + sb - inter + 1e-10)


# -------------------- 5. 训练循环 --------------------
def train_one_epoch(loader, model, opt, cls_loss_fn, reg_loss_fn, device, roi_processor, scaler=None):
    model.train()
    cls_tot, reg_tot, samples = 0., 0., 0

    opt.zero_grad()

    for batch_idx, (img, rois, gt_boxes, gt_labels, _) in enumerate(tqdm(loader, ncols=80, desc="训练")):
        # 数据转移到GPU
        img = img.to(device, non_blocking=True)
        rois, gt_boxes, gt_labels = rois[0], gt_boxes[0], gt_labels[0]

        # 分配标签
        labels, targets, mask = assign_labels(rois.numpy(), gt_boxes.numpy(), gt_labels.numpy())
        if mask.sum() == 0:
            continue

        valid_rois = rois[mask].numpy()

        # 提取ROI crops
        img_np = (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        crops = roi_processor.extract_crops(img_np, valid_rois)

        if len(crops) == 0:
            continue

        # 准备标签
        labels_tensor = torch.from_numpy(labels).to(device)
        targets_tensor = torch.from_numpy(targets).to(device)

        # 混合精度训练
        with amp.autocast(enabled=scaler is not None):
            cls_pred, reg_pred = model(crops)

            # 分类损失
            cls_loss = cls_loss_fn(cls_pred, labels_tensor)

            # 回归损失（仅正样本）
            pos_mask = labels_tensor != 0
            reg_loss = torch.tensor(0.0, device=device)

            if pos_mask.sum() > 0:
                idx = labels_tensor[pos_mask, None] * 4 + torch.arange(4, device=device)
                reg_pred_pos = reg_pred[pos_mask].gather(1, idx)
                reg_loss = reg_loss_fn(reg_pred_pos, targets_tensor[pos_mask])

            # 总损失
            loss = (cls_loss + 10 * reg_loss) / config.accumulation_steps

        # 反向传播
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 梯度累积
        if (batch_idx + 1) % config.accumulation_steps == 0:
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad()

        # 统计
        cls_tot += cls_loss.item() * len(labels)
        reg_tot += reg_loss.item() * pos_mask.sum().item() if pos_mask.sum() > 0 else 0.
        samples += len(labels)

    # 处理剩余的梯度
    if scaler is not None:
        scaler.step(opt)
        scaler.update()
    else:
        opt.step()

    return cls_tot / max(samples, 1), reg_tot / max(samples, 1)


# -------------------- 6. 验证函数 --------------------
@torch.no_grad()
def validate(loader, model, device, roi_processor):
    model.eval()
    all_scores, all_labels = [], []

    for img, rois, gt_boxes, gt_labels, _ in tqdm(loader, ncols=80, desc="验证"):
        img = img.to(device, non_blocking=True)
        rois, gt_boxes, gt_labels = rois[0], gt_boxes[0], gt_labels[0]

        labels, _, mask = assign_labels(rois.numpy(), gt_boxes.numpy(), gt_labels.numpy())
        if mask.sum() == 0:
            continue

        valid_rois = rois[mask].numpy()

        # 提取ROI crops
        img_np = (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        crops = roi_processor.extract_crops(img_np, valid_rois)

        if len(crops) > 0:
            # 混合精度推理
            with amp.autocast(enabled=config.use_amp):
                cls_scores = model(crops)[0]
                # 前景最大概率
                fg_scores = cls_scores.softmax(1)[:, 1:].max(1)[0]

            all_scores.append(fg_scores.cpu())
            all_labels.append(torch.from_numpy(labels != 0))

    if len(all_scores) == 0:
        return 0.0

    all_scores = torch.cat(all_scores)
    all_labels = torch.cat(all_labels)

    return average_precision_score(all_labels.numpy(), all_scores.numpy())


# -------------------- 7. 主函数 --------------------
def main():
    print("=" * 60)
    print("R-CNN 训练配置")
    print("=" * 60)

    # 检查GPU
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个GPU")

    # 设置设备
    if config.use_multi_gpu and num_gpus > 1:
        print(f"使用多GPU训练，GPU IDs: {config.gpu_ids}")
        device = torch.device(config.device)
    else:
        print(f"使用单GPU训练，设备: {config.device}")
        device = torch.device(config.device)

    # 路径设置
    VOC = './VOC2012'
    img_dir = os.path.join(VOC, 'JPEGImages')
    annot_dir = os.path.join(VOC, 'Annotations_txt')
    train_txt = os.path.join(VOC, 'ImageSets/Main/train.txt')
    val_txt = os.path.join(VOC, 'ImageSets/Main/val.txt')

    # 数据变换
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 创建数据集
    print("创建数据集...")
    train_set = VOCDataset(img_dir, annot_dir, train_txt, transform)
    val_set = VOCDataset(img_dir, annot_dir, val_txt, transform)

    # 数据加载器
    train_loader = DataLoader(
        train_set,
        batch_size=1,  # R-CNN保持batch_size=1
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    # 初始化模型
    print(f"初始化R-CNN模型，使用 {config.backbone} 作为backbone...")
    feat_extractor = FeatureExtractor()
    model = RCNN(feat_extractor)

    # 多GPU支持
    if config.use_multi_gpu and num_gpus > 1:
        from torch.nn.parallel import DataParallel
        model = DataParallel(model, device_ids=config.gpu_ids)

    model = model.to(device)

    # 优化器
    opt = torch.optim.SGD([
        {'params': model.module.feat.fc.parameters() if config.use_multi_gpu else model.feat.fc.parameters(),
         'lr': config.learning_rate},
        {'params': model.module.cls.parameters() if config.use_multi_gpu else model.cls.parameters(),
         'lr': config.learning_rate * 10},
        {'params': model.module.reg.parameters() if config.use_multi_gpu else model.reg.parameters(),
         'lr': config.learning_rate * 10}
    ], momentum=config.momentum, weight_decay=config.weight_decay)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)

    # 混合精度训练
    scaler = amp.GradScaler() if config.use_amp else None

    # ROI处理器
    roi_processor = ROIBatchProcessor(device, crop_size=227)

    # 损失函数
    cls_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.SmoothL1Loss()

    # 训练
    best_ap = 0
    print(f"\n开始训练，共 {config.num_epochs} 个epoch")
    print("=" * 60)

    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()
        print(f'\n----- Epoch {epoch}/{config.num_epochs} -----')

        # 训练
        tr_cls, tr_reg = train_one_epoch(
            train_loader, model, opt,
            cls_loss_fn, reg_loss_fn,
            device, roi_processor, scaler
        )

        # 验证
        val_ap = validate(val_loader, model, device, roi_processor)

        # 更新学习率
        scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = opt.param_groups[0]['lr']

        print(f"训练时间: {epoch_time:.1f}s | 学习率: {current_lr:.2e}")
        print(f"训练损失 - 分类: {tr_cls:.4f}, 回归: {tr_reg:.4f}")
        print(f"验证AP: {val_ap:.4f}")

        # 显存使用情况
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            memory_reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"GPU显存使用: {memory_allocated:.1f}/{memory_reserved:.1f} GB")

        # 保存最佳模型
        if val_ap > best_ap:
            best_ap = val_ap
            # 保存时去除DataParallel包装
            if config.use_multi_gpu and isinstance(model, nn.DataParallel):
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()

            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_ap': best_ap,
            }, 'rcnn_best.pth')
            print(f"保存最佳模型, AP: {val_ap:.4f}")

        # 定期保存检查点
        if epoch % 5 == 0:
            checkpoint_path = f'rcnn_checkpoint_epoch_{epoch}.pth'
            if config.use_multi_gpu and isinstance(model, nn.DataParallel):
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()

            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'ap': val_ap,
            }, checkpoint_path)
            print(f"检查点已保存: {checkpoint_path}")

        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"训练完成! 最佳AP: {best_ap:.4f}")

    # 最终评估
    print("\n在验证集上进行最终评估...")
    final_ap = validate(val_loader, model, device, roi_processor)
    print(f"最终验证AP: {final_ap:.4f}")


if __name__ == '__main__':
    # 设置CUDA优化参数
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    main()
