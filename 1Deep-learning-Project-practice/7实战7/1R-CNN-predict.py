#!/usr/bin/env python3
"""
R-CNN 预测 & 可视化
author : you
"""
import os
import cv2
import torch
import joblib
import argparse
import numpy as np
import torchvision.transforms as T
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt

# --------------- 与训练代码一致的配置 ---------------
VOC_CLASSES = ['background'] + \
              'aeroplane bicycle bird boat bottle bus car cat chair cow diningtable ' \
              'dog horse motorbike person pottedplant sheep sofa train tvmonitor'.split()
NUM_CLS = len(VOC_CLASSES)

# --------------- 模型定义（与训练代码保持一致） ---------------
class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision.models as M
        alex = M.alexnet(pretrained=False)   # 只载结构，权重由ckpt提供
        self.features = alex.features
        self.fc = torch.nn.Sequential(*list(alex.classifier.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.fc(x)

class RCNN(torch.nn.Module):
    def __init__(self, feat_extractor):
        super().__init__()
        self.feat = feat_extractor
        self.cls = torch.nn.Linear(4096, NUM_CLS)
        self.reg = torch.nn.Linear(4096, NUM_CLS * 4)

    def forward(self, x):
        feat = self.feat(x)
        return self.cls(feat), self.reg(feat)

# --------------- 通用工具 ---------------
def compute_iou(a, b):
    x1, y1, x2, y2 = np.maximum(a[:4], b[:4]), np.maximum(a[1], b[1]), \
                      np.minimum(a[2], b[2]), np.minimum(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    sa, sb = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return inter / (sa + sb - inter + 1e-10)

def nms(dets, thresh=0.3):
    """纯Python NMS，dets: [[x1,y1,x2,y2,score],...]"""
    if len(dets) == 0:
        return []
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        w, h = np.maximum(0, xx2 - xx1), np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)
        order = order[1:][iou <= thresh]
    return dets[keep]

# --------------- 单张图像推理 ---------------
def inference_one(model, roi_processor, img_cv, rois, device,
                  conf_thresh=0.6, nms_thresh=0.3):
    """返回 [[x1,y1,x2,y2,label,score], ...]"""
    H, W = img_cv.shape[:2]
    img_tensor = T.ToTensor()(Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)))
    img_tensor = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_tensor).unsqueeze(0).to(device)

    # 提取ROI crops
    crops = roi_processor(img_cv, rois)        # Tensor[N,3,227,227]
    if len(crops) == 0:
        return []

    # 前向
    with torch.no_grad():
        cls_logit, reg_delta = model(crops)
        cls_prob = torch.softmax(cls_logit, dim=1)
    scores, labels = cls_prob[:, 1:].max(1)   # 跳过背景
    labels += 1                                 # 0-based -> 1-based
    reg_delta = reg_delta.view(-1, NUM_CLS, 4)  # [N,20,4]
    reg_delta = reg_delta[torch.arange(len(reg_delta)), labels]  # [N,4]

    # 解码回归
    rois = torch.from_numpy(rois).float()
    widths, heights = rois[:, 2] - rois[:, 0], rois[:, 3] - rois[:, 1]
    dx, dy, dw, dh = reg_delta[:, 0], reg_delta[:, 1], reg_delta[:, 2], reg_delta[:, 3]
    x1 = rois[:, 0] + dx * widths
    y1 = rois[:, 1] + dy * heights
    x2 = rois[:, 2] + dx * widths
    y2 = rois[:, 3] + dy * heights
    boxes = torch.stack([x1, y1, x2, y2], dim=1)

    # 裁剪到图像
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, W)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, H)

    # 组装
    dets = torch.cat([boxes, scores.unsqueeze(1), labels.float().unsqueeze(1)], dim=1).cpu().numpy()
    # 过滤置信度
    dets = dets[dets[:, 4] > conf_thresh]
    # NMS
    keep = nms(dets[:, :5], nms_thresh)
    return keep

# --------------- ROI 处理器（与训练代码一致） ---------------
class ROIBatchProcessor:
    def __init__(self, device, crop_size=227):
        self.device = device
        self.crop_size = crop_size
        self.normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.to_tensor = T.ToTensor()

    def __call__(self, img_cv, rois):
        crops = []
        for roi in rois:
            x1, y1, x2, y2 = map(int, roi)
            if x2 <= x1 or y2 <= y1:
                crop = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
            else:
                crop = img_cv[y1:y2, x1:x2]
                if crop.size == 0:
                    crop = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
                else:
                    crop = cv2.resize(crop, (self.crop_size, self.crop_size))
            crop_tensor = self.normalize(self.to_tensor(crop)).unsqueeze(0)
            crops.append(crop_tensor)
        if not crops:
            return torch.tensor([], device=self.device)
        return torch.cat(crops).to(self.device)

# --------------- Selective Search  ---------------
def get_rois(img_cv, max_rois=2000):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img_cv)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()[:max_rois]
    rois = []
    H, W = img_cv.shape[:2]
    for x, y, w, h in rects:
        if w < 20 or h < 20:
            continue
        x2, y2 = x + w, y + h
        x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x2), min(H, y2)
        if x2 > x and y2 > y:
            rois.append([x1, y1, x2, y2])
    return np.array(rois, dtype=np.float32)

# --------------- 可视化 ---------------
def draw(img, dets, thickness=2, font_scale=0.6):
    """dets: [[x1,y1,x2,y2,score,label], ...]"""
    for x1, y1, x2, y2, score, label in dets:
        label = int(label)
        cls_name = VOC_CLASSES[label]
        color = tuple(map(int, plt.cm.tab20(label % 20)[:3] * 255))
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        text = f'{cls_name}:{score:.2f}'
        t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
        cv2.rectangle(img, (int(x1), int(y1) - t_size[1] - 4),
                      (int(x1) + t_size[0], int(y1)), color, -1)
        cv2.putText(img, text, (int(x1), int(y1) - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
    return img

# --------------- 主入口 ---------------
def main():
    parser = argparse.ArgumentParser('R-CNN Predict')
    parser.add_argument('--weights', required=True, help='rcnn_best.pth')
    parser.add_argument('--source', required=True, help='image/folder/video/0')
    parser.add_argument('--mode', choices=['image', 'folder', 'video'], default='image')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--conf', type=float, default=0.6, help='confidence threshold')
    parser.add_argument('--nms', type=float, default=0.3, help='nms threshold')
    parser.add_argument('--out', default='output', help='output folder')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 加载模型
    feat = FeatureExtractor()
    model = RCNN(feat)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    roi_processor = ROIBatchProcessor(device)

    # ------------- 模式分支 -------------
    if args.mode == 'image':
        img_cv = cv2.imread(args.source)
        assert img_cv is not None, 'Image not found'
        rois = get_rois(img_cv)
        dets = inference_one(model, roi_processor, img_cv, rois, device, args.conf, args.nms)
        vis = draw(img_cv.copy(), dets)
        out_path = os.path.join(args.out, os.path.basename(args.source))
        cv2.imwrite(out_path, vis)
        print(f'Saved to {out_path}')

    elif args.mode == 'folder':
        img_list = [os.path.join(args.source, x) for x in os.listdir(args.source)
                    if x.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for path in tqdm(img_list, desc='Infer'):
            img_cv = cv2.imread(path)
            if img_cv is None:
                continue
            rois = get_rois(img_cv)
            dets = inference_one(model, roi_processor, img_cv, rois, device, args.conf, args.nms)
            vis = draw(img_cv.copy(), dets)
            out_path = os.path.join(args.out, os.path.basename(path))
            cv2.imwrite(out_path, vis)
        print(f'All results saved to {args.out}')

    elif args.mode == 'video':
        cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_vid = os.path.join(args.out, 'result.mp4')
        writer = cv2.VideoWriter(out_vid, fourcc, fps, (w, h))
        print('Press q to quit.')
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rois = get_rois(frame)
            dets = inference_one(model, roi_processor, frame, rois, device, args.conf, args.nms)
            vis = draw(frame, dets)
            writer.write(vis)
            cv2.imshow('rcnn', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print(f'Saved video to {out_vid}')

if __name__ == '__main__':
    main()