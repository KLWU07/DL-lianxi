import os
import torch
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor


class VOCDataset(Dataset):
    def __init__(self, root_dir, split='train', transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_set_dir = os.path.join(root_dir, 'ImageSets', 'Main')
        self.img_dir = os.path.join(root_dir, 'JPEGImages')
        self.ann_dir = os.path.join(root_dir, 'Annotations')

        # 读取图像ID列表
        split_file = os.path.join(self.image_set_dir, f'{split}.txt')
        with open(split_file) as f:
            self.image_ids = [x.strip() for x in f.readlines()]

        # 定义类别名称
        self.classes = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, f'{image_id}.jpg')
        ann_path = os.path.join(self.ann_dir, f'{image_id}.xml')

        # 加载图像
        img = Image.open(img_path).convert('RGB')

        # 解析XML标注
        boxes, labels = self._parse_voc_xml(ann_path)

        # 转换为Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # 创建目标字典
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros_like(labels)
        }

        if self.transforms is not None:
            transformed = self.transforms(
                image=np.array(img),
                bboxes=target['boxes'].numpy(),
                labels=target['labels'].numpy()
            )
            img = Image.fromarray(transformed['image'])
            target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            target['labels'] = torch.as_tensor(transformed['labels'], dtype=torch.int64)

        img = ToTensor()(img)

        return img, target

    def _parse_voc_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall('object'):
            # 获取类别标签
            name = obj.find('name').text
            label = self.class_to_idx[name]

            # 获取边界框坐标
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        return boxes, labels