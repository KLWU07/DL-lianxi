import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform(train):
    if train:
        return A.Compose([
            A.Resize(height=800, width=800),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        return A.Compose([
            A.Resize(height=800, width=800),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))