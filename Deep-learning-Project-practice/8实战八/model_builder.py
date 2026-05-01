import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(num_classes):
    # 加载预训练模型
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # 获取分类器的输入特征数
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # 替换预训练的头部
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model