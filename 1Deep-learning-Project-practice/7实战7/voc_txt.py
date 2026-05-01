import os
import xml.etree.ElementTree as ET

# VOC2012的20个类别（按顺序对应class_id 1-20）
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


def xml_to_txt(xml_dir, txt_dir):
    """
    将VOC XML标注转换为TXT格式
    :param xml_dir: XML标注文件所在目录（如VOCdevkit/VOC2007/Annotations）
    :param txt_dir: 输出TXT文件的保存目录
    """
    # 创建输出目录（如果不存在）
    os.makedirs(txt_dir, exist_ok=True)

    # 遍历所有XML文件
    for xml_filename in os.listdir(xml_dir):
        if not xml_filename.endswith('.xml'):
            continue  # 只处理.xml文件

        # 解析XML
        xml_path = os.path.join(xml_dir, xml_filename)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取图像尺寸（可选，用于验证坐标是否合理）
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        # 提取所有目标的标注
        txt_content = []
        for obj in root.iter('object'):
            # 获取类别名称
            cls_name = obj.find('name').text.strip().lower()
            if cls_name not in VOC_CLASSES:
                continue  # 跳过不在VOC类别中的目标

            # 转换类别名称为class_id（1-20）
            cls_id = VOC_CLASSES.index(cls_name) + 1  # 索引+1，确保从1开始

            # 获取边界框坐标（xmin, ymin, xmax, ymax）
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)

            # 确保坐标在图像范围内（可选，防止越界）
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(x1, min(x2, width))
            y2 = max(y1, min(y2, height))

            # 写入TXT内容（格式：x1 y1 x2 y2 class_id）
            txt_content.append(f"{x1} {y1} {x2} {y2} {cls_id}")

        # 保存为TXT文件（与XML同名，后缀改为.txt）
        txt_filename = xml_filename.replace('.xml', '.txt')
        txt_path = os.path.join(txt_dir, txt_filename)
        with open(txt_path, 'w') as f:
            f.write('\n'.join(txt_content))

        # 打印进度
        if (len(os.listdir(txt_dir)) % 100 == 0):
            print(f"已转换 {len(os.listdir(txt_dir))} 个文件...")

    print(f"转换完成！共处理 {len(os.listdir(txt_dir))} 个XML文件，保存至 {txt_dir}")


# -------------------------- 运行转换脚本 --------------------------
if __name__ == "__main__":
    # 请替换为你的实际路径
    xml_dir = "./VOC2012/Annotations"  # VOC原始XML标注目录
    txt_dir = "./VOC2012/Annotations_txt"  # 输出TXT目录（需与RCNN代码中的annot_dir一致）

    # 执行转换
    xml_to_txt(xml_dir, txt_dir)