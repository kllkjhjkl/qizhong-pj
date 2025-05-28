import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from pycocotools import mask as maskUtils
import cv2
from collections import defaultdict

def convert_voc_to_coco(voc_root, output_file, split='train'):
    """
    将VOC2007格式的数据集转换为COCO格式
    
    Args:
        voc_root: VOC数据集根目录
        output_file: 输出的COCO格式json文件路径
        split: 数据集划分 ('train', 'val')
    """
    
    # COCO格式的基本结构
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # VOC2007的20个标准类别
    categories = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    # 添加类别信息
    for i, cat in enumerate(categories):
        coco_format["categories"].append({
            "id": i + 1,
            "name": cat,
            "supercategory": "object"
        })
    
    # 路径设置
    img_dir = os.path.join(voc_root, 'JPEGImages')
    ann_dir = os.path.join(voc_root, 'Annotations')
    seg_obj_dir = os.path.join(voc_root, 'SegmentationObject')
    seg_class_dir = os.path.join(voc_root, 'SegmentationClass')
    
    # 获取分割数据的图像列表
    split_file = os.path.join(voc_root, 'ImageSets', 'Segmentation', f'{split}.txt')
    if not os.path.exists(split_file):
        # 如果没有分割的划分文件，使用主划分文件
        split_file = os.path.join(voc_root, 'ImageSets', 'Main', f'{split}.txt')
    
    with open(split_file, 'r') as f:
        image_ids = [line.strip() for line in f.readlines()]
    
    annotation_id = 1
    
    for img_id, image_id in enumerate(image_ids, 1):
        # 检查图像是否存在
        img_path = os.path.join(img_dir, f'{image_id}.jpg')
        if not os.path.exists(img_path):
            continue
            
        # 读取图像信息
        img = Image.open(img_path)
        width, height = img.size
        
        # 添加图像信息
        coco_format["images"].append({
            "id": img_id,
            "file_name": f'{image_id}.jpg',
            "width": width,
            "height": height
        })
        
        # 读取XML标注
        xml_path = os.path.join(ann_dir, f'{image_id}.xml')
        if not os.path.exists(xml_path):
            continue
            
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 读取SegmentationObject掩码
        seg_obj_path = os.path.join(seg_obj_dir, f'{image_id}.png')
        
        seg_obj_mask = None
        if os.path.exists(seg_obj_path):
            seg_obj_mask = cv2.imread(seg_obj_path, cv2.IMREAD_GRAYSCALE)
        
        # 处理每个对象
        objects = root.findall('object')
        
        for obj_idx, obj in enumerate(objects):
            class_name = obj.find('name').text
            if class_name not in categories:
                continue
                
            category_id = categories.index(class_name) + 1
            
            # 获取边界框
            bbox_elem = obj.find('bndbox')
            xmin = int(float(bbox_elem.find('xmin').text))
            ymin = int(float(bbox_elem.find('ymin').text))
            xmax = int(float(bbox_elem.find('xmax').text))
            ymax = int(float(bbox_elem.find('ymax').text))
            
            # 确保边界框在图像范围内
            xmin = max(0, min(xmin, width - 1))
            ymin = max(0, min(ymin, height - 1))
            xmax = max(0, min(xmax, width - 1))
            ymax = max(0, min(ymax, height - 1))
            
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
            area = (xmax - xmin) * (ymax - ymin)
            
            if area <= 0:
                continue
            
            annotation = {
                "id": annotation_id,
                "image_id": img_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0
            }
            
            # 处理分割掩码 - 只使用SegmentationObject
            if seg_obj_mask is not None:
                # VOC格式中，SegmentationObject的像素值对应实例ID
                # 实例ID通常从1开始，按照XML中object的顺序递增
                instance_id = obj_idx + 1
                
                # 创建当前实例的二值掩码
                instance_mask = (seg_obj_mask == instance_id).astype(np.uint8)
                
                if instance_mask.sum() > 0:
                    # 转换为RLE格式
                    rle = maskUtils.encode(np.asfortranarray(instance_mask))
                    rle['counts'] = rle['counts'].decode('utf-8')
                    annotation["segmentation"] = rle
                    
                    # 重新计算基于掩码的面积
                    mask_area = int(instance_mask.sum())
                    annotation["area"] = mask_area
                    
                    # 根据掩码重新计算更精确的边界框
                    y_indices, x_indices = np.where(instance_mask)
                    if len(x_indices) > 0 and len(y_indices) > 0:
                        x_min, x_max = int(x_indices.min()), int(x_indices.max())
                        y_min, y_max = int(y_indices.min()), int(y_indices.max())
                        mask_bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
                        annotation["bbox"] = mask_bbox
                else:
                    # 如果找不到对应的实例掩码，跳过该对象
                    print(f"警告: 图像 {image_id} 中的对象 {obj_idx+1} 在SegmentationObject中未找到对应掩码")
                    continue
            else:
                # 如果没有SegmentationObject文件，跳过该图像的所有对象
                print(f"警告: 图像 {image_id} 没有对应的SegmentationObject文件，跳过所有对象")
                break
            
            coco_format["annotations"].append(annotation)
            annotation_id += 1
    
    # 保存COCO格式文件
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(coco_format, f, indent=2)
    
    print(f"转换完成! 输出文件: {output_file}")
    print(f"图像数量: {len(coco_format['images'])}")
    print(f"标注数量: {len(coco_format['annotations'])}")
    print(f"类别数量: {len(coco_format['categories'])}")

# 使用示例
if __name__ == "__main__":
    voc_root = "/path/to/your/VOCdevkit/VOC2007"  # 修改为你的数据集路径
    
    # 创建annotations目录
    ann_dir = os.path.join(voc_root, 'annotations')
    os.makedirs(ann_dir, exist_ok=True)
    
    # 转换训练集
    convert_voc_to_coco(
        voc_root=voc_root,
        output_file=os.path.join(ann_dir, 'instances_train.json'),
        split='train'
    )
    
    # 转换验证集
    convert_voc_to_coco(
        voc_root=voc_root,
        output_file=os.path.join(ann_dir, 'instances_val.json'),
        split='val'
    )
    
    print("\n使用提示:")
    print("1. 确保VOC2007数据集结构完整")
    print("2. 检查ImageSets/Segmentation/目录下是否有train.txt和val.txt")
    print("3. 只会使用SegmentationObject中的实例掩码")
    print("4. 如果对象在SegmentationObject中没有对应掩码，该对象将被跳过")
    print("5. 建议先用少量数据测试转换结果")