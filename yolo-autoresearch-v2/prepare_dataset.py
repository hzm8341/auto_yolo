"""
NEU-DET 数据集准备脚本

将 Pascal VOC XML 格式的标注转换为 YOLO txt 格式。
此文件锁死，不允许 Agent 修改。
"""
import os
import xml.etree.ElementTree as ET
import shutil
import random
import glob
from pathlib import Path


# NEU-DET 类别映射（顺序固定，对应 crazing=0）
CLASS_MAP = {
    'crazing': 0,
    'inclusion': 1,
    'patches': 2,
    'pitted_surface': 3,
    'rolled-in_scale': 4,
    'scratches': 5
}


def convert_voc_to_yolo(xml_path: str, img_w: int, img_h: int) -> list:
    """将 Pascal VOC XML 转换为 YOLO txt 格式

    Args:
        xml_path: XML 文件路径
        img_w: 图像宽度
        img_h: 图像高度

    Returns:
        list of YOLO format strings
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []

    for obj in root.findall('object'):
        cls_name = obj.find('name').text
        if cls_name not in CLASS_MAP:
            continue
        cls_id = CLASS_MAP[cls_name]
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('bndbox').find('ymax').text) if bbox.find('ymax') is None else float(bbox.find('ymax').text)

        # 修复：如果 ymax 不存在，尝试其他方式获取
        if bbox.find('ymax') is None:
            ymax = float(bbox.find('y_max').text)

        # 转换为 YOLO 归一化格式 (中心x, 中心y, 宽, 高)
        cx = (xmin + xmax) / 2 / img_w
        cy = (ymin + ymax) / 2 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines


def prepare(src_dir: str, out_dir: str, val_ratio: float = 0.2):
    """处理数据集，输出 train/val/ 目录

    Args:
        src_dir: 源数据目录（包含 images 和 annotations）
        out_dir: 输出目录
        val_ratio: 验证集比例 (默认 0.2 = 20%)
    """
    random.seed(42)

    # 查找所有图像
    all_imgs = sorted(glob.glob(f"{src_dir}/**/*.jpg", recursive=True))
    if not all_imgs:
        # 尝试其他格式
        all_imgs = sorted(glob.glob(f"{src_dir}/**/*.png", recursive=True))

    print(f"找到 {len(all_imgs)} 张图像")

    random.shuffle(all_imgs)
    n_val = int(len(all_imgs) * val_ratio)
    splits = {'val': all_imgs[:n_val], 'train': all_imgs[n_val:]}

    for split, imgs in splits.items():
        img_out = Path(out_dir) / split / 'images'
        lbl_out = Path(out_dir) / split / 'labels'
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in imgs:
            xml_path = img_path.replace('.jpg', '.xml').replace('.png', '.xml')

            # 复制图像
            shutil.copy(img_path, img_out / Path(img_path).name)

            # 转换标注
            if os.path.exists(xml_path):
                lines = convert_voc_to_yolo(xml_path, 200, 200)
                lbl_file = lbl_out / Path(img_path).stem
                lbl_file.with_suffix('.txt').write_text('\n'.join(lines))

    print(f"完成：train {len(splits['train'])} 张，val {len(splits['val'])} 张")


if __name__ == '__main__':
    # 默认处理 NEU-DET 数据集
    prepare('NEU-DET', 'data')
