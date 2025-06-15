import os
from glob import glob

import cv2
import numpy as np


def create_corrupted_image(
    original_path, mask_path, output_path="corrupted.png"
):
    """
    创建破损图片
    Args:
        original_path: 原始图片路径
        mask_path: 掩码图片路径（白色区域保留，黑色区域擦除）
        output_path: 输出图片路径，默认为'corrupted.png'
    """
    # 加载数据
    original_img = cv2.imread(original_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)  # 转为RGB
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 单通道掩码

    resized_mask = cv2.resize(
        mask,
        (original_img.shape[1], original_img.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    # 二值化（可选，确保严格0/255）
    _, binary_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)
    # 调整掩码尺寸

    # 合成破损图片 - 直接使用mask_float，不再使用(1 - mask_float)
    mask_float = 1 - (binary_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
    corrupted_img = original_img * mask_float  # 修改这里，直接使用mask_float

    # 保存结果
    cv2.imwrite(
        output_path,
        cv2.cvtColor(corrupted_img.astype(np.uint8), cv2.COLOR_RGB2BGR),
    )

    return corrupted_img


def process_dataset(
    original_root: str, mask_root: str, output_root: str, max_images=10000
):
    """
    处理整个数据集
    Args:
        original_root: 原始图片根目录 (./imagenet100)
        mask_root: mask图片目录 (./mask/testing_mask_dataset)
        output_root: 输出目录
    """
    # 获取所有原始图片路径
    original_images = []
    for root, dirs, files in os.walk(original_root):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                original_images.append(os.path.join(root, file))
        if len(original_images) >= max_images:
            break

    # 获取所有mask图片路径
    mask_images = sorted(glob(os.path.join(mask_root, "*.png")))
    if not mask_images:
        raise ValueError(f"No mask images found in {mask_root}")

    # 确保输出目录存在
    os.makedirs(output_root, exist_ok=True)

    # 处理每张原始图片
    for i, original_path in enumerate(original_images):
        # 循环使用mask图片
        mask_path = mask_images[i % len(mask_images)]

        # 创建相对路径结构
        rel_path = os.path.relpath(original_path, original_root)
        output_path = os.path.join(output_root, rel_path)

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 创建破损图片
        create_corrupted_image(original_path, mask_path, output_path)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(original_images)} images")


if __name__ == "__main__":
    # 配置路径
    original_root = "/root/autodl-tmp/imagenet100"
    mask_root = "/root/autodl-tmp/mask/testing_mask_dataset"
    output_root = "/root/autodl-tmp/masked_images"

    # 处理数据集
    print("Starting dataset processing...")
    process_dataset(original_root, mask_root, output_root)
    print("Dataset processing completed!")
