import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class LamaDataset(Dataset):
    def __init__(
        self,
        original_dir: str,
        mask_dir: str,
        image_size: tuple[int, int] = (256, 256),
    ):
        """
        参数说明:
            original_dir: 存放原始图片的目录路径
            mask_dir: 存放掩码图片的目录路径（二值化掩码，1表示需要修复的区域）
            image_size: 图片调整尺寸 (高度, 宽度)
            transform: 自定义的数据增强变换组合
        """
        self.original_dir = original_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

        # 获取所有原始图片
        self.original_files = [
            f
            for f in os.listdir(original_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not self.original_files:
            raise ValueError("错误: 在原始图片目录中未找到有效的图片文件")

        # 获取所有掩码图片
        self.mask_files = [
            f
            for f in os.listdir(mask_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not self.mask_files:
            raise ValueError("错误: 在掩码目录中未找到有效的图片文件")

        # 原始图像转换流程
        self.original_transform = T.Compose(
            [
                T.Resize(image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # 掩码图像转换流程（保持单通道）
        self.mask_transform = T.Compose(
            [
                T.Resize(image_size),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        """返回数据集样本数量"""
        return len(self.original_files)

    def __getitem__(self, idx: int):
        """获取单个样本数据"""
        original_filename = self.original_files[idx]
        # 循环使用掩码图片
        mask_filename = self.mask_files[idx % len(self.mask_files)]

        # 加载原始图像（RGB三通道）
        original_path = os.path.join(self.original_dir, original_filename)
        original_img = Image.open(original_path).convert("RGB")

        # 加载掩码图像（灰度单通道）
        mask_path = os.path.join(self.mask_dir, mask_filename)
        mask = Image.open(mask_path).convert("L")

        # 应用不同的数据变换
        original_img = self.original_transform(original_img)
        mask = self.mask_transform(mask)

        # 将掩码二值化（>0.5的值设为1，其余为0）
        mask = (mask > 0.5).float()

        # 生成破损图像：原始图像 * (1 - 掩码)
        # 注意：这里需要确保mask的形状与original_img匹配
        # 通过unsqueeze(0)和expand_as来广播mask的形状
        # mask = mask.expand_as(original_img)
        corrupted_img = original_img * (1 - mask)

        # 返回模型训练所需的三要素：
        # 1. 带缺失区域的破损图像
        # 2. 缺失区域标识掩码（1=缺失，0=保留）
        # 3. 原始完整图像（用于计算损失）
        return corrupted_img, mask, original_img


# 使用示例
if __name__ == "__main__":
    # 数据集参数
    original_dir = "/root/autodl-tmp/imagenet100/n01729322"
    mask_dir = "/root/autodl-tmp/mask/testing_mask_dataset"
    batch_size = 128
    num_workers = 4

    # 创建数据集和数据加载器
    dataset = LamaDataset(
        original_dir=original_dir,
        mask_dir=mask_dir,
        image_size=(256, 256),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    # 使用示例
    for corrupted_imgs, masks, original_imgs in dataloader:
        print("Batch shapes:")
        print(f"{masks.shape=}")
        print(f"Corrupted images: {corrupted_imgs.shape}")
        print(f"Original images: {original_imgs.shape}")
        break
