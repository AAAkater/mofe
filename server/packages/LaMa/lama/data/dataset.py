import os

import matplotlib.pyplot as plt
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
        max_categories: int = 1,
    ):
        """
        参数说明:
            original_dir: 存放原始图片的目录路径
            mask_dir: 存放掩码图片的目录路径(二值化掩码,1表示需要修复的区域)
            image_size: 图片调整尺寸 (高度, 宽度)
            transform: 自定义的数据增强变换组合
        """
        self.original_dir = original_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

        # 获取所有分类子文件夹
        category_dirs = [
            d
            for d in os.listdir(original_dir)
            if os.path.isdir(os.path.join(original_dir, d))
        ]

        if not category_dirs:
            raise ValueError("错误: 在原始图片目录中未找到任何分类子文件夹")

        category_dirs = category_dirs[:max_categories]

        # 获取所有原始图片路径 (保留分类子文件夹结构)
        self.original_files: list[str] = []
        for category in category_dirs:
            category_path = os.path.join(original_dir, category)
            files = [
                os.path.join(category, f)  # 保留相对路径
                for f in os.listdir(category_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            self.original_files.extend(files)

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
        original_filename = self.original_files[idx]
        mask_filename = self.mask_files[idx % len(self.mask_files)]

        # 加载原始图像（RGB三通道）
        original_path = os.path.join(self.original_dir, original_filename)
        original_img = Image.open(original_path).convert("RGB")

        # 加载掩码图像（灰度单通道）
        mask_path = os.path.join(self.mask_dir, mask_filename)
        mask = Image.open(mask_path).convert("L")

        # 应用变换
        original_img_tensor: Tensor = self.original_transform(original_img)  # type: ignore
        mask_tensor: Tensor = (self.mask_transform(mask) > 0.5).float()  # type: ignore

        # 生成破损图像
        corrupted_img_tensor: Tensor = original_img_tensor * (1 - mask_tensor)

        # 拼接为4通道输入 (3通道图像 + 1通道mask)
        model_input = torch.cat([corrupted_img_tensor, mask_tensor], dim=0)

        return model_input, original_img_tensor, original_filename

    def show_dataset(self, batch_size: int = 64, num_workers: int = 4):
        dataloader = DataLoader(
            self,
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


class HfDataset(Dataset):
    pass


def tensor_to_image(tensor: Tensor) -> Image.Image:
    """将 tensor 转换为 PIL 图像，支持单通道（掩码）和 3 通道（RGB）"""
    tensor = tensor.cpu()  # 确保 tensor 在 CPU 上
    if tensor.dim() == 3 and tensor.shape[0] == 1:  # 单通道掩码 (1, H, W)
        tensor = tensor.squeeze(0)  # (1, H, W) → (H, W)
        tensor = (tensor * 255).byte()  # [0,1] → [0,255] (uint8)
        return Image.fromarray(tensor.numpy(), mode="L")  # 'L' 表示灰度图
    else:  # 3 通道 RGB 图像 (C, H, W)
        tensor = tensor * 0.5 + 0.5  # 反归一化 [-1,1] → [0,1]
        tensor = tensor.clamp(0, 1)  # 确保值在 [0,1] 范围内
        tensor = (tensor * 255).byte()  # [0,1] → [0,255] (uint8)
        npy = tensor.permute(1, 2, 0).numpy()  # (C, H, W) → (H, W, C)
        return Image.fromarray(npy)


def show_images():
    # 获取数据集
    original_dir = "/root/autodl-tmp/imagenet100/n01729322"
    mask_dir = "/root/autodl-tmp/mask/testing_mask_dataset"
    dataset = LamaDataset(original_dir=original_dir, mask_dir=mask_dir)

    # 获取 3 个不同的样本（索引 0, 1, 2）
    samples = [dataset[i] for i in range(3)]

    # 创建 matplotlib 画布（3 行 3 列）
    plt.figure(figsize=(12, 9))

    for i, (corrupted, mask, original) in enumerate(samples):
        # 转换为 PIL 图像
        corrupted_img = tensor_to_image(corrupted)
        original_img = tensor_to_image(original)
        mask_img = tensor_to_image(mask)

        # 显示原始图像（第 1 列）
        plt.subplot(3, 3, i * 3 + 1)
        plt.title(f"Original {i + 1}")
        plt.imshow(original_img)
        plt.axis("off")

        # 显示掩码（第 2 列）
        plt.subplot(3, 3, i * 3 + 2)
        plt.title(f"Mask {i + 1}")
        plt.imshow(mask_img, cmap="gray")
        plt.axis("off")

        # 显示破损图像（第 3 列）
        plt.subplot(3, 3, i * 3 + 3)
        plt.title(f"Corrupted {i + 1}")
        plt.imshow(corrupted_img)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("three_samples.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()


# 使用示例
if __name__ == "__main__":
    # 数据集参数
    # show_dataset()
    show_images()
