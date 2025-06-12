import os
import random
from pathlib import Path

import torchvision.transforms as T
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from lama.data.make_masked_images import create_corrupted_image


def create_dataset(
    original_dir: str,
    mask_dir: str,
    output_dir: str,
    num_samples: int | None = None,
    random_pair: bool = True,
):
    """
    创建用于图像修复训练的数据集
    Args:
        original_dir: 原始图片文件夹路径
        mask_dir: mask图片文件夹路径
        output_dir: 输出文件夹路径
        num_samples: 需要生成的样本数量,None表示处理所有图片
        random_pair: 是否随机配对原图和mask
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图片和mask文件列表
    original_files = [
        f
        for f in os.listdir(original_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    mask_files = [
        f
        for f in os.listdir(mask_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # 如果指定了样本数量，随机选择对应数量的原始图片
    if num_samples is not None:
        num_samples = min(num_samples, len(original_files))
        original_files = random.sample(original_files, num_samples)

    print(
        f"Found {len(original_files)} original images and {len(mask_files)} masks"
    )

    # 处理每张图片
    for idx, orig_file in enumerate(
        tqdm(original_files, desc="Creating dataset")
    ):
        # 获取对应的mask文件
        if random_pair:
            # 随机选择一个mask
            mask_file = random.choice(mask_files)
        else:
            # 使用相同索引的mask（如果存在）
            mask_file = mask_files[idx % len(mask_files)]

        # 构建完整路径
        orig_path = os.path.join(original_dir, orig_file)
        mask_path = os.path.join(mask_dir, mask_file)

        # 构建输出文件名
        output_name = (
            f"{Path(orig_file).stem}_masked_{Path(mask_file).stem}.png"
        )
        output_path = os.path.join(output_dir, output_name)

        try:
            # 创建破损图片
            create_corrupted_image(orig_path, mask_path, output_path)
        except Exception as e:
            print(f"Error processing {orig_file} with {mask_file}: {str(e)}")
            continue


class LamaDataset(Dataset):
    def __init__(
        self,
        original_dir: str,
        mask_dir: str,
        image_size: tuple[int, int] = (256, 256),
        transform: T.Compose | None = None,
    ):
        """
        LaMa数据集
        Args:
            original_dir: 原始图片文件夹路径
            mask_dir: mask图片文件夹路径
            image_size: 调整图片大小 (height, width)
            transform: 自定义的转换操作
        """
        self.original_dir = original_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

        # 获取所有图片和mask文件列表
        self.original_files = [
            f
            for f in os.listdir(original_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.mask_files = [
            f
            for f in os.listdir(mask_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        # 默认转换操作
        if transform is None:
            self.transform = T.Compose(
                [
                    T.Resize(image_size),
                    T.ToTensor(),
                    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.transform = transform

        self.mask_transform = T.Compose([T.Resize(image_size), T.ToTensor()])

    def __len__(self) -> int:
        return len(self.original_files)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        # 获取原始图片路径
        orig_path = os.path.join(self.original_dir, self.original_files[idx])

        # 随机选择一个mask
        mask_file = random.choice(self.mask_files)
        mask_path = os.path.join(self.mask_dir, mask_file)

        # 加载图片
        original_img = Image.open(orig_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # 应用转换
        original_img = self.transform(original_img)
        mask = self.mask_transform(mask)

        # 创建破损图片
        corrupted_img = original_img * mask

        return corrupted_img, mask, original_img


# 在文件末尾添加使用示例
if __name__ == "__main__":
    # 数据集参数
    original_dir = "path/to/original/images"
    mask_dir = "path/to/mask/images"
    batch_size = 8
    num_workers = 4
    output_dir = "path/to/output/dataset"

    # 创建1000张图片的数据集，随机配对原图和mask
    create_dataset(
        original_dir=original_dir,
        mask_dir=mask_dir,
        output_dir=output_dir,
        num_samples=1000,
        random_pair=True,
    )

    # 创建数据集和数据加载器
    dataset = LamaDataset(
        original_dir=original_dir, mask_dir=mask_dir, image_size=(256, 256)
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
        print(f"Corrupted images: {corrupted_imgs.shape}")
        print(f"Masks: {masks.shape}")
        print(f"Original images: {original_imgs.shape}")
        break
