import csv
import os
import random

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations import LongestMaxSize, Normalize, PadIfNeeded, ToTensorV2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from unet.models.unet import UNet
from unet.utils.data_augmentation import RestorationDataset


def add_scratch(img):
    scratch_img = img.copy()
    h, w = scratch_img.shape[:2]
    for _ in range(random.randint(1, 3)):
        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2, y2 = random.randint(0, w), random.randint(0, h)
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        thickness = random.randint(1, 3)
        cv2.line(scratch_img, (x1, y1), (x2, y2), color, thickness)
    return scratch_img


def denormalize(tensor):
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * 0.5 + 0.5) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型
    model = UNet().to(device)
    model.load_state_dict(
        torch.load(
            "models/unet_final.pth", map_location=device, weights_only=True
        )
    )
    model.eval()

    # 定义测试专用的 transform（不含增强）
    test_transform = A.Compose(
        [
            # 保持比例 + 填充，但有黑边
            LongestMaxSize(max_size=512, interpolation=cv2.INTER_AREA),
            PadIfNeeded(
                min_height=512,
                min_width=512,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,  # 黑边填充
            ),
            # A.Resize(512, 512),  # 直接缩放到 512x512
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ],
        additional_targets={"degraded": "image"},
    )

    # 加载测试集
    dataset_path = "data/test"
    test_dataset = RestorationDataset(dataset_path, transform=test_transform)
    print("图像已加载，开始测试...")
    print(f"测试集包含 {len(test_dataset)} 张图像")

    # 创建日志目录
    os.makedirs("logs", exist_ok=True)

    # 初始化指标列表
    all_psnr = []
    all_ssim = []
    all_mae = []
    all_l1 = []  # 新增 L1 指标
    all_names = []

    def normalize(img):
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # [-1, 1]
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        return torch.from_numpy(img).float()

    # ...existing code...
    with torch.no_grad():
        for i in range(len(test_dataset)):
            # 读取原始图像
            img_path = test_dataset.image_files[i]
            original_img = cv2.imread(img_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

            # 让 original_img 也做同样的缩放和填充
            processed = test_transform(
                image=original_img, degraded=original_img
            )
            original_img_resized = denormalize(processed["image"])

            # 获取退化图像和目标图像（来自 dataset 的增强）
            degraded_img, target_img = test_dataset[i]

            # 反归一化，得到 numpy 图像
            degraded_np = denormalize(degraded_img)

            # 在退化图像上手动添加划痕
            scratched_np = add_scratch(degraded_np)

            # 重新归一化为 tensor，用于模型输入
            scratched_tensor = normalize(scratched_np).to(device)

            # 模型推理
            input_tensor = scratched_tensor.unsqueeze(0)
            output_tensor = model(input_tensor)

            # 反归一化修复图像
            restored_np = denormalize(output_tensor.squeeze(0))

            # 计算指标时用 original_img_resized
            current_psnr = psnr(original_img_resized, restored_np)
            current_ssim = ssim(
                original_img_resized,
                restored_np,
                multichannel=True,
                channel_axis=2,
            )
            current_mae = np.mean(
                np.abs(
                    original_img_resized.astype(np.float32)
                    - restored_np.astype(np.float32)
                )
            )
            current_l1 = np.mean(
                np.abs(
                    original_img_resized.astype(np.float32)
                    - restored_np.astype(np.float32)
                )
            )
            # ...existing code...

            # 保存指标
            all_psnr.append(current_psnr)
            all_ssim.append(current_ssim)
            all_mae.append(current_mae)
            all_l1.append(current_l1)
            all_names.append(os.path.basename(img_path))

            # 注释掉保存图像的代码

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            images = [original_img, scratched_np, restored_np]
            titles = [
                "Original Image",
                "Degraded + Scratch",
                f"Restored\nPSNR: {current_psnr:.2f} dB\nSSIM: {current_ssim:.4f}\nMAE: {current_mae:.2f}",
            ]
            for ax, img, title in zip(axes, images, titles):
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(title, fontsize=12)
            plt.tight_layout()
            plt.savefig(f"results/result_{i}.png", bbox_inches="tight")
            plt.close()

    # 保存指标到 CSV 文件
    with open("logs/metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image Name", "PSNR (dB)", "SSIM", "MAE", "L1"])
        for name, p, s, m, l in zip(
            all_names, all_psnr, all_ssim, all_mae, all_l1
        ):
            writer.writerow([name, p, s, m, l])

    # 计算平均指标
    avg_psnr = np.mean(all_psnr)
    avg_ssim = np.mean(all_ssim)
    avg_mae = np.mean(all_mae)
    avg_l1 = np.mean(all_l1)

    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average MAE: {avg_mae:.2f}")
    print(f"Average L1: {avg_l1:.2f}")

    # 绘制指标趋势图
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.plot(all_psnr, label="PSNR")
    plt.axhline(
        avg_psnr, color="r", linestyle="--", label=f"Avg PSNR: {avg_psnr:.2f}"
    )
    plt.title("PSNR Trend Over Images")
    plt.xlabel("Image Index")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.ylim(10, 50)

    plt.subplot(1, 4, 2)
    plt.plot(all_ssim, label="SSIM")
    plt.axhline(
        avg_ssim, color="r", linestyle="--", label=f"Avg SSIM: {avg_ssim:.4f}"
    )
    plt.title("SSIM Trend Over Images")
    plt.xlabel("Image Index")
    plt.ylabel("SSIM")
    plt.legend()
    plt.ylim(-0.1, 1.1)

    plt.subplot(1, 4, 3)
    plt.plot(all_mae, label="MAE")
    plt.axhline(
        avg_mae, color="r", linestyle="--", label=f"Avg MAE: {avg_mae:.2f}"
    )
    plt.title("MAE Trend Over Images")
    plt.xlabel("Image Index")
    plt.ylabel("MAE")
    plt.legend()
    plt.ylim(0, 255)

    plt.subplot(1, 4, 4)
    plt.plot(all_l1, label="L1")
    plt.axhline(
        avg_l1, color="r", linestyle="--", label=f"Avg L1: {avg_l1:.2f}"
    )
    plt.title("L1 Trend Over Images")
    plt.xlabel("Image Index")
    plt.ylabel("L1")
    plt.legend()
    plt.ylim(0, 255)

    plt.tight_layout()
    plt.savefig("logs/metrics_trend_final_epoch2.png")
    plt.close()

    print("Test results saved to logs/")


if __name__ == "__main__":
    test()
