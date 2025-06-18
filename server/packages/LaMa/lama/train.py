from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lama.data.lama_dataset import LamaDataset
from lama.model import Lama


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        learning_rate: float = 2e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 初始化优化器和损失函数
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.L1Loss()

        # 记录训练损失
        self.losses = []

    def train_epoch(self) -> float:
        """训练一个epoch并返回平均损失"""
        self.model.train()
        total_loss = 0

        for corrupted_imgs, masks, original_imgs in tqdm(
            self.train_loader, desc="Training"
        ):
            # 将数据移到设备上
            corrupted_imgs = corrupted_imgs.to(self.device)
            masks = masks.to(self.device)
            original_imgs = original_imgs.to(self.device)

            # 将corrupted_imgs和masks拼接作为输入
            model_input = torch.cat([corrupted_imgs, masks], dim=1)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(model_input)

            # 计算损失
            loss = self.criterion(outputs, original_imgs)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def save_checkpoint(self, epoch: int):
        """保存模型检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": self.losses,
        }
        torch.save(checkpoint, self.save_dir / f"checkpoint_epoch_{epoch}.pth")

    def plot_losses(self):
        """绘制损失曲线"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.legend()
        plt.savefig(self.save_dir / "loss_plot.png")
        plt.close()


def main():
    # 训练参数
    num_epochs = 100
    batch_size = 64
    learning_rate = 2e-4
    save_interval = 10

    original_dir = "/root/autodl-tmp/imagenet100/n01729322"
    mask_dir = "/root/autodl-tmp/mask/testing_mask_dataset"

    # 创建数据集和数据加载器
    dataset = LamaDataset(
        original_dir=original_dir,
        mask_dir=mask_dir,
        image_size=(256, 256),
    )
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # 初始化模型
    model = Lama(
        in_ch=4,  # RGB + mask
        out_ch=3,  # RGB
        base_ch=64,
        down_n=3,
        up_n=3,
        ffc_n=9,
    )

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        learning_rate=learning_rate,
        save_dir="checkpoints",
    )

    # 训练循环
    print(f"Starting training on {trainer.device}")
    print("Press Ctrl+C to interrupt training and save checkpoint")

    try:
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # 训练一个epoch
            avg_loss = trainer.train_epoch()
            trainer.losses.append(avg_loss)

            print(f"Average loss: {avg_loss:.4f}")

            # 每save_interval个epoch保存一次模型
            if (epoch + 1) % save_interval == 0:
                trainer.save_checkpoint(epoch + 1)
                trainer.plot_losses()

        # 训练结束后保存最终模型和损失曲线
        trainer.save_checkpoint(num_epochs)
        trainer.plot_losses()
        print("Training completed!")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    finally:
        print("Training completed!")


if __name__ == "__main__":
    main()
