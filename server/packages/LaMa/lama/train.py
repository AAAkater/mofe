import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lama.data.dataset import HfDataset, LamaDataset
from lama.model import Lama


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 2e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 初始化优化器和损失函数
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),  # 保持默认的动量参数
            weight_decay=0.01,  # 添加权重衰减(L2正则化)
            eps=1e-8,  # 更稳定的数值计算
        )

        self.criterion = nn.L1Loss()

        # 记录训练和验证损失
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def train_epoch(self) -> float:
        """训练一个epoch并返回平均损失"""
        self.model.train()
        total_loss = 0

        for model_input, original_img_tensor in tqdm(
            self.train_loader, desc="Training"
        ):
            # 将数据移到设备上
            model_input = model_input.to(self.device)
            original_img_tensor = original_img_tensor.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(model_input)

            # 计算损失
            loss = self.criterion(outputs, original_img_tensor)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def val_epoch(self) -> float:
        """验证一个epoch并返回平均损失"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for model_input, original_img_tensor in tqdm(
                self.val_loader, desc="Validation"
            ):
                # 将数据移到设备上
                model_input = model_input.to(self.device)
                original_img_tensor = original_img_tensor.to(self.device)

                # 前向传播
                outputs = self.model(model_input)

                # 计算损失
                loss = self.criterion(outputs, original_img_tensor)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def save_checkpoint(self, epoch: int):
        """保存模型检查点和损失数据"""
        # 保存PyTorch模型检查点
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, self.save_dir / f"checkpoint_epoch_{epoch}.pth")

        # 保存损失数据到JSON文件
        loss_data = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "epoch": epoch,
        }
        with open(self.save_dir / f"losses_epoch_{epoch}.json", "w") as f:
            json.dump(loss_data, f, indent=4)

    def plot_losses(self):
        """绘制损失曲线"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label="Training Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Time")
        plt.legend()
        plt.savefig(self.save_dir / "loss_plot.png")
        plt.close()


def main():
    # 训练参数
    num_epochs = 50
    batch_size = 32
    learning_rate = 2e-4
    save_interval = 10

    original_dir = "/root/code/datasets/imagenet-100"
    mask_dir = "/root/code/datasets/mask/testing_mask_dataset"
    save_dir = "./checkpoints/1"

    # 创建数据集和数据加载器
    train_dataset = HfDataset(original_dir, mask_dir, split="train")
    val_dataset = HfDataset(original_dir, mask_dir, split="validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
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
        val_loader=val_loader,
        learning_rate=learning_rate,
        save_dir=save_dir,
    )

    # 训练循环
    print(f"Starting training on {trainer.device}")
    print("Press Ctrl+C to interrupt training and save checkpoint")

    try:
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # 训练一个epoch
            train_loss = trainer.train_epoch()
            trainer.train_losses.append(train_loss)

            # 验证一个epoch
            val_loss = trainer.val_epoch()
            trainer.val_losses.append(val_loss)

            print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

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
