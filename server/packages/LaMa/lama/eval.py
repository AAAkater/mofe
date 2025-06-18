import os

import lpips
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from lama.data.dataset import LamaDataset
from lama.model import Lama


class Inferencer:
    def __init__(
        self,
        model: nn.Module,
        checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.load_checkpoint(checkpoint_path)
        self.model.eval()  # 设置为评估模式

    def load_checkpoint(self, checkpoint_path: str):
        """加载训练好的模型权重"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})"
        )

    @torch.no_grad()
    def infer(self, model_input: torch.Tensor) -> torch.Tensor:
        """执行推理"""
        model_input = model_input.to(self.device)
        output = self.model(model_input)
        return output.cpu()  # 将结果移回CPU

    @torch.no_grad()
    def infer_batch(self, dataloader: DataLoader) -> list:
        """批量推理"""
        results = []
        for batch in dataloader:
            # 假设dataloader返回的batch结构与训练时相同
            model_input = batch[0].to(self.device)
            outputs = self.model(model_input)
            results.append(outputs.cpu())
        return results


# 使用示例
if __name__ == "__main__":
    # 1. 初始化你的模型 (需要和训练时相同的模型结构)
    # model = YourModelClass()

    # 2. 创建推理器
    # inferencer = Inferencer(
    #     model=model,
    #     checkpoint_path="checkpoints/checkpoint_epoch_X.pth"  # 替换为你的检查点路径
    # )

    # 3. 准备输入数据 (需要和训练时相同的预处理)
    # input_tensor = ... # 你的输入数据，形状应该和训练时一致

    # 4. 执行推理
    # output = inferencer.infer(input_tensor)

    # 5. 处理输出 (例如保存图像、显示结果等)
    # ...
    pass


# 直接定义变量代替命令行参数
data_path = "/root/autodl-tmp/imagenet100"  # 测试数据路径
checkpoint_path = "/root/autodl-tmp/mofe/server/packages/LaMa/checkpoints/3/checkpoint_epoch_50.pth"  # 模型权重路径
out_dir = "./test_images"  # 输出目录
mask_dir = "/root/autodl-tmp/mask/testing_mask_dataset"
experiment = "baseline"  # 实验类型：baseline, conv_change, wgan
batch_size = 64  # 批次大小

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化LPIPS指标
metric_fn = lpips.LPIPS(net="alex").to(device)

# 加载模型
model = Lama(
    in_ch=4,  # RGB + mask
    out_ch=3,  # RGB
    base_ch=64,
    down_n=3,
    up_n=3,
    ffc_n=9,
).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# 数据加载
dataset = LamaDataset(
    data_path,
    mask_dir,
)
test_dataloader = DataLoader(dataset, batch_size=batch_size)

# 创建输出目录
os.makedirs(out_dir, exist_ok=True)

# 计算LPIPS指标
metric_list = []
with tqdm(
    total=len(test_dataloader),
    unit_scale=True,
    postfix={"lpips": 0.0},
    ncols=150,
) as pbar:
    for i, (model_input, gt_image, filenames) in enumerate(test_dataloader):
        model_input = model_input.to(device)
        gt_image = gt_image.to(device)

        with torch.no_grad():
            reconstructed = model(model_input)

        # 保存重建结果
        for j in range(reconstructed.shape[0]):
            save_path = os.path.join(out_dir, os.path.basename(filenames[j]))
            save_image(reconstructed[j], save_path)

        # 计算LPIPS
        current_lpips = torch.mean(metric_fn(gt_image, reconstructed)).item()
        metric_list.append(current_lpips)
        pbar.set_postfix({"lpips": np.mean(metric_list[: i + 1])})
        pbar.update(1)

# 输出统计结果
print("LPIPS mean:", np.mean(metric_list))
print("LPIPS std:", np.std(metric_list))
