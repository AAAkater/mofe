import torch
from lama.blocks.spectral_transform import SpectralTransform
from torch import Tensor, nn


class FFC(nn.Module):
    """快速傅里叶卷积模块

    将输入特征分为局部分支和全局分支,分别进行处理后再合并
    局部分支使用普通卷积,全局分支使用频谱变换

    Args:
        in_channels (int): 输入通道数
        global_percent (float): 全局分支的通道数比例 (0-1)
        experiment (str): 实验模式,若为"conv_change"则局部到全局使用频谱变换
    """

    def __init__(
        self, in_channels: int, global_percent: float, experiment: str = ""
    ):
        super().__init__()

        # 计算局部和全局通道数
        self.channels_global = round(in_channels * global_percent)
        self.channels_local = in_channels - self.channels_global

        # 局部到局部的卷积
        self.conv_ll = nn.Conv2d(
            self.channels_local,
            self.channels_local,
            kernel_size=3,
            padding="same",
        )

        # 局部到全局的变换
        if experiment == "conv_change":
            self.conv_lg = SpectralTransform(
                self.channels_local,
                self.channels_global,
                self.channels_global // 2,
                1,
            )
        else:
            self.conv_lg = nn.Conv2d(
                self.channels_local,
                self.channels_global,
                kernel_size=3,
                padding="same",
            )

        # 全局到局部的变换
        self.conv_gl = nn.Conv2d(
            self.channels_global,
            self.channels_local,
            kernel_size=3,
            padding="same",
        )

        # 全局到全局的频谱变换
        self.conv_gg = SpectralTransform(
            self.channels_global,
            self.channels_global,
            self.channels_global // 2,
            1,
        )

        # 归一化和激活函数
        self.norm_act = nn.Sequential(
            nn.BatchNorm2d(self.channels_local), nn.ReLU(inplace=True)
        )
        self.norm_act_global = nn.Sequential(
            nn.BatchNorm2d(self.channels_global), nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        """前向传播

        Args:
            x (Tensor): 输入特征图 [B, C, H, W]

        Returns:
            Tensor: 输出特征图 [B, C, H, W]
        """
        # 1. 分离局部和全局特征
        x_local = x[:, : self.channels_local]  # 局部分支
        x_global = x[:, self.channels_local :]  # 全局分支

        # 2. 四路特征变换
        out_local = self.conv_ll(x_local) + self.conv_gl(x_global)  # 局部输出
        out_global = self.conv_gg(x_global) + self.conv_lg(x_local)  # 全局输出

        # 3. 归一化和激活
        out_local = self.norm_act(out_local)
        out_global = self.norm_act_global(out_global)

        # 4. 合并局部和全局特征
        return torch.concat((out_local, out_global), dim=1)
