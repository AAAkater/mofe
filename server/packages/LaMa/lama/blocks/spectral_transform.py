import torch
from torch import Tensor, nn


class SpectralTransform(nn.Module):
    """频谱变换模块

    将输入特征图进行傅里叶变换,在频域进行特征处理后再逆变换回空域

    Args:
        channels_in (int): 输入通道数
        channels_out (int): 输出通道数
        channels_hidden (int): 隐藏层通道数
        kernel_size (int): 卷积核大小
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels_hidden: int,
        kernel_size: int,
    ):
        super().__init__()
        self.channels_hidden = channels_hidden

        # 空域特征提取
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                channels_hidden,
                kernel_size,
                padding="same",
            ),
            nn.BatchNorm2d(channels_hidden),
            nn.ReLU(inplace=True),
        )

        # 频域特征处理 (实部和虚部concatenate后的维度要乘2)
        self.conv_spectral = nn.Sequential(
            nn.Conv2d(2 * channels_hidden, 2 * channels_hidden, 1),
            nn.BatchNorm2d(2 * channels_hidden),
            nn.ReLU(inplace=True),
        )

        # 输出投影层
        self.proj = nn.Conv2d(channels_hidden, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """前向传播

        Args:
            x (Tensor): 输入特征图 [B, C, H, W]

        Returns:
            Tensor: 输出特征图 [B, C, H, W]
        """
        # 1. 空域特征提取
        x_spatial: Tensor = self.conv_spatial(x)

        # 2. FFT变换到频域
        x_freq = torch.fft.rfft2(x_spatial, norm="ortho")
        x_freq = torch.concat((x_freq.real, x_freq.imag), dim=1)

        # 3. 频域特征处理
        x_freq = self.conv_spectral(x_freq)

        # 4. IFFT变换回空域
        x_out = torch.complex(
            x_freq[:, : self.channels_hidden], x_freq[:, self.channels_hidden :]
        )
        x_out = torch.fft.irfft2(x_out, norm="ortho")

        # 5. 残差连接
        x_out = x_out + x_spatial

        # 6. 输出投影
        return self.proj(x_out)
