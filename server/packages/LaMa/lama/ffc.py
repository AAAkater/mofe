import math

import torch
import torch.fft as fft
import torch.nn as nn


class FFTConvNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(FFTConvNet, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2  # 保持输出与输入大小相同的padding

        # 定义频域卷积的参数，使用实部和虚部表示
        self.weight_real = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.weight_imag = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_imag, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_real)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # 将输入数据转换为频域表示
        x_freq = fft.fftn(x, dim=(-2, -1))

        # 获取权重的实部和虚部
        weight_real = self.weight_real.unsqueeze(2).unsqueeze(3)
        weight_imag = self.weight_imag.unsqueeze(2).unsqueeze(3)

        # 计算频域上的卷积
        conv_real = torch.sum(
            x_freq.real.unsqueeze(1) * weight_real
            - x_freq.imag.unsqueeze(1) * weight_imag,
            dim=(-3, -2),
        )
        conv_imag = torch.sum(
            x_freq.real.unsqueeze(1) * weight_imag
            + x_freq.imag.unsqueeze(1) * weight_real,
            dim=(-3, -2),
        )

        # 将频域卷积结果转换回时域
        conv = fft.ifftn(torch.complex(conv_real, conv_imag), dim=(-2, -1))

        # 加上偏置并应用激活函数
        output = conv.real + self.bias.unsqueeze(-1).unsqueeze(-1)
        return torch.relu(output)
