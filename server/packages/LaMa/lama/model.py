from torch import Tensor, nn

from lama.blocks.ffc_conv_residual import FFCConvResidual


class Lama(nn.Module):
    """LaMa (Large Mask) 图像修复模型

    参数:
        exp (str): 实验配置名称
        in_ch (int): 输入通道数,默认4(RGB+mask)
        out_ch (int): 输出通道数,默认3(RGB)
        down_n (int): 下采样次数,默认3
        up_n (int): 上采样次数,默认3
        base_ch (int): 基础通道数,默认64
        ffc_n (int): FFC Res数量,默认9
        global_ratio (float): FFC Res中全局分支的比例,默认0.6
    """

    def __init__(
        self,
        exp: str = "baseline",
        in_ch: int = 4,
        out_ch: int = 3,
        down_n: int = 3,
        up_n: int = 3,
        base_ch: int = 64,
        ffc_n: int = 9,
        global_ratio: float = 0.6,
    ) -> None:
        super().__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, base_ch, kernel_size=7),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            *[
                nn.Sequential(
                    nn.Conv2d(
                        base_ch * (2**i),
                        base_ch * (2 ** (i + 1)),
                        3,
                        stride=2,
                        padding=1,
                        bias=True,
                    ),
                    nn.BatchNorm2d(base_ch * (2 ** (i + 1))),
                    nn.ReLU(inplace=True),
                )
                for i in range(down_n)
            ],
        )

        # 瓶颈层
        max_ch = base_ch * (2**down_n)
        self.bottleneck = nn.Sequential(
            *[FFCConvResidual(max_ch, global_ratio, exp) for _ in range(ffc_n)]
        )

        # 解码器
        self.decoder = nn.Sequential(
            *[
                nn.Sequential(
                    nn.ConvTranspose2d(
                        base_ch * (2 ** (up_n - i)),
                        base_ch * (2 ** (up_n - i - 1)),
                        3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(base_ch * (2 ** (up_n - i - 1))),
                    nn.ReLU(inplace=True),
                )
                for i in range(up_n)
            ],
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_ch, out_ch, kernel_size=7, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """前向传播

        参数:
            x (Tensor): 输入张量 [B, in_ch, H, W]

        返回:
            Tensor: 输出张量 [B, out_ch, H, W]
        """
        feat: Tensor = self.encoder(x)  # 编码
        feat = self.bottleneck(feat)  # 特征处理
        out: Tensor = self.decoder(feat)  # 解码
        return out
