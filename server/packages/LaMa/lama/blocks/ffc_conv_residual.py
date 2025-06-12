from lama.blocks.ffc import FFC
from torch import Tensor, nn


class FFCConvResidual(nn.Module):
    def __init__(
        self, in_channels: int, global_percent: float, experiment: str
    ):
        super(FFCConvResidual, self).__init__()
        # definition of layers
        self.conv = FFC(in_channels, global_percent, experiment)


def forward(self, x: Tensor):
    x_ffc: Tensor = self.conv(x)
    x_ffc: Tensor = self.conv(x_ffc)
    x = x + x_ffc

    return x
