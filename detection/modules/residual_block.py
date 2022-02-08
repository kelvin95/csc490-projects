from torch import Tensor, nn


class ResidualBlock(nn.Module):
    """Residual Block.

    References:
        He et al. (2015). Deep Residual Learning for Image Recognition.
            https://arxiv.org/abs/1512.03385
    """

    def __init__(self, channels: int) -> None:
        """Initialization.

        Args:
            channels: Number of input/output channels.
        """
        super(ResidualBlock, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Perform a forward pass of the residual block.

        Args:
            x: A batch_size x C x H x W tensor.

        Returns:
            A batch_size x C x H x W tensor.
        """
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x + residual
        out = self.relu(x)
        return out
