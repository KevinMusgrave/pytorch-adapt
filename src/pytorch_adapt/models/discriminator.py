import torch.nn as nn


class Discriminator(nn.Module):
    """
    A 3-layer MLP for domain classification.
    """

    def __init__(self, in_size=2048, h=2048, out_size=1):
        """
        Arguments:
            in_size: size of the input
            h: hidden layer size
            out_size: size of the output
        """

        super().__init__()
        self.h = h
        self.net = nn.Sequential(
            nn.Linear(in_size, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, out_size),
        )
        self.out_size = out_size

    def forward(self, x):
        """"""
        return self.net(x).squeeze(1)
