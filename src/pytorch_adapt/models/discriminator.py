import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_size=2048, h=2048, out_size=1):
        super().__init__()
        self.h = h
        self.net = nn.Sequential(
            nn.Linear(in_size, h),
            nn.ReLU(),
            nn.Linear(h, h // 2),
            nn.ReLU(),
            nn.Linear(h // 2, out_size),
        )
        self.out_size = out_size

    def forward(self, x):
        return self.net(x).squeeze(1)
