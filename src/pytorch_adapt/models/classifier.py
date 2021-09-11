import torch.nn as nn


class Classifier(nn.Module):
    """
    A 3-layer MLP for clasification.
    """

    def __init__(self, num_classes, in_size=2048, h=1024):
        """
        Arguments:
            num_classes: size of the output
            in_size: size of the input
            h: hidden layer size
        """
        super().__init__()
        self.h = h
        self.net = nn.Sequential(
            nn.Linear(in_size, h),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(h, h // 2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(h // 2, num_classes),
        )

    def forward(self, x):
        """"""
        return self.net(x)
