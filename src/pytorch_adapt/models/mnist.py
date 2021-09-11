import torch
import torch.nn as nn
import torch.nn.functional as F


# from https://arxiv.org/pdf/1409.7495.pdf
class MNISTFeatures(nn.Module):
    """
    A small convnet for extracting features
    from MNIST.
    """

    def __init__(self):
        """ """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 48, 5, 1)
        self.fc = nn.Identity()

    def forward(self, x):
        """ """
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)
