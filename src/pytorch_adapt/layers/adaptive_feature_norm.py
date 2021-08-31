import math

import torch

from ..utils import common_functions as c_f


# adapted from https://github.com/jihanyang/AFN/blob/master/vanilla/Office31/SAFN/code/train.py
class AdaptiveFeatureNorm(torch.nn.Module):
    def __init__(self, step_size=1):
        super().__init__()
        self.step_size = step_size

    def forward(self, x):
        l2_norm = x.norm(p=2, dim=1)
        radius = l2_norm.detach() + self.step_size
        return torch.mean((l2_norm - radius) ** 2)

    def extra_repr(self):
        return c_f.extra_repr(self, ["step_size"])


class L2PreservedDropout(torch.nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=p, inplace=inplace)
        self.scale = math.sqrt(1 - p)

    def forward(self, x):
        x = self.dropout(x)
        if self.training:
            return x * self.scale
        return x
