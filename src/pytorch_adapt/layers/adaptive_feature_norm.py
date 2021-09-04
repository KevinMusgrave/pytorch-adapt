import math

import torch

from ..utils import common_functions as c_f


# adapted from https://github.com/jihanyang/AFN/blob/master/vanilla/Office31/SAFN/code/train.py
class AdaptiveFeatureNorm(torch.nn.Module):
    """
    Implementation of the loss in
    [Larger Norm More Transferable:
    An Adaptive Feature Norm Approach for
    Unsupervised Domain Adaptation](https://arxiv.org/abs/1811.07456).
    Encourages features to gradually have larger and larger L2 norms.
    """

    def __init__(self, step_size: float = 1):
        """
        Arguments:
            step_size: The desired increase in L2 norm at each iteration.
                Note that the loss will always be equal to ```step_size```
                because the goal is always to make the L2 norm ```step_size```
                larger than whatever the current L2 norm is.
        """
        super().__init__()
        self.step_size = step_size

    def forward(self, x):
        """"""
        l2_norm = x.norm(p=2, dim=1)
        radius = l2_norm.detach() + self.step_size
        return torch.mean((l2_norm - radius) ** 2)

    def extra_repr(self):
        """"""
        return c_f.extra_repr(self, ["step_size"])


class L2PreservedDropout(torch.nn.Module):
    """
    Implementation of the dropout layer described in
    [Larger Norm More Transferable:
    An Adaptive Feature Norm Approach for
    Unsupervised Domain Adaptation](https://arxiv.org/abs/1811.07456).
    Regular dropout preserves the L1 norm of features, whereas this
    layer preserves the L2 norm.
    """

    def __init__(self, p: float = 0.5, inplace: bool = False):
        """
        Arguments:
            p: probability of an element to be zeroed
            inplace: if set to True, will do this operation in-place
        """
        super().__init__()
        self.dropout = torch.nn.Dropout(p=p, inplace=inplace)
        self.scale = math.sqrt(1 - p)

    def forward(self, x):
        """"""
        x = self.dropout(x)
        if self.training:
            return x * self.scale
        return x
