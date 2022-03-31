import unittest

import numpy as np
import torch

from pytorch_adapt.layers import MMDLoss
from pytorch_adapt.layers.utils import get_kernel_scales

from .. import TEST_DEVICE


# from https://github.com/thuml/Xlearn/blob/master/pytorch/src/loss.py
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1))
    )
    total1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1))
    )
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [
        torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list
    ]
    return sum(kernel_val) / len(kernel_val)


# from https://github.com/thuml/Xlearn/blob/master/pytorch/src/loss.py
def DAN_Linear(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(
        source,
        target,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma,
    )

    # Linear version
    loss = 0

    # original
    # for i in range(batch_size):
    #     s1, s2 = i, (i+1)%batch_size
    #     t1, t2 = s1+batch_size, s2+batch_size
    #     loss += kernels[s1, s2] + kernels[t1, t2]
    #     loss -= kernels[s1, t2] + kernels[s2, t1]

    # return loss / float(batch_size)

    # according to https://arxiv.org/pdf/1502.02791.pdf and https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf
    for i in range(0, batch_size // 2):
        s1, s2 = (2 * i), (2 * i) + 1
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]

    return 2 * loss / float(batch_size)


class TestMMDLossWithOriginal(unittest.TestCase):
    def test_mmd_loss_with_original(self):
        s = torch.randn(128, 32, device=TEST_DEVICE)
        t = torch.randn(128, 32, device=TEST_DEVICE) + 0.5

        kernel_num = 5
        half = kernel_num // 2
        kernel_scales = get_kernel_scales(low=-half, high=half, num_kernels=kernel_num)

        for bandwidth in [None, 0.5, 1]:
            loss_fn = MMDLoss(kernel_scales=kernel_scales, bandwidth=bandwidth)
            loss = loss_fn(s, t)

            if bandwidth is None:
                bandwidth = torch.median(torch.cdist(s, s) ** 2)
            correct = DAN_Linear(s, t, kernel_num=kernel_num, fix_sigma=bandwidth)

            self.assertTrue(np.isclose(loss.item(), correct.item()))
