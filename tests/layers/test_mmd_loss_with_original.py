import unittest

import numpy as np
import torch

from pytorch_adapt.layers import MMDBatchedLoss, MMDLoss
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
def DAN(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(
        source,
        target,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma,
    )

    loss1 = 0
    for s1 in range(batch_size):
        for s2 in range(s1 + 1, batch_size):
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss1 += kernels[s1, s2] + kernels[t1, t2]
    loss1 = loss1 / float(batch_size * (batch_size - 1) / 2)

    loss2 = 0
    for s1 in range(batch_size):
        for s2 in range(batch_size):
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss2 -= kernels[s1, t2] + kernels[s2, t1]
    loss2 = loss2 / float(batch_size * batch_size)
    return loss1 + loss2


# modified version of above function
def DAN_diff_size(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    s_size = int(source.size()[0])
    t_size = int(target.size()[0])
    kernels = guassian_kernel(
        source,
        target,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma,
    )

    loss1 = 0
    for s1 in range(s_size):
        for s2 in range(s1 + 1, s_size):
            loss1 += kernels[s1, s2]
    loss1 = loss1 / float(s_size * (s_size - 1) / 2)

    loss2 = 0
    for t1 in range(t_size):
        for t2 in range(t1 + 1, t_size):
            loss2 += kernels[s_size + t1, s_size + t2]
    loss2 = loss2 / float(t_size * (t_size - 1) / 2)

    loss3 = 0
    for s in range(s_size):
        for t in range(t_size):
            loss3 -= kernels[s, s_size + t]
    loss3 = 2 * loss3 / float(s_size * t_size)
    return loss1 + loss2 + loss3


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


def get_bandwidth(s, original_bandwidth):
    bandwidth = original_bandwidth
    if bandwidth is None:
        bandwidth = torch.median(torch.cdist(s, s) ** 2)
    return bandwidth


class TestMMDLossWithOriginal(unittest.TestCase):
    def test_mmd_loss_with_original(self):
        torch.manual_seed(49)

        kernel_num = 5
        half = kernel_num // 2
        kernel_scales = get_kernel_scales(low=-half, high=half, num_kernels=kernel_num)

        for s_size, t_size in [(128, 128), (128, 70), (70, 128)]:
            s = torch.randn(s_size, 32, device=TEST_DEVICE)
            t = torch.randn(t_size, 32, device=TEST_DEVICE) + 0.5
            same_size = s_size == t_size
            for mmd_type in ["linear", "quadratic"]:
                for original_bandwidth in [None, 0.5, 1]:
                    loss_fn = MMDLoss(
                        kernel_scales=kernel_scales,
                        mmd_type=mmd_type,
                        bandwidth=original_bandwidth,
                    )
                    if not same_size and mmd_type == "linear":
                        with self.assertRaises(ValueError):
                            loss = loss_fn(s, t)
                        continue

                    loss = loss_fn(s, t)
                    if same_size:
                        correct_fn = {"linear": DAN_Linear, "quadratic": DAN}[mmd_type]
                    else:
                        correct_fn = DAN_diff_size

                    correct = correct_fn(
                        s,
                        t,
                        kernel_num=kernel_num,
                        fix_sigma=get_bandwidth(s, original_bandwidth),
                    )
                    self.assertTrue(np.isclose(loss.item(), correct.item(), rtol=1e-4))

                    if mmd_type == "quadratic":
                        for batch_size in [2, 10, 31, 32, 128, 199]:
                            loss_fn = MMDBatchedLoss(
                                kernel_scales=kernel_scales,
                                mmd_type=mmd_type,
                                bandwidth=original_bandwidth,
                                batch_size=batch_size,
                            )
                            loss = loss_fn(s, t)

                            rtol = 1e-4 if original_bandwidth is not None else 1e-2
                            self.assertTrue(
                                np.isclose(loss.item(), correct.item(), rtol=rtol)
                            )
