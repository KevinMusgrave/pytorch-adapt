import unittest

import torch

from pytorch_adapt.layers import MMDLoss
from pytorch_adapt.layers import utils as l_u

from .. import TEST_DEVICE, TEST_DTYPES


class TestMKMMDLoss(unittest.TestCase):
    def test_mkmmd_loss_quadratic(self):
        kernel_scales = l_u.get_kernel_scales(low=-2, high=2, num_kernels=5, base=2)
        for dtype in TEST_DTYPES:
            loss_fn = MMDLoss(kernel_scales=kernel_scales, mmd_type="quadratic")
            x = torch.tensor([[1], [2], [3]], device=TEST_DEVICE, dtype=dtype)
            y = torch.tensor([[2], [3], [4]], device=TEST_DEVICE, dtype=dtype)
            loss = loss_fn(x, y)

            xx_dists = torch.tensor([1, 1, 1, 1, 2, 2]) ** 2
            median_dist = torch.median(xx_dists)
            xx_dists = xx_dists.to(TEST_DEVICE).type(dtype)
            xy_dists = torch.tensor([1, 2, 3, 0, 1, 2, 1, 0, 1]) ** 2
            xy_dists = xy_dists.to(TEST_DEVICE).type(dtype)

            xx_kernels = torch.zeros_like(xx_dists)
            xy_kernels = torch.zeros_like(xy_dists)

            scales = [0.25, 0.5, 1, 2, 4]
            for s in scales:
                scale = -s / median_dist
                xx_kernels += torch.exp(xx_dists * scale)
                xy_kernels += torch.exp(xy_dists * scale)

            xx_loss = torch.mean(xx_kernels / len(scales))
            yy_loss = xx_loss
            xy_loss = torch.mean(xy_kernels / len(scales))
            correct_loss = xx_loss + yy_loss - 2 * xy_loss

            self.assertTrue(torch.isclose(loss, correct_loss, rtol=1e-2))

    def test_mkmmd_loss_linear(self):
        kernel_scales = l_u.get_kernel_scales(low=-2, high=2, num_kernels=5, base=2)
        for dtype in TEST_DTYPES:
            loss_fn = MMDLoss(kernel_scales=kernel_scales, mmd_type="linear")
            x = torch.tensor([[1], [2], [3], [4]], device=TEST_DEVICE, dtype=dtype)
            y = torch.tensor([[2], [3], [4], [5]], device=TEST_DEVICE, dtype=dtype)
            loss = loss_fn(x, y)

            xx_dists = (
                torch.tensor([0, 1, 2, 3, 1, 0, 1, 2, 2, 1, 0, 1, 3, 2, 1, 0]) ** 2
            )
            xx_dists = xx_dists.view(4, 4)
            median_dist = torch.median(xx_dists)
            xx_dists = xx_dists.to(TEST_DEVICE).type(dtype)
            yy_dists = xx_dists
            xy_dists = (
                torch.tensor([1, 2, 3, 4, 0, 1, 2, 3, 1, 0, 1, 2, 2, 1, 0, 1]) ** 2
            )
            xy_dists = xy_dists.to(TEST_DEVICE).type(dtype)
            xy_dists = xy_dists.view(4, 4)

            correct_loss = 0
            for idx in range(len(x) // 2):
                idx2 = idx + 1
                xx_kernels = 0
                yy_kernels = 0
                xy_kernels1 = 0
                xy_kernels2 = 0

                scales = [0.25, 0.5, 1, 2, 4]
                for s in scales:
                    scale = -s / median_dist
                    xx_kernels += torch.exp(xx_dists[idx, idx2] * scale)
                    yy_kernels += torch.exp(yy_dists[idx, idx2] * scale)
                    xy_kernels1 += torch.exp(xy_dists[idx, idx2] * scale)
                    xy_kernels2 += torch.exp(xy_dists[idx2, idx] * scale)

                correct_loss += (
                    xx_kernels + yy_kernels - xy_kernels1 - xy_kernels2
                ) / len(scales)

            correct_loss /= float(len(x) // 2)
            correct_loss = correct_loss.type(dtype)

            self.assertTrue(torch.isclose(loss, correct_loss, rtol=1e-2))
