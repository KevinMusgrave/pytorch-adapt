import unittest

import numpy as np
import torch

from pytorch_adapt.layers import MMDLoss

from .. import TEST_DEVICE, TEST_DTYPES


class TestMMDLoss(unittest.TestCase):
    def test_mmd_loss_quadratic(self):
        for dtype in TEST_DTYPES:
            loss_fn = MMDLoss(kernel_scales=1, mmd_type="quadratic")

            x = torch.tensor([[1], [2], [3]], device=TEST_DEVICE, dtype=dtype)
            y = torch.tensor([[2], [3], [4]], device=TEST_DEVICE, dtype=dtype)
            loss = loss_fn(x, y)

            xx_dists = torch.tensor([1, 1, 1, 1, 2, 2]) ** 2
            median_dist = torch.median(xx_dists)
            xx_dists = xx_dists.to(TEST_DEVICE).type(dtype)
            xx_loss = torch.mean(torch.exp(-xx_dists / median_dist))
            yy_loss = xx_loss

            xy_dists = torch.tensor([1, 2, 3, 0, 1, 2, 1, 0, 1]) ** 2
            xy_dists = xy_dists.to(TEST_DEVICE).type(dtype)
            xy_loss = torch.mean(torch.exp(-xy_dists / median_dist))

            correct_loss = xx_loss + yy_loss - 2 * xy_loss
            self.assertTrue(torch.isclose(loss, correct_loss, rtol=1e-2))

    def test_mmd_loss_linear(self):
        for dtype in TEST_DTYPES:
            loss_fn = MMDLoss(kernel_scales=1, mmd_type="linear")
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
                scale = -1.0 / median_dist
                correct_loss += torch.exp(xx_dists[idx, idx2] * scale)
                correct_loss += torch.exp(yy_dists[idx, idx2] * scale)
                correct_loss -= torch.exp(xy_dists[idx, idx2] * scale)
                correct_loss -= torch.exp(xy_dists[idx2, idx] * scale)

            correct_loss /= float(len(x) // 2)
            correct_loss = correct_loss.type(dtype)

            self.assertTrue(torch.isclose(loss, correct_loss, rtol=1e-2))
