import unittest

import numpy as np
import torch

from pytorch_adapt.layers import CORALLoss

from .. import TEST_DEVICE, TEST_DTYPES


class TestCORALLoss(unittest.TestCase):
    def test_coral_loss(self):
        for dtype in TEST_DTYPES:
            loss_fn = CORALLoss()
            batch_size = 32
            embedding_size = 128
            x = torch.randn(batch_size, embedding_size, device=TEST_DEVICE).type(dtype)
            y = torch.randn(batch_size, embedding_size, device=TEST_DEVICE).type(dtype)
            loss = loss_fn(x, y)

            cx = np.cov(x.cpu().numpy(), rowvar=False)
            cy = np.cov(y.cpu().numpy(), rowvar=False)

            correct_loss = np.linalg.norm(cx - cy, ord="fro") ** 2
            correct_loss /= 4 * (embedding_size ** 2)
            self.assertTrue(np.isclose(loss.item(), correct_loss, rtol=1e-2))
