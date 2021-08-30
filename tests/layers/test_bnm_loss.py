import unittest

import torch

from pytorch_adapt.layers import BNMLoss

from .. import TEST_DEVICE, TEST_DTYPES


class TestBNMLoss(unittest.TestCase):
    def test_bnm_loss(self):
        for dtype in TEST_DTYPES:
            if dtype == torch.float16:
                continue
            batch_size = 32
            embedding_size = 100
            x = torch.randn(
                batch_size, embedding_size, device=TEST_DEVICE, requires_grad=True
            ).type(dtype)
            loss_fn = BNMLoss()
            loss = loss_fn(x)

            x = torch.nn.functional.softmax(x, dim=1)
            correct_loss = -torch.mean(torch.linalg.svdvals(x))
            self.assertTrue(torch.isclose(loss, correct_loss))
