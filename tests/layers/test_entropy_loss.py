import unittest

import torch

from pytorch_adapt.layers import EntropyLoss

from .. import TEST_DEVICE, TEST_DTYPES


class TestEntropyLoss(unittest.TestCase):
    def test_entropy_loss(self):
        loss_fn1 = EntropyLoss()
        loss_fn2 = EntropyLoss(after_softmax=True)
        softmax_fn = torch.nn.Softmax(dim=1)
        for dtype in TEST_DTYPES:
            batch_size = 32
            embedding_size = 100
            x = torch.randn(
                batch_size, embedding_size, device=TEST_DEVICE, requires_grad=True
            ).type(dtype)
            loss1 = loss_fn1(x)
            loss2 = loss_fn2(softmax_fn(x))
            self.assertTrue(torch.isclose(loss1, loss2, rtol=1e-2))
