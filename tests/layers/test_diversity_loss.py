import unittest

import torch

from pytorch_adapt.layers import DiversityLoss, EntropyLoss

from .. import TEST_DEVICE, TEST_DTYPES


class TestDiversityLoss(unittest.TestCase):
    def test_diversity_loss(self):
        loss_fn1 = DiversityLoss()
        loss_fn2 = EntropyLoss()
        for dtype in TEST_DTYPES:
            for batch_size in [1, 32]:
                embedding_size = 100
                x = torch.randn(
                    batch_size, embedding_size, device=TEST_DEVICE, requires_grad=True
                ).type(dtype)
                loss1 = loss_fn1(x)
                if batch_size == 1:
                    loss2 = loss_fn2(x)
                else:
                    loss2 = loss_fn2(torch.mean(x, dim=0).unsqueeze(0))
                self.assertTrue(torch.isclose(loss1, -loss2, rtol=1e-2))
