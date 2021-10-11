import unittest

import torch
import torch.nn.functional as F

from pytorch_adapt.layers import NLLLoss

from .. import TEST_DEVICE, TEST_DTYPES


class TestNLLLoss(unittest.TestCase):
    def test_nll_loss(self):
        for dtype in TEST_DTYPES:
            if dtype == torch.float16:
                continue
            batch_size = 128
            num_cls = 10
            x = torch.randn(
                batch_size, num_cls, device=TEST_DEVICE, requires_grad=True
            ).type(dtype)
            y = torch.randint(0, num_cls, size=(batch_size,), device=TEST_DEVICE)
            loss_fn = NLLLoss()
            loss = loss_fn(F.softmax(x, dim=1), y)

            correct_loss = F.nll_loss(F.log_softmax(x, dim=1), y)
            self.assertTrue(torch.isclose(loss, correct_loss))
