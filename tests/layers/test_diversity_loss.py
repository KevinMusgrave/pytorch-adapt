import unittest

import torch

from pytorch_adapt.layers import DiversityLoss, EntropyLoss

from .. import TEST_DEVICE, TEST_DTYPES


class TestDiversityLoss(unittest.TestCase):
    def test_diversity_loss(self):
        loss_fn2 = EntropyLoss(after_softmax=True)
        for after_softmax in [False, True]:
            loss_fn1 = DiversityLoss(after_softmax=after_softmax)
            for dtype in TEST_DTYPES:
                for batch_size in [1, 32]:
                    embedding_size = 100
                    logits = torch.randn(
                        batch_size,
                        embedding_size,
                        device=TEST_DEVICE,
                        requires_grad=True,
                    ).type(dtype)
                    preds = torch.softmax(logits, dim=1)
                    if after_softmax:
                        loss1 = loss_fn1(preds)
                    else:
                        loss1 = loss_fn1(logits)
                    if batch_size == 1:
                        loss2 = loss_fn2(preds)
                    else:
                        loss2 = loss_fn2(torch.mean(preds, dim=0).unsqueeze(0))
                    self.assertTrue(torch.isclose(loss1, -loss2))
