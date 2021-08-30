import unittest

import torch

from pytorch_adapt.layers import BatchSpectralLoss

from .. import TEST_DEVICE, TEST_DTYPES


# https://github.com/thuml/Batch-Spectral-Penalization/blob/master/train.py
def original_implementation(x, y):
    _, s_s, _ = torch.svd(x)
    _, s_t, _ = torch.svd(y)
    sigma = torch.pow(s_s[0], 2) + torch.pow(s_t[0], 2)
    return sigma


class TestBatchSpectralLoss(unittest.TestCase):
    def test_batch_spectral_loss(self):
        for dtype in TEST_DTYPES:
            if dtype == torch.float16:
                continue
            batch_size = 32
            embedding_size = 128
            x = torch.randn(
                batch_size, embedding_size, device=TEST_DEVICE, requires_grad=True
            ).type(dtype)
            x.retain_grad()
            y = torch.randn(
                batch_size, embedding_size, device=TEST_DEVICE, requires_grad=True
            ).type(dtype)
            y.retain_grad()
            loss_fn = BatchSpectralLoss(k=1)
            loss = loss_fn(x) + loss_fn(y)
            correct_loss = original_implementation(x, y)
            self.assertTrue(torch.isclose(loss, correct_loss))

            loss.backward()
            grad1 = x.grad.clone()
            x.grad = None

            correct_loss.backward()
            grad2 = x.grad.clone()

            self.assertTrue(torch.allclose(grad1, grad2))
