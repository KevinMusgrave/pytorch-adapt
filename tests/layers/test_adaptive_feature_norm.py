import unittest

import torch

from pytorch_adapt.layers import AdaptiveFeatureNorm

from .. import TEST_DEVICE, TEST_DTYPES


# https://github.com/thuml/Batch-Spectral-Penalization/blob/master/train.py
def original_implementation(x):
    radius = x.norm(p=2, dim=1).detach()
    assert radius.requires_grad == False
    radius = radius + 1.0
    l = ((x.norm(p=2, dim=1) - radius) ** 2).mean()
    return l


class TestAdaptiveFeatureNorm(unittest.TestCase):
    def test_adaptive_feature_norm(self):
        for dtype in TEST_DTYPES:
            batch_size = 32
            embedding_size = 128
            x = torch.randn(
                batch_size, embedding_size, device=TEST_DEVICE, requires_grad=True
            ).type(dtype)
            x.retain_grad()
            loss_fn = AdaptiveFeatureNorm(step_size=1)
            loss = loss_fn(x)
            correct_loss = original_implementation(x)
            self.assertTrue(torch.isclose(loss, correct_loss))

            loss.backward()
            grad1 = x.grad.clone()
            x.grad = None

            correct_loss.backward()
            grad2 = x.grad.clone()

            self.assertTrue(torch.allclose(grad1, grad2))
