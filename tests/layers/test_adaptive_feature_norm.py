import unittest

import numpy as np
import torch

from pytorch_adapt.layers import AdaptiveFeatureNorm, L2PreservedDropout

from .. import TEST_DEVICE, TEST_DTYPES


# https://github.com/jihanyang/AFN/blob/master/vanilla/Office31/SAFN/code/train.py
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

    def test_l2_preserved_dropout(self):
        for p in np.linspace(0.1, 0.9, 9):
            regular = torch.nn.Dropout(p=p)
            l2_preserved = L2PreservedDropout(p=p)
            emb = torch.randn(32, 128)
            regular.eval()
            l2_preserved.eval()
            self.assertTrue(torch.equal(regular(emb), l2_preserved(emb)))
            self.assertTrue(torch.equal(emb, l2_preserved(emb)))

            regular.train()
            l2_preserved.train()
            regular_out = regular(emb)
            l2_out = l2_preserved(emb)
            self.assertTrue(not torch.equal(regular_out, l2_out))
            self.assertTrue(not torch.equal(emb, l2_out))

            l1_norm = torch.norm(emb, p=1)
            l2_norm = torch.norm(emb, p=2)
            self.assertTrue(
                torch.isclose(torch.norm(regular_out, p=1), l1_norm, rtol=1e-1)
            )
            self.assertTrue(torch.isclose(torch.norm(l2_out, p=2), l2_norm, rtol=1e-1))
