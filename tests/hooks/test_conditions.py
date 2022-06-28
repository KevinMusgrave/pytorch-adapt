import unittest

import numpy as np
import torch

from pytorch_adapt.hooks import StrongDHook

from .utils import Net


class TestConditions(unittest.TestCase):
    def test_strong_d_hook(self):
        torch.manual_seed(3420)
        for threshold in [0.1, 0.4, 0.6]:
            h = StrongDHook(threshold=threshold)
            src_domain = torch.zeros(10)
            src_domain[:5] = 1
            target_domain = torch.ones(10)
            src_imgs = torch.randn(10, 32)
            target_imgs = torch.randn(10, 32)
            G = Net(32, 16)
            D = Net(16, 1)

            result = h(locals())
            self.assertTrue(G.count == D.count == 2)

            logits = D(G(torch.cat([src_imgs, target_imgs], dim=0)))
            logits = torch.sigmoid(logits)
            preds = torch.round(logits)
            labels = torch.cat([src_domain, target_domain], dim=0)
            accuracy = torch.mean((preds == labels).float()).item()
            self.assertTrue(np.isclose(accuracy, h.accuracy_fn.accuracy, rtol=1e-2))
            self.assertTrue(result == (accuracy > threshold))
