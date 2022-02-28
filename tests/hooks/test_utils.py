import unittest

import torch

from pytorch_adapt.hooks import AssertHook, ChainHook, DomainLossHook, FeaturesHook

from .utils import Net


class TestUtils(unittest.TestCase):
    def test_assert_hook(self):
        src_imgs = torch.randn(100, 32)
        target_imgs = torch.randn(100, 32)
        src_domain = torch.randint(0, 2, size=(100,)).float()
        target_domain = torch.randint(0, 2, size=(100,)).float()
        G = Net(32, 16)
        D = Net(16, 1)
        h = AssertHook(DomainLossHook(), "_dlogits$")
        with self.assertRaises(ValueError):
            outputs, losses = h(locals())

        h = ChainHook(FeaturesHook(), h)
        outputs, losses = h(locals())
