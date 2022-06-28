import unittest

import torch

from pytorch_adapt.hooks import BSPHook
from pytorch_adapt.layers.batch_spectral_loss import batch_spectral_loss

from .utils import Net, assertRequiresGrad


class TestBSP(unittest.TestCase):
    def test_bsp_hook(self):
        torch.manual_seed(453094)
        h = BSPHook()
        src_imgs = torch.randn(100, 32)
        target_imgs = torch.randn(100, 32)
        G = Net(32, 16)

        outputs, losses = h(locals())
        self.assertTrue(G.count == 2)
        assertRequiresGrad(self, outputs)

        outputs, losses2 = h({**locals(), **outputs})
        assertRequiresGrad(self, outputs)
        self.assertTrue(G.count == 2)
        self.assertTrue(losses == losses2)

        src_features = G(src_imgs)
        target_features = G(target_imgs)

        self.assertTrue(
            losses["bsp_loss"]
            == batch_spectral_loss(src_features, 1)
            + batch_spectral_loss(target_features, 1)
        )
