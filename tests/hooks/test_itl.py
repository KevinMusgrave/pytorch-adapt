import unittest

import torch

from pytorch_adapt.hooks import ISTLossHook
from pytorch_adapt.layers import ISTLoss

from .utils import assertRequiresGrad, get_models_and_data


class TestITL(unittest.TestCase):
    def test_ist_loss_hook(self):
        torch.manual_seed(334)
        h = ISTLossHook()
        (
            G,
            _,
            _,
            src_imgs,
            _,
            target_imgs,
            src_domain,
            target_domain,
        ) = get_models_and_data()

        outputs, losses = h(locals())
        self.assertTrue(G.count == 2)
        assertRequiresGrad(self, outputs)

        outputs, losses2 = h({**locals(), **outputs})
        assertRequiresGrad(self, outputs)
        self.assertTrue(G.count == 2)
        self.assertTrue(losses == losses2)

        src_features = G(src_imgs)
        target_features = G(target_imgs)

        loss_fn = ISTLoss()
        self.assertTrue(
            losses["ist_loss"]
            == loss_fn(
                torch.cat([src_features, target_features], dim=0),
                torch.cat([src_domain, target_domain], dim=0),
            )
        )
