import unittest

import torch

from pytorch_adapt.hooks import VATHook, VATPlusEntropyHook
from pytorch_adapt.layers import EntropyLoss

from .utils import Net, assertRequiresGrad


class TestVADA(unittest.TestCase):
    def test_vat_hook(self):
        torch.manual_seed(87948)
        src_imgs = torch.randn(100, 32)
        target_imgs = torch.randn(100, 32)
        G = Net(32, 16)
        C = Net(16, 10)
        combined_model = torch.nn.Sequential(G, C)
        h = VATHook()

        outputs, losses = h(locals())

        # 2 for src+target features/logits
        # 4 inside the vat loss
        self.assertTrue(G.count == C.count == 6)
        assertRequiresGrad(self, outputs)

        self.assertTrue(
            outputs.keys()
            == {
                "src_imgs_features",
                "target_imgs_features",
                "src_imgs_features_logits",
                "target_imgs_features_logits",
            }
        )
        self.assertTrue(losses.keys() == {"src_vat_loss", "target_vat_loss"})

    def test_vat_plus_entropy_hook(self):
        torch.manual_seed(27391)
        src_imgs = torch.randn(100, 32)
        target_imgs = torch.randn(100, 32)
        G = Net(32, 16)
        C = Net(16, 10)
        combined_model = torch.nn.Sequential(G, C)
        h = VATPlusEntropyHook()

        outputs, losses = h(locals())

        # 2 for src+target features/logits
        # 4 inside the vat loss
        self.assertTrue(G.count == C.count == 6)
        assertRequiresGrad(self, outputs)

        self.assertTrue(
            outputs.keys()
            == {
                "src_imgs_features",
                "target_imgs_features",
                "src_imgs_features_logits",
                "target_imgs_features_logits",
            }
        )
        self.assertTrue(
            losses.keys() == {"src_vat_loss", "target_vat_loss", "entropy_loss"}
        )
        correct_entropy_loss = EntropyLoss()(C(G(target_imgs)))
        self.assertTrue(correct_entropy_loss == losses["entropy_loss"])
