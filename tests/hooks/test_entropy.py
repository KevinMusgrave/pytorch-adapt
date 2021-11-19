import unittest

import torch

from pytorch_adapt.hooks import TargetEntropyHook
from pytorch_adapt.layers import EntropyLoss

from .utils import Net, assertRequiresGrad


class TestEntropy(unittest.TestCase):
    def test_entropy_hook(self):
        h = TargetEntropyHook()
        src_imgs = torch.randn(100, 32)
        target_imgs = torch.randn(100, 32)
        src_labels = torch.randint(0, 10, size=(100,))
        G = Net(32, 16)
        C = Net(16, 10)
        losses, outputs = h({}, locals())
        assertRequiresGrad(self, outputs)

        base_key = "target_imgs_features"
        self.assertTrue(outputs.keys() == {base_key, f"{base_key}_logits"})
        self.assertTrue(losses == {"entropy_loss": EntropyLoss()(C(G(target_imgs)))})
