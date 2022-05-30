import unittest

import torch

from pytorch_adapt.hooks import TargetDiversityHook
from pytorch_adapt.layers import DiversityLoss

from .utils import Net, assertRequiresGrad


class TestDiversity(unittest.TestCase):
    def test_diversity_hook(self):
        h = TargetDiversityHook()
        src_imgs = torch.randn(100, 32)
        target_imgs = torch.randn(100, 32)
        src_labels = torch.randint(0, 10, size=(100,))
        G = Net(32, 16)
        C = Net(16, 10)
        outputs, losses = h(locals())
        assertRequiresGrad(self, outputs)

        base_key = "target_imgs_features"
        self.assertTrue(outputs.keys() == {base_key, f"{base_key}_logits"})
        self.assertTrue(
            losses == {"diversity_loss": DiversityLoss()(C(G(target_imgs)))}
        )

        # manual calc
        logits = C(G(target_imgs))
        preds = torch.softmax(logits, dim=1)
        mean_preds = torch.mean(preds, dim=0)
        entropy = -torch.sum(mean_preds * torch.log(mean_preds))
        self.assertTrue(losses["diversity_loss"] == -entropy)
