import unittest

import torch

from pytorch_adapt.hooks import CLossHook, SoftmaxHook

from .utils import Net, assertRequiresGrad


class TestClassification(unittest.TestCase):
    def test_softmax_hook(self):
        h = SoftmaxHook(apply_to=["src_imgs_features"])
        src_imgs_features = torch.randn(100, 10)
        losses, outputs = h({}, locals())
        self.assertTrue(losses == {})
        self.assertTrue(
            torch.equal(
                outputs["src_imgs_features"],
                torch.nn.functional.softmax(src_imgs_features, dim=1),
            )
        )

    def test_closs_hook(self):
        for detach_features in [True, False]:
            h = CLossHook(detach_features=detach_features)
            src_imgs = torch.randn(100, 32)
            target_imgs = torch.randn(100, 32)
            src_labels = torch.randint(0, 10, size=(100,))
            G = Net(32, 16)
            C = Net(16, 10)
            losses, outputs = h({}, locals())
            assertRequiresGrad(self, outputs)
            base_key = "src_imgs_features"
            if detach_features:
                base_key += "_detached"
            self.assertTrue(outputs.keys() == {base_key, f"{base_key}_logits"})
