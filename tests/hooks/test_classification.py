import unittest

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_adapt.hooks import (
    ClassifierHook,
    CLossHook,
    MultiLabelClassifierHook,
    SoftmaxHook,
)

from .utils import Net, assertRequiresGrad, get_models_and_data, get_opts


class TestClassification(unittest.TestCase):
    def test_softmax_hook(self):
        torch.manual_seed(8112)
        h = SoftmaxHook(apply_to=["src_imgs_features"])
        src_imgs_features = torch.randn(100, 10)
        outputs, losses = h(locals())
        self.assertTrue(losses == {})
        self.assertTrue(
            torch.equal(
                outputs["src_imgs_features"],
                torch.nn.functional.softmax(src_imgs_features, dim=1),
            )
        )

    def test_closs_hook(self):
        torch.manual_seed(24242)
        src_imgs = torch.randn(100, 32)
        target_imgs = torch.randn(100, 32)
        src_labels = torch.randint(0, 10, size=(100,))
        target_labels = torch.randint(0, 10, size=(100,))
        G = Net(32, 16)
        C = Net(16, 10)

        for detach_features in [True, False]:
            for domains in [None, ("src",), ("target",), ("src", "target")]:
                if domains is None:
                    h = CLossHook(detach_features=detach_features)
                else:
                    h = CLossHook(detach_features=detach_features, domains=domains)
                outputs, losses = h(locals())
                assertRequiresGrad(self, outputs)
                base_keys = (
                    [f"{d}_imgs_features" for d in domains]
                    if domains
                    else ["src_imgs_features"]
                )
                if detach_features:
                    base_keys = [f"{x}_detached" for x in base_keys]
                logit_keys = [f"{x}_logits" for x in base_keys]
                self.assertTrue(outputs.keys() == {*base_keys, *logit_keys})

                correct_loss_fn = torch.nn.functional.cross_entropy
                for k, v in losses.items():
                    if k.startswith("src"):
                        self.assertTrue(
                            torch.equal(
                                v,
                                correct_loss_fn(
                                    C(G(src_imgs)), src_labels, reduction="none"
                                ),
                            )
                        )
                    elif k.startswith("target"):
                        self.assertTrue(
                            torch.equal(
                                v,
                                correct_loss_fn(
                                    C(G(target_imgs)), target_labels, reduction="none"
                                ),
                            )
                        )
                    else:
                        raise KeyError

    def test_classifier_hook(self):
        torch.manual_seed(53430)

        num_classes = 12

        for loss_fn, hook_cls in [
            (F.cross_entropy, ClassifierHook),
            (F.binary_cross_entropy_with_logits, MultiLabelClassifierHook),
        ]:
            (
                G,
                C,
                _,
                src_imgs,
                src_labels,
                _,
                _,
                _,
            ) = get_models_and_data(num_classes=num_classes)

            src_labels_for_loss_fn = src_labels
            if hook_cls is MultiLabelClassifierHook:
                src_labels = torch.randint(
                    0, 2, size=(src_labels.shape[0], num_classes)
                )
                src_labels_for_loss_fn = src_labels.float()

            correct = loss_fn(C(G(src_imgs)), src_labels_for_loss_fn).item()
            opts = get_opts(G, C)
            h = hook_cls(opts)
            _, losses = h(
                {"G": G, "C": C, "src_imgs": src_imgs, "src_labels": src_labels}
            )
            self.assertTrue(np.isclose(losses["total_loss"]["c_loss"], correct))
