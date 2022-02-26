import unittest

import torch

from pytorch_adapt.hooks import (
    DLogitsHook,
    FeaturesAndLogitsHook,
    FeaturesHook,
    LogitsHook,
)
from pytorch_adapt.utils import common_functions as c_f

from .utils import Net, assertRequiresGrad


class TestFeatures(unittest.TestCase):
    def test_features_hook_logits_hook(self):
        for hook_type in ["features", "logits"]:
            G = Net(32, 8)
            if hook_type == "features":
                src_imgs = torch.randn(100, 32)
                target_imgs = torch.randn(100, 32)
                hook = FeaturesHook()
                src_key = "src_imgs_features"
                target_key = "target_imgs_features"
            else:
                src_imgs_features = torch.randn(100, 32)
                target_imgs_features = torch.randn(100, 32)
                hook = LogitsHook(key_map={"G": "C"})
                src_key = "src_imgs_features_logits"
                target_key = "target_imgs_features_logits"

            # none yet computed
            losses, outputs = hook({}, locals())
            self.assertTrue(losses == {})
            self.assertTrue(
                outputs.keys()
                == {
                    src_key,
                    target_key,
                }
            )
            self.assertTrue(G.count == 2)
            assertRequiresGrad(self, outputs)

            # both already computed
            losses, outputs = hook({}, {**locals(), **outputs})
            self.assertTrue(losses == {})
            self.assertTrue(outputs == {})
            self.assertTrue(G.count == 2)
            assertRequiresGrad(self, outputs)

            # src_features already computed
            if hook_type == "features":
                src_imgs_features = torch.randn(8)
            else:
                src_imgs_features_logits = torch.randn(8)
            losses, outputs = hook({}, {**locals(), **outputs})
            self.assertTrue(losses == {})
            self.assertTrue(G.count == 3)
            assertRequiresGrad(self, outputs)

            [x] = c_f.extract(outputs, [target_key], pop=True)
            if hook_type == "features":
                target_imgs_features = x
                hook = FeaturesHook(detach=True)
            else:
                target_imgs_features_logits = x
                hook = LogitsHook(detach=True, key_map={"G": "C"})
            outputs = {}
            losses, outputs = hook({}, {**locals(), **outputs})
            self.assertTrue(
                outputs.keys() == {f"{src_key}_detached", f"{target_key}_detached"}
            )
            [src_x, target_x] = c_f.extract(locals(), [src_key, target_key])
            assertRequiresGrad(self, outputs)
            self.assertTrue(
                torch.equal(x, y) for x, y in zip(outputs.values(), [src_x, target_x])
            )
            # count remains the same because the with_grad tensors already exist in locals
            self.assertTrue(G.count == 3)

    def test_features_and_logits_hook(self):
        G = Net(32, 8)
        C = Net(8, 2)
        src_imgs = torch.randn(100, 32)
        target_imgs = torch.randn(100, 32)
        hook = FeaturesAndLogitsHook()
        # none yet computed
        losses, outputs = hook({}, locals())
        self.assertTrue(losses == {})
        self.assertTrue(
            outputs.keys()
            == {
                "src_imgs_features",
                "target_imgs_features",
                "src_imgs_features_logits",
                "target_imgs_features_logits",
            }
        )
        self.assertTrue(all(x.count == 2 for x in [G, C]))
        assertRequiresGrad(self, outputs)

        # both already computed
        losses, outputs = hook({}, {**locals(), **outputs})
        self.assertTrue(all(x == {} for x in [losses, outputs]))
        self.assertTrue(all(x.count == 2 for x in [G, C]))

        src_imgs_features = torch.randn(8)
        target_imgs_features_logits = torch.randn(2)
        losses, outputs = hook({}, {**locals(), **outputs})
        self.assertTrue(losses == {})
        self.assertTrue(
            outputs.keys() == {"target_imgs_features", "src_imgs_features_logits"}
        )
        self.assertTrue(all(x.count == 3 for x in [G, C]))

        hook = FeaturesAndLogitsHook(detach_features=True)
        losses, outputs = hook({}, {**locals(), **outputs})
        self.assertTrue(
            outputs.keys()
            == {
                "src_imgs_features_detached",
                "target_imgs_features_detached",
                "src_imgs_features_detached_logits",
                "target_imgs_features_detached_logits",
            }
        )
        assertRequiresGrad(self, outputs)
        self.assertTrue(G.count == 3)
        self.assertTrue(C.count == 5)

        losses, outputs = hook({}, {**locals(), **outputs})
        self.assertTrue(G.count == 3)
        self.assertTrue(C.count == 5)

    def test_d_logits_hook(self):
        D = Net(32, 2)
        src_imgs_features = torch.randn(100, 32)
        target_imgs_features = torch.randn(100, 32)
        hook = DLogitsHook()
        # none yet computed
        losses, outputs = hook({}, locals())
        self.assertTrue(losses == {})
        self.assertTrue(
            outputs.keys()
            == {
                "src_imgs_features_dlogits",
                "target_imgs_features_dlogits",
            }
        )
        self.assertTrue(D.count == 2)
        assertRequiresGrad(self, outputs)

        hook = DLogitsHook(detach=True)
        with torch.no_grad():
            losses, outputs = hook({}, {**locals(), **outputs})
        self.assertTrue(
            outputs.keys()
            == {
                "src_imgs_features_dlogits_detached",
                "target_imgs_features_dlogits_detached",
            }
        )
        assertRequiresGrad(self, outputs)
        self.assertTrue(D.count == 2)

        with self.assertRaises(ValueError):
            hook = DLogitsHook(detach=False)
            with torch.no_grad():
                losses, outputs = hook({}, locals())

    def test_multiple_detach_modes(self):
        G = Net(32, 8)
        C = Net(8, 2)
        src_imgs = torch.randn(100, 32)
        target_imgs = torch.randn(100, 32)
        hook = FeaturesHook(detach={"src": True, "target": False})
        _, outputs = hook({}, locals())
        assertRequiresGrad(self, outputs)
        self.assertTrue(
            outputs.keys() == {"src_imgs_features_detached", "target_imgs_features"}
        )

        hook = FeaturesAndLogitsHook(detach_features={"src": False, "target": True})
        _, outputs = hook({}, locals())
        assertRequiresGrad(self, outputs)
        self.assertTrue(
            outputs.keys()
            == {
                "src_imgs_features",
                "target_imgs_features_detached",
                "src_imgs_features_logits",
                "target_imgs_features_detached_logits",
            }
        )
