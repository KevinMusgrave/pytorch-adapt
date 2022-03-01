import copy
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_adapt.adapters import RTN
from pytorch_adapt.containers import Misc, Models, Optimizers
from pytorch_adapt.hooks import (
    FeaturesAndLogitsHook,
    ResidualHook,
    RTNAlignerHook,
    RTNHook,
    RTNLogitsHook,
    validate_hook,
)
from pytorch_adapt.layers import (
    EntropyLoss,
    MMDLoss,
    PlusResidual,
    RandomizedDotProduct,
)

from .utils import (
    Net,
    assert_equal_models,
    assertRequiresGrad,
    get_models_and_data,
    get_opt_tuple,
    get_opts,
)


def test_equivalent_adapter(G, C, R, feature_combiner, data):
    models = Models(
        {
            "G": copy.deepcopy(G),
            "C": copy.deepcopy(C),
            "residual_model": PlusResidual(copy.deepcopy(R)),
        }
    )
    optimizers = Optimizers(get_opt_tuple())
    misc = Misc({"feature_combiner": copy.deepcopy(feature_combiner)})
    adapter = RTN(models, optimizers, misc=misc)
    adapter.training_step(data)
    return models


class TestRTN(unittest.TestCase):
    def test_residual_hook(self):
        src_imgs = torch.randn(100, 32)
        target_imgs = torch.randn(100, 32)
        G = Net(32, 16)
        C = Net(16, 10)
        residual_layers = Net(10, 10)
        residual_model = PlusResidual(residual_layers)
        h1 = FeaturesAndLogitsHook()
        h2 = ResidualHook(domains=["src"])

        outputs1 = h1(locals())[0]
        outputs2 = h2({**locals(), **outputs1})[0]

        self.assertTrue(G.count == C.count == 2)
        self.assertTrue(residual_model.layer.count == 1)
        assertRequiresGrad(self, outputs1)
        assertRequiresGrad(self, outputs2)

        self.assertTrue(
            outputs1.keys()
            == {
                "src_imgs_features",
                "target_imgs_features",
                "src_imgs_features_logits",
                "target_imgs_features_logits",
            }
        )
        self.assertTrue(outputs2.keys() == {"src_imgs_features_logits_plus_residual"})

        C_out = C(G(src_imgs))

        self.assertTrue(
            torch.equal(
                C_out + residual_layers(C_out),
                outputs2["src_imgs_features_logits_plus_residual"],
            )
        )

    def test_rtn_logits_hook(self):
        src_imgs = torch.randn(100, 32)
        src_labels = torch.randint(0, 10, size=(100,))
        target_imgs = torch.randn(100, 32)
        G = Net(32, 16)
        C = Net(16, 10)
        residual_layers = Net(10, 10)
        residual_model = PlusResidual(residual_layers)
        h = RTNLogitsHook()

        outputs, losses = h(locals())

        self.assertTrue(G.count == C.count == 2)
        self.assertTrue(residual_model.layer.count == 1)
        assertRequiresGrad(self, outputs)

        self.assertTrue(
            outputs.keys()
            == {
                "src_imgs_features",
                "target_imgs_features",
                "src_imgs_features_logits",
                "target_imgs_features_logits",
                "src_imgs_features_logits_plus_residual",
            }
        )
        self.assertTrue(losses.keys() == {"entropy_loss", "c_loss"})

        correct_entropy_loss = EntropyLoss()(C(G(target_imgs)))
        self.assertTrue(correct_entropy_loss == losses["entropy_loss"])

        C_out = C(G(src_imgs))

        correct_c_loss = torch.nn.functional.cross_entropy(
            C_out + residual_layers(C_out), src_labels
        )
        self.assertTrue(torch.isclose(correct_c_loss, torch.mean(losses["c_loss"])))

    def test_rtn_aligner_hook(self):
        src_imgs = torch.randn(100, 32)
        target_imgs = torch.randn(100, 32)
        G = Net(32, 16)
        C = Net(16, 10)
        feature_combiner = RandomizedDotProduct([16, 10], 16)
        h = RTNAlignerHook()

        outputs, losses = h(locals())

        self.assertTrue(G.count == C.count == 2)
        assertRequiresGrad(self, outputs)

        self.assertTrue(
            outputs.keys()
            == {
                "src_imgs_features",
                "target_imgs_features",
                "src_imgs_features_logits",
                "target_imgs_features_logits",
                "src_imgs_features_AND_src_imgs_features_logits_combined",
                "target_imgs_features_AND_target_imgs_features_logits_combined",
            }
        )
        self.assertTrue(losses.keys() == {"features_confusion_loss"})

        src_features = G(src_imgs)
        src_logits = C(src_features)

        target_features = G(target_imgs)
        target_logits = C(target_features)

        src_combined_features = feature_combiner(src_features, src_logits)
        target_combined_features = feature_combiner(target_features, target_logits)

        confusion_loss = MMDLoss()(src_combined_features, target_combined_features)
        self.assertTrue(confusion_loss == losses["features_confusion_loss"])

    def test_rtn_hook(self):
        (
            G,
            C,
            D,
            src_imgs,
            src_labels,
            target_imgs,
            src_domain,
            target_domain,
        ) = get_models_and_data()
        feature_combiner = RandomizedDotProduct([16, 10], 16)

        residual_layers = Net(10, 10)
        residual_model = PlusResidual(residual_layers)

        originalG = copy.deepcopy(G)
        originalC = copy.deepcopy(C)
        originalR = copy.deepcopy(residual_layers)

        opts = get_opts(G, C, residual_model)
        h = RTNHook(opts)
        models = {
            "G": G,
            "C": C,
            "feature_combiner": feature_combiner,
            "residual_model": residual_model,
        }
        data = {
            "src_imgs": src_imgs,
            "target_imgs": target_imgs,
            "src_labels": src_labels,
        }
        model_counts = validate_hook(h, list(data.keys()))

        outputs, losses = h({**models, **data})
        assertRequiresGrad(self, outputs)
        self.assertTrue(
            outputs.keys()
            == {
                "src_imgs_features",
                "src_imgs_features_logits",
                "target_imgs_features",
                "target_imgs_features_logits",
                "src_imgs_features_AND_src_imgs_features_logits_combined",
                "target_imgs_features_AND_target_imgs_features_logits_combined",
                "src_imgs_features_logits_plus_residual",
            }
        )

        loss_keys = {
            "features_confusion_loss",
            "entropy_loss",
            "c_loss",
            "total",
        }

        self.assertTrue(losses["total_loss"].keys() == loss_keys)
        self.assertTrue(
            G.count == model_counts["G"] == C.count == model_counts["C"] == 2
        )

        adapter_models = test_equivalent_adapter(
            originalG, originalC, originalR, feature_combiner, data
        )

        opts = get_opts(originalG, originalC, originalR)

        src_features = originalG(src_imgs)
        target_features = originalG(target_imgs)
        src_logits = originalC(src_features)
        target_logits = originalC(target_features)

        c_loss = F.cross_entropy(originalR(src_logits) + src_logits, src_labels)
        src_features = feature_combiner(src_features, src_logits)
        target_features = feature_combiner(target_features, target_logits)
        f_loss = MMDLoss()(src_features, target_features)
        entropy_loss = EntropyLoss()(target_logits)
        total_loss = (c_loss + f_loss + entropy_loss) / 3
        correct_losses = [c_loss, entropy_loss, f_loss, total_loss]

        computed_losses = [losses["total_loss"][k] for k in sorted(list(loss_keys))]
        self.assertTrue(
            all(
                np.isclose(x.item(), y) for x, y in zip(correct_losses, computed_losses)
            )
        )

        [x.zero_grad() for x in opts]
        total_loss.backward()
        [x.step() for x in opts]

        assert_equal_models(self, (G, adapter_models["G"], originalG), rtol=1e-6)
        assert_equal_models(self, (C, adapter_models["C"], originalC), rtol=1e-6)
        assert_equal_models(
            self,
            (residual_layers, adapter_models["residual_model"].layer, originalR),
            rtol=1e-6,
        )
