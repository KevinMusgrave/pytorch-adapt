import copy
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_adapt.hooks import (
    AFNHook,
    BNMHook,
    BSPHook,
    DANNHook,
    MCCHook,
    validate_hook,
)
from pytorch_adapt.layers import GradientReversal
from pytorch_adapt.utils import common_functions as c_f

from .utils import (
    Net,
    assertRequiresGrad,
    get_models_and_data,
    get_opts,
    post_g_hook_update_keys,
    post_g_hook_update_total_loss,
)


class TestDANN(unittest.TestCase):
    def test_dann(self):
        for post_g in [None, BSPHook(), BNMHook(), MCCHook(), AFNHook()]:
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
            originalG = copy.deepcopy(G)
            originalC = copy.deepcopy(C)
            originalD = copy.deepcopy(D)

            opts = get_opts(G, C, D)
            post_g_ = [post_g] if post_g is not None else post_g
            h = DANNHook(opts, post_g=post_g_)
            models = {"G": G, "D": D, "C": C}
            data = {
                "src_imgs": src_imgs,
                "target_imgs": target_imgs,
                "src_labels": src_labels,
                "src_domain": src_domain,
                "target_domain": target_domain,
            }
            model_counts = validate_hook(h, list(data.keys()))

            losses, outputs = h({}, {**models, **data})
            assertRequiresGrad(self, outputs)

            output_keys = {
                "src_imgs_features",
                "src_imgs_features_dlogits",
                "target_imgs_features",
                "target_imgs_features_dlogits",
                "src_imgs_features_logits",
            }

            loss_keys = {"src_domain_loss", "target_domain_loss", "c_loss", "total"}

            post_g_hook_update_keys(post_g, loss_keys, output_keys)

            self.assertTrue(outputs.keys() == output_keys)
            # none of the outputs should have gradient reversal
            # the gradient reversal is self-contained
            self.assertTrue(
                all("GradientReversal" not in str(x.grad_fn) for x in outputs.values())
            )

            self.assertTrue(losses["total_loss"].keys() == loss_keys)
            correct_c_count = (
                1 if post_g is None or isinstance(post_g, (BSPHook, AFNHook)) else 2
            )
            self.assertTrue(C.count == model_counts["C"] == correct_c_count)
            self.assertTrue(
                G.count == D.count == model_counts["G"] == model_counts["D"] == 2
            )

            opts = get_opts(originalD, originalG, originalC)

            grl = GradientReversal()
            src_features = originalG(src_imgs)
            target_features = originalG(target_imgs)
            src_logits = originalC(src_features)
            src_dlogits = originalD(grl(src_features))
            target_dlogits = originalD(grl(target_features))

            src_domain_loss = F.binary_cross_entropy_with_logits(
                src_dlogits, src_domain
            )
            target_domain_loss = F.binary_cross_entropy_with_logits(
                target_dlogits, target_domain
            )
            self.assertTrue(src_domain_loss == losses["total_loss"]["src_domain_loss"])
            self.assertTrue(
                target_domain_loss == losses["total_loss"]["target_domain_loss"]
            )

            c_loss = F.cross_entropy(src_logits, src_labels)
            self.assertTrue(np.isclose(c_loss.item(), losses["total_loss"]["c_loss"]))

            total_loss = [src_domain_loss, target_domain_loss, c_loss]
            post_g_hook_update_total_loss(
                post_g,
                total_loss,
                src_features,
                target_features,
                originalC(target_features),
            )
            total_loss = sum(total_loss) / len(total_loss)
            self.assertTrue(
                np.isclose(total_loss.item(), losses["total_loss"]["total"])
            )
            [x.zero_grad() for x in opts]
            total_loss.backward()
            [x.step() for x in opts]

            for x, y in [(G, originalG), (C, originalC), (D, originalD)]:
                self.assertTrue(
                    c_f.state_dicts_are_equal(x.state_dict(), y.state_dict(), rtol=1e-5)
                )
