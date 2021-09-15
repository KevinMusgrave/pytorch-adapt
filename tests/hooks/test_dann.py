import copy
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_adapt.hooks import (
    AFNHook,
    BNMHook,
    BSPHook,
    CDANNEHook,
    DANNEHook,
    DANNHook,
    MCCHook,
    validate_hook,
)
from pytorch_adapt.layers import GradientReversal, RandomizedDotProduct
from pytorch_adapt.utils import common_functions as c_f

from .utils import (
    Net,
    assertRequiresGrad,
    get_entropy_weights,
    get_models_and_data,
    get_opts,
    post_g_hook_update_keys,
    post_g_hook_update_total_loss,
)


class TestDANN(unittest.TestCase):
    def test_dann(self):
        for post_g in [None, BSPHook(), BNMHook(), MCCHook(), AFNHook()]:
            for hook_cls in [DANNHook, DANNEHook, CDANNEHook]:
                for detach_reducer in [False, True]:
                    if detach_reducer and hook_cls is DANNHook:
                        continue
                    if hook_cls in [DANNEHook, CDANNEHook] and post_g is not None:
                        continue

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
                    hook_kwargs = {
                        "opts": opts,
                        "post_g": post_g_,
                    }
                    if hook_cls in [DANNEHook, CDANNEHook]:
                        hook_kwargs["detach_entropy_reducer"] = detach_reducer
                    h = hook_cls(**hook_kwargs)

                    models = {"G": G, "D": D, "C": C}
                    if hook_cls is CDANNEHook:
                        feature_combiner = RandomizedDotProduct([16, 10], 16)
                        originalFeatureCombiner = copy.deepcopy(feature_combiner)
                        models["feature_combiner"] = feature_combiner

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
                    if hook_cls in [DANNEHook, CDANNEHook]:
                        if detach_reducer:
                            output_keys.update(
                                {
                                    "target_imgs_features_logits",
                                    "src_imgs_features_detached",
                                    "target_imgs_features_detached",
                                    "src_imgs_features_logits_detached",
                                    "target_imgs_features_logits_detached",
                                }
                            )
                        else:
                            output_keys.update({"target_imgs_features_logits"})
                    if hook_cls is CDANNEHook:
                        output_keys.update(
                            {
                                "src_imgs_features_AND_src_imgs_features_logits_combined",
                                "target_imgs_features_AND_target_imgs_features_logits_combined",
                                "src_imgs_features_AND_src_imgs_features_logits_combined_dlogits",
                                "target_imgs_features_AND_target_imgs_features_logits_combined_dlogits",
                            }
                        )
                        output_keys -= {
                            "src_imgs_features_dlogits",
                            "target_imgs_features_dlogits",
                        }

                    loss_keys = {
                        "src_domain_loss",
                        "target_domain_loss",
                        "c_loss",
                        "total",
                    }

                    post_g_hook_update_keys(post_g, loss_keys, output_keys)

                    self.assertTrue(outputs.keys() == output_keys)
                    # none of the outputs should have gradient reversal
                    # the gradient reversal is self-contained
                    self.assertTrue(
                        all(
                            "GradientReversal" not in str(x.grad_fn)
                            for x in outputs.values()
                        )
                    )

                    self.assertTrue(losses["total_loss"].keys() == loss_keys)
                    if hook_cls in [DANNEHook, CDANNEHook]:
                        correct_c_count = 2
                    elif post_g is None or isinstance(post_g, (BSPHook, AFNHook)):
                        correct_c_count = 1
                    else:
                        correct_c_count = 2
                    self.assertTrue(C.count == model_counts["C"] == correct_c_count)
                    self.assertTrue(
                        G.count
                        == D.count
                        == model_counts["G"]
                        == model_counts["D"]
                        == 2
                    )

                    opts = get_opts(originalD, originalG, originalC)

                    grl = GradientReversal()
                    src_features = originalG(src_imgs)
                    target_features = originalG(target_imgs)
                    src_logits = originalC(src_features)

                    if hook_cls in [DANNHook, DANNEHook]:
                        src_dlogits = originalD(grl(src_features))
                        target_dlogits = originalD(grl(target_features))
                    elif hook_cls is CDANNEHook:
                        src_dlogits = originalD(
                            grl(
                                originalFeatureCombiner(
                                    src_features,
                                    torch.nn.functional.softmax(src_logits, dim=1),
                                )
                            )
                        )
                        target_dlogits = originalD(
                            grl(
                                originalFeatureCombiner(
                                    target_features,
                                    torch.nn.functional.softmax(
                                        originalC(target_features), dim=1
                                    ),
                                )
                            )
                        )

                    src_domain_loss = F.binary_cross_entropy_with_logits(
                        src_dlogits, src_domain, reduction="none"
                    )
                    target_domain_loss = F.binary_cross_entropy_with_logits(
                        target_dlogits, target_domain, reduction="none"
                    )

                    if hook_cls in [DANNEHook, CDANNEHook]:
                        target_logits = originalC(target_features)
                        c_logits = torch.cat([src_logits, target_logits], dim=0)
                        c_logits = grl(c_logits)
                        (
                            src_entropy_weights,
                            target_entropy_weights,
                        ) = get_entropy_weights(
                            c_logits, len(target_logits), detach_reducer
                        )
                        src_domain_loss = torch.mean(
                            src_domain_loss * src_entropy_weights
                        )
                        target_domain_loss = torch.mean(
                            target_domain_loss * target_entropy_weights
                        )
                    else:
                        src_domain_loss = torch.mean(src_domain_loss)
                        target_domain_loss = torch.mean(target_domain_loss)

                    self.assertTrue(
                        src_domain_loss == losses["total_loss"]["src_domain_loss"]
                    )
                    self.assertTrue(
                        target_domain_loss == losses["total_loss"]["target_domain_loss"]
                    )

                    c_loss = F.cross_entropy(src_logits, src_labels)
                    self.assertTrue(
                        np.isclose(c_loss.item(), losses["total_loss"]["c_loss"])
                    )

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
                            c_f.state_dicts_are_equal(
                                x.state_dict(), y.state_dict(), rtol=1e-5
                            )
                        )
