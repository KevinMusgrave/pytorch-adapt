import copy
import unittest
from contextlib import nullcontext

import numpy as np
import torch

from pytorch_adapt.hooks import (
    AFNHook,
    BNMHook,
    BSPHook,
    CDANDomainHookD,
    CDANDomainHookG,
    CDANEHook,
    MCCHook,
    validate_hook,
)
from pytorch_adapt.layers import RandomizedDotProduct
from pytorch_adapt.utils import common_functions as c_f

from .utils import (
    assertRequiresGrad,
    get_entropy_weights,
    get_models_and_data,
    get_opts,
    post_g_hook_update_keys,
    post_g_hook_update_total_loss,
)


def get_correct_domain_losses(
    G,
    C,
    D,
    feature_combiner,
    src_imgs,
    target_imgs,
    src_domain,
    target_domain,
    softmax,
    bs,
):
    features = G(torch.cat([src_imgs, target_imgs], dim=0))
    c_logits = C(features)
    use_logits = c_logits
    if softmax:
        c_logits_softmaxed = torch.nn.functional.softmax(c_logits, dim=1)
        use_logits = c_logits_softmaxed
    combined_features = feature_combiner(features, use_logits)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    d_logits = D(combined_features)
    d_losses = {
        "d_src_domain_loss": loss_fn(d_logits[:bs], src_domain),
        "d_target_domain_loss": loss_fn(d_logits[bs:], target_domain),
    }
    g_losses = {
        "g_src_domain_loss": loss_fn(d_logits[:bs], target_domain),
        "g_target_domain_loss": loss_fn(d_logits[bs:], src_domain),
    }
    return features, c_logits, combined_features, d_logits, d_losses, g_losses


class TestCDAN(unittest.TestCase):
    def test_cdan_domain_hooks(self):
        for fc_out_size in [15, 32, 16]:
            bad_fc_size = fc_out_size in [15, 32]
            for softmax in [False, True]:
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
                feature_combiner = RandomizedDotProduct([16, 10], fc_out_size)
                d_hook = CDANDomainHookD(softmax=softmax)
                g_hook = CDANDomainHookG(softmax=softmax)

                ctx = self.assertRaises(RuntimeError) if bad_fc_size else nullcontext()
                with ctx:
                    losses_d, outputs_d = d_hook({}, locals())
                if bad_fc_size:
                    break
                self.assertTrue(G.count == C.count == D.count == 2)
                self.assertTrue(
                    outputs_d.keys()
                    == {
                        "src_imgs_features",
                        "target_imgs_features",
                        "src_imgs_features_logits",
                        "target_imgs_features_logits",
                        "src_imgs_features_AND_src_imgs_features_logits_combined",
                        "target_imgs_features_AND_target_imgs_features_logits_combined",
                        "src_imgs_features_AND_src_imgs_features_logits_combined_detached",
                        "target_imgs_features_AND_target_imgs_features_logits_combined_detached",
                        "src_imgs_features_AND_src_imgs_features_logits_combined_detached_dlogits",
                        "target_imgs_features_AND_target_imgs_features_logits_combined_detached_dlogits",
                    }
                )
                assertRequiresGrad(self, outputs_d)

                losses_g, outputs_g = g_hook({}, {**locals(), **outputs_d})
                self.assertTrue(G.count == C.count == 2)
                self.assertTrue(D.count == 4)
                self.assertTrue(
                    outputs_g.keys()
                    == {
                        "src_imgs_features_AND_src_imgs_features_logits_combined_dlogits",
                        "target_imgs_features_AND_target_imgs_features_logits_combined_dlogits",
                    }
                )
                assertRequiresGrad(self, outputs_g)

                bs = 100
                (
                    features,
                    c_logits,
                    combined_features,
                    d_logits,
                    d_losses,
                    g_losses,
                ) = get_correct_domain_losses(
                    G,
                    C,
                    D,
                    feature_combiner,
                    src_imgs,
                    target_imgs,
                    src_domain,
                    target_domain,
                    softmax,
                    bs,
                )

                self.assertTrue(
                    torch.equal(c_logits[:100], outputs_d["src_imgs_features_logits"])
                )
                self.assertTrue(
                    torch.equal(
                        c_logits[100:], outputs_d["target_imgs_features_logits"]
                    )
                )
                self.assertTrue(
                    all(
                        torch.equal(losses_d[k], d_losses[k])
                        for k in ["d_src_domain_loss", "d_target_domain_loss"]
                    )
                )
                self.assertTrue(
                    all(
                        torch.equal(losses_g[k], g_losses[k])
                        for k in ["g_src_domain_loss", "g_target_domain_loss"]
                    )
                )

    def test_cdan_hook(self):
        torch.manual_seed(985)
        for detach_reducer in [False, True]:
            for post_g in [None, BSPHook(), BNMHook(), MCCHook(), AFNHook()]:
                softmax = True
                fc_out_size = 16
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
                d_opts = get_opts({"D": D})
                g_opts = get_opts({"G": G, "C": C})
                feature_combiner = RandomizedDotProduct([16, 10], fc_out_size)

                originalG = copy.deepcopy(G)
                originalC = copy.deepcopy(C)
                originalD = copy.deepcopy(D)
                originalFeatureCombiner = copy.deepcopy(feature_combiner)

                models = {"G": G, "C": C, "D": D, "feature_combiner": feature_combiner}
                data = {
                    "src_imgs": src_imgs,
                    "target_imgs": target_imgs,
                    "src_labels": src_labels,
                    "src_domain": src_domain,
                    "target_domain": target_domain,
                }
                post_g_ = [post_g] if post_g is not None else post_g
                hook = CDANEHook(
                    detach_entropy_reducer=detach_reducer,
                    d_opts=list(d_opts.keys()),
                    g_opts=list(g_opts.keys()),
                    softmax=softmax,
                    post_g=post_g_,
                )
                model_counts = validate_hook(hook, list(data.keys()))
                losses, outputs = hook({}, {**models, **d_opts, **g_opts, **data})
                assertRequiresGrad(self, outputs)

                output_keys = {
                    "src_imgs_features",
                    "target_imgs_features",
                    "src_imgs_features_logits",
                    "target_imgs_features_logits",
                    "src_imgs_features_AND_src_imgs_features_logits_combined",
                    "target_imgs_features_AND_target_imgs_features_logits_combined",
                    "src_imgs_features_AND_src_imgs_features_logits_combined_detached",
                    "target_imgs_features_AND_target_imgs_features_logits_combined_detached",
                    "src_imgs_features_AND_src_imgs_features_logits_combined_detached_dlogits",
                    "target_imgs_features_AND_target_imgs_features_logits_combined_detached_dlogits",
                    "src_imgs_features_AND_src_imgs_features_logits_combined_dlogits",
                    "target_imgs_features_AND_target_imgs_features_logits_combined_dlogits",
                    "src_imgs_features_detached",
                    "target_imgs_features_detached",
                    "src_imgs_features_logits_detached",
                    "target_imgs_features_logits_detached",
                }

                g_loss_keys = {
                    "g_src_domain_loss",
                    "g_target_domain_loss",
                    "c_loss",
                    "total",
                }

                post_g_hook_update_keys(post_g, g_loss_keys, output_keys)
                self.assertTrue(outputs.keys() == output_keys)
                self.assertTrue(losses["g_loss"].keys() == g_loss_keys)

                self.assertTrue(
                    G.count == C.count == model_counts["G"] == model_counts["C"] == 2
                )
                self.assertTrue(D.count == model_counts["D"] == 4)

                d_opts = get_opts({"D": originalD})["D_opt"]
                g_opts = get_opts({"G": originalG, "C": originalC})

                bs = 100
                (
                    features,
                    c_logits,
                    combined_features,
                    d_logits,
                    d_losses,
                    g_losses,
                ) = get_correct_domain_losses(
                    originalG,
                    originalC,
                    originalD,
                    originalFeatureCombiner,
                    src_imgs,
                    target_imgs,
                    src_domain,
                    target_domain,
                    softmax,
                    bs,
                )

                src_entropy_weights, target_entropy_weights = get_entropy_weights(
                    c_logits, bs, detach_reducer
                )
                d_losses["d_src_domain_loss"] = torch.mean(
                    d_losses["d_src_domain_loss"] * src_entropy_weights
                )
                d_losses["d_target_domain_loss"] = torch.mean(
                    d_losses["d_target_domain_loss"] * target_entropy_weights
                )

                self.assertTrue(
                    all(
                        losses["d_loss"][k] == d_losses[k].item()
                        for k in ["d_src_domain_loss", "d_target_domain_loss"]
                    )
                )
                total_loss = sum(v for v in d_losses.values()) / len(d_losses)
                d_opts.zero_grad()
                total_loss.backward()
                d_opts.step()

                (
                    features,
                    c_logits,
                    combined_features,
                    d_logits,
                    d_losses,
                    g_losses,
                ) = get_correct_domain_losses(
                    originalG,
                    originalC,
                    originalD,
                    originalFeatureCombiner,
                    src_imgs,
                    target_imgs,
                    src_domain,
                    target_domain,
                    softmax,
                    bs,
                )

                src_entropy_weights, target_entropy_weights = get_entropy_weights(
                    c_logits, bs, detach_reducer
                )
                g_losses["g_src_domain_loss"] = torch.mean(
                    g_losses["g_src_domain_loss"] * src_entropy_weights
                )
                g_losses["g_target_domain_loss"] = torch.mean(
                    g_losses["g_target_domain_loss"] * target_entropy_weights
                )

                g_losses["c_loss"] = torch.nn.functional.cross_entropy(
                    c_logits[:bs], src_labels
                )

                self.assertTrue(
                    all(
                        np.isclose(losses["g_loss"][k], g_losses[k].item())
                        for k in ["g_src_domain_loss", "g_target_domain_loss", "c_loss"]
                    )
                )
                g_losses = list(g_losses.values())
                target_features = originalG(target_imgs)
                post_g_hook_update_total_loss(
                    post_g,
                    g_losses,
                    originalG(src_imgs),
                    target_features,
                    originalC(target_features),
                )
                total_loss = sum(g_losses) / len(g_losses)
                self.assertTrue(
                    np.isclose(total_loss.item(), losses["g_loss"]["total"])
                )
                [x.zero_grad() for x in g_opts.values()]
                total_loss.backward()
                [x.step() for x in g_opts.values()]

                for x, y in [(G, originalG), (C, originalC), (D, originalD)]:
                    self.assertTrue(
                        c_f.state_dicts_are_equal(
                            x.state_dict(), y.state_dict(), rtol=1e-2
                        )
                    )
