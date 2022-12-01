import copy
import unittest

import numpy as np
import torch

from pytorch_adapt.adapters import GVB
from pytorch_adapt.containers import Models, Optimizers
from pytorch_adapt.hooks import GVBEHook, GVBGANHook, GVBHook, validate_hook
from pytorch_adapt.layers import GradientReversal, ModelWithBridge
from pytorch_adapt.utils import common_functions as c_f

from .utils import (
    Net,
    assert_equal_models,
    assertRequiresGrad,
    get_entropy_weights,
    get_opt_tuple,
    get_opts,
)


def test_equivalent_adapter(G, D, C, data):
    models = Models(
        {"G": copy.deepcopy(G), "D": copy.deepcopy(D), "C": copy.deepcopy(C)}
    )
    optimizers = Optimizers(get_opt_tuple())
    adapter = GVB(models, optimizers)
    adapter.training_step(data)
    return models


def get_models_and_data():
    bs = 100
    src_domain = torch.randint(0, 2, size=(bs,)).float()
    target_domain = torch.randint(0, 2, size=(bs,)).float()
    src_labels = torch.randint(0, 8, size=(bs,))
    src_imgs = torch.randn(bs, 32)
    target_imgs = torch.randn(bs, 32)
    G = Net(32, 16)
    C = ModelWithBridge(Net(16, 8))
    D = ModelWithBridge(Net(8, 1))
    originalG = copy.deepcopy(G)
    originalC = copy.deepcopy(C)
    originalD = copy.deepcopy(D)
    return (
        G,
        C,
        D,
        originalG,
        originalC,
        originalD,
        src_imgs,
        src_labels,
        target_imgs,
        src_domain,
        target_domain,
        bs,
    )


class TestGVB(unittest.TestCase):
    def test_gvb_hook(self):
        torch.manual_seed(509)
        for hook_cls in [GVBHook, GVBEHook]:
            for detach_reducer in [False, True]:
                if detach_reducer and hook_cls is not GVBEHook:
                    continue
                (
                    G,
                    C,
                    D,
                    originalG,
                    originalC,
                    originalD,
                    src_imgs,
                    src_labels,
                    target_imgs,
                    src_domain,
                    target_domain,
                    bs,
                ) = get_models_and_data()
                opts = get_opts(G, C, D)
                hook_kwargs = {
                    "opts": opts,
                }
                if hook_cls is GVBEHook:
                    hook_kwargs["detach_entropy_reducer"] = detach_reducer

                h = hook_cls(**hook_kwargs)

                data = {
                    "src_imgs": src_imgs,
                    "src_labels": src_labels,
                    "target_imgs": target_imgs,
                    "src_domain": src_domain,
                    "target_domain": target_domain,
                }

                model_counts = validate_hook(h, list(data.keys()))
                outputs, losses = h(locals())
                self.assertTrue(
                    losses["total_loss"].keys()
                    == {
                        "d_src_bridge_loss",
                        "g_src_bridge_loss",
                        "d_target_bridge_loss",
                        "g_target_bridge_loss",
                        "src_c_loss",
                        "src_domain_loss",
                        "target_domain_loss",
                        "total",
                    }
                )
                losses = losses["total_loss"]
                assertRequiresGrad(self, outputs)
                self.assertTrue(
                    G.count
                    == C.model.count
                    == D.model.count
                    == model_counts["G"]
                    == model_counts["C"]
                    == model_counts["D"]
                )

                adapter_models = None
                if hook_cls is GVBHook:
                    adapter_models = test_equivalent_adapter(
                        originalG, originalD, originalC, data
                    )

                original_opts = get_opts(originalG, originalC, originalD)
                grl = GradientReversal()
                features = originalG(torch.cat([src_imgs, target_imgs], dim=0))
                logits, gbridge = originalC(features, return_bridge=True)
                dlogits, dbridge = originalD(
                    grl(torch.nn.functional.softmax(logits, dim=1)), return_bridge=True
                )
                self.assertTrue(
                    torch.allclose(logits[:bs], outputs["src_imgs_features_logits"])
                )
                self.assertTrue(
                    torch.allclose(logits[bs:], outputs["target_imgs_features_logits"])
                )
                self.assertTrue(
                    torch.allclose(gbridge[:bs], outputs["src_imgs_features_gbridge"])
                )
                self.assertTrue(
                    torch.allclose(
                        gbridge[bs:], outputs["target_imgs_features_gbridge"]
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        dlogits[:bs], outputs["src_imgs_features_logits_dlogits"]
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        dlogits[bs:], outputs["target_imgs_features_logits_dlogits"]
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        dbridge[:bs], outputs["src_imgs_features_logits_dbridge"]
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        dbridge[bs:], outputs["target_imgs_features_logits_dbridge"]
                    )
                )

                total_loss = 0
                correct_loss = torch.nn.functional.cross_entropy(
                    logits[:bs], src_labels
                )
                self.assertTrue(np.isclose(losses["src_c_loss"], correct_loss.item()))
                total_loss += correct_loss

                correct_loss = torch.mean(torch.abs(gbridge[:bs]))
                self.assertTrue(losses["g_src_bridge_loss"] == correct_loss)
                total_loss += correct_loss

                correct_loss = torch.mean(torch.abs(gbridge[bs:]))
                self.assertTrue(losses["g_target_bridge_loss"] == correct_loss)
                total_loss += correct_loss

                correct_loss = torch.mean(torch.abs(dbridge[:bs]))
                self.assertTrue(losses["d_src_bridge_loss"] == correct_loss)
                total_loss += correct_loss

                correct_loss = torch.mean(torch.abs(dbridge[bs:]))
                self.assertTrue(losses["d_target_bridge_loss"] == correct_loss)
                total_loss += correct_loss

                src_domain_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    dlogits[:bs], src_domain, reduction="none"
                )
                target_domain_loss = (
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        dlogits[bs:], target_domain, reduction="none"
                    )
                )

                if hook_cls is GVBEHook:
                    (
                        src_entropy_weights,
                        target_entropy_weights,
                    ) = get_entropy_weights(grl(logits), bs, detach_reducer)
                    src_domain_loss = torch.mean(src_domain_loss * src_entropy_weights)
                    target_domain_loss = torch.mean(
                        target_domain_loss * target_entropy_weights
                    )
                else:
                    src_domain_loss = torch.mean(src_domain_loss)
                    target_domain_loss = torch.mean(target_domain_loss)

                self.assertTrue(losses["src_domain_loss"] == src_domain_loss)
                total_loss += src_domain_loss
                self.assertTrue(losses["target_domain_loss"] == target_domain_loss)
                total_loss += target_domain_loss

                total_loss /= 7
                [x.zero_grad() for x in original_opts]
                total_loss.backward()
                [x.step() for x in original_opts]

                g_models = [G, originalG]
                c_models = [C, originalC]
                d_models = [D, originalD]
                if adapter_models:
                    g_models.append(adapter_models["G"])
                    c_models.append(adapter_models["C"])
                    d_models.append(adapter_models["D"])

                assert_equal_models(self, g_models, rtol=1e-6)
                assert_equal_models(self, c_models, rtol=1e-6)
                assert_equal_models(self, d_models, rtol=1e-6)

    def test_gvb_gan_hook(self):
        torch.manual_seed(123)
        (
            G,
            C,
            D,
            originalG,
            originalC,
            originalD,
            src_imgs,
            src_labels,
            target_imgs,
            src_domain,
            target_domain,
            bs,
        ) = get_models_and_data()
        d_opts = get_opts(D)
        g_opts = get_opts(G, C)
        h = GVBGANHook(d_opts=d_opts, g_opts=g_opts)
        model_counts = validate_hook(
            h, ["src_imgs", "src_labels", "target_imgs", "src_domain", "target_domain"]
        )
        outputs, losses = h(locals())

        self.assertTrue(
            losses["d_loss"].keys()
            == {
                "d_src_bridge_loss",
                "d_target_bridge_loss",
                "d_src_domain_loss",
                "d_target_domain_loss",
                "total",
            }
        )
        self.assertTrue(
            losses["g_loss"].keys()
            == {
                "d_src_bridge_loss",
                "d_target_bridge_loss",
                "g_src_bridge_loss",
                "g_target_bridge_loss",
                "g_src_domain_loss",
                "g_target_domain_loss",
                "src_c_loss",
                "total",
            }
        )
        assertRequiresGrad(self, outputs)
        self.assertTrue(
            G.count == C.model.count == 2 == model_counts["G"] == model_counts["C"]
        )
        self.assertTrue(D.model.count == 4 == model_counts["D"])

        d_opts = get_opts(originalD)
        g_opts = get_opts(originalG, originalC)
        features = originalG(torch.cat([src_imgs, target_imgs], dim=0))
        logits, gbridge = originalC(features, return_bridge=True)
        dlogits, dbridge = originalD(
            torch.nn.functional.softmax(logits.detach(), dim=1), return_bridge=True
        )
        self.assertTrue(
            torch.allclose(logits[:bs], outputs["src_imgs_features_logits"])
        )
        self.assertTrue(
            torch.allclose(logits[bs:], outputs["target_imgs_features_logits"])
        )
        self.assertTrue(
            torch.allclose(gbridge[:bs], outputs["src_imgs_features_gbridge"])
        )
        self.assertTrue(
            torch.allclose(gbridge[bs:], outputs["target_imgs_features_gbridge"])
        )
        self.assertTrue(
            torch.allclose(
                dlogits[:bs], outputs["src_imgs_features_logits_detached_dlogits"]
            )
        )
        self.assertTrue(
            torch.allclose(
                dlogits[bs:], outputs["target_imgs_features_logits_detached_dlogits"]
            )
        )
        self.assertTrue(
            torch.allclose(
                dbridge[:bs], outputs["src_imgs_features_logits_detached_dbridge"]
            )
        )
        self.assertTrue(
            torch.allclose(
                dbridge[bs:], outputs["target_imgs_features_logits_detached_dbridge"]
            )
        )

        total_loss = 0
        correct_loss = torch.mean(torch.abs(dbridge[:bs]))
        self.assertTrue(losses["d_loss"]["d_src_bridge_loss"] == correct_loss)
        total_loss += correct_loss

        correct_loss = torch.mean(torch.abs(dbridge[bs:]))
        self.assertTrue(losses["d_loss"]["d_target_bridge_loss"] == correct_loss)
        total_loss += correct_loss

        correct_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            dlogits[:bs], src_domain
        )
        self.assertTrue(losses["d_loss"]["d_src_domain_loss"] == correct_loss)
        total_loss += correct_loss

        correct_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            dlogits[bs:], target_domain
        )
        self.assertTrue(losses["d_loss"]["d_target_domain_loss"] == correct_loss)
        total_loss += correct_loss
        total_loss /= 4

        d_opts[0].zero_grad()
        total_loss.backward()
        d_opts[0].step()

        dlogits, dbridge = originalD(
            torch.nn.functional.softmax(logits, dim=1), return_bridge=True
        )

        self.assertTrue(
            torch.allclose(dlogits[:bs], outputs["src_imgs_features_logits_dlogits"])
        )
        self.assertTrue(
            torch.allclose(dlogits[bs:], outputs["target_imgs_features_logits_dlogits"])
        )
        self.assertTrue(
            torch.allclose(dbridge[:bs], outputs["src_imgs_features_logits_dbridge"])
        )
        self.assertTrue(
            torch.allclose(dbridge[bs:], outputs["target_imgs_features_logits_dbridge"])
        )

        total_loss = 0
        correct_loss = torch.nn.functional.cross_entropy(logits[:bs], src_labels)
        self.assertTrue(np.isclose(losses["g_loss"]["src_c_loss"], correct_loss.item()))
        total_loss += correct_loss

        correct_loss = torch.mean(torch.abs(gbridge[:bs]))
        self.assertTrue(losses["g_loss"]["g_src_bridge_loss"] == correct_loss)
        total_loss += correct_loss

        correct_loss = torch.mean(torch.abs(gbridge[bs:]))
        self.assertAlmostEqual(
            losses["g_loss"]["g_target_bridge_loss"], correct_loss.item()
        )
        total_loss += correct_loss

        correct_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            dlogits[:bs], target_domain
        )
        self.assertTrue(losses["g_loss"]["g_src_domain_loss"] == correct_loss)
        total_loss += correct_loss

        correct_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            dlogits[bs:], src_domain
        )
        self.assertTrue(losses["g_loss"]["g_target_domain_loss"] == correct_loss)
        total_loss += correct_loss

        correct_loss = -torch.mean(torch.abs(dbridge[:bs]))
        self.assertTrue(losses["g_loss"]["d_src_bridge_loss"] == correct_loss)
        total_loss += correct_loss

        correct_loss = -torch.mean(torch.abs(dbridge[bs:]))
        self.assertTrue(losses["g_loss"]["d_target_bridge_loss"] == correct_loss)
        total_loss += correct_loss

        total_loss /= 7
        g_opts[0].zero_grad()
        g_opts[1].zero_grad()
        total_loss.backward()
        g_opts[0].step()
        g_opts[1].step()

        for x, y in [(G, originalG), (C, originalC), (D, originalD)]:
            self.assertTrue(
                c_f.state_dicts_are_equal(x.state_dict(), y.state_dict(), rtol=1e-6)
            )
