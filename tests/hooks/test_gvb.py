import copy
import unittest

import numpy as np
import torch

from pytorch_adapt.hooks import GVBEHook, GVBGANHook, GVBHook, validate_hook
from pytorch_adapt.layers import GradientReversal, ModelWithBridge
from pytorch_adapt.utils import common_functions as c_f

from .utils import Net, assertRequiresGrad, get_entropy_weights, get_opts


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
                models = {"G": G, "C": C, "D": D}
                opts = get_opts(models)
                hook_kwargs = {
                    "opts": list(opts.keys()),
                }
                if hook_cls is GVBEHook:
                    hook_kwargs["detach_entropy_reducer"] = detach_reducer

                data = {
                    "src_imgs": src_imgs,
                    "src_labels": src_labels,
                    "target_imgs": target_imgs,
                    "src_domain": src_domain,
                    "target_domain": target_domain,
                }

                h = hook_cls(**hook_kwargs)
                model_counts = validate_hook(h, list(data.keys()))
                losses, outputs = h({}, {**models, **opts, **data})
                self.assertTrue(
                    losses["total_loss"].keys()
                    == {
                        "d_src_bridge_loss",
                        "g_src_bridge_loss",
                        "d_target_bridge_loss",
                        "g_target_bridge_loss",
                        "c_loss",
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

                original_opts = get_opts(
                    {"G": originalG, "C": originalC, "D": originalD}
                )
                grl = GradientReversal()
                features = originalG(torch.cat([src_imgs, target_imgs], dim=0))
                logits, gbridge = originalC(features, return_bridge=True)
                dlogits, dbridge = originalD(
                    grl(torch.nn.functional.softmax(logits, dim=1)), return_bridge=True
                )
                self.assertTrue(
                    torch.equal(logits[:bs], outputs["src_imgs_features_logits"])
                )
                self.assertTrue(
                    torch.equal(logits[bs:], outputs["target_imgs_features_logits"])
                )
                self.assertTrue(
                    torch.equal(gbridge[:bs], outputs["src_imgs_features_gbridge"])
                )
                self.assertTrue(
                    torch.equal(gbridge[bs:], outputs["target_imgs_features_gbridge"])
                )
                self.assertTrue(
                    torch.equal(
                        dlogits[:bs], outputs["src_imgs_features_logits_dlogits"]
                    )
                )
                self.assertTrue(
                    torch.equal(
                        dlogits[bs:], outputs["target_imgs_features_logits_dlogits"]
                    )
                )
                self.assertTrue(
                    torch.equal(
                        dbridge[:bs], outputs["src_imgs_features_logits_dbridge"]
                    )
                )
                self.assertTrue(
                    torch.equal(
                        dbridge[bs:], outputs["target_imgs_features_logits_dbridge"]
                    )
                )

                total_loss = 0
                correct_loss = torch.nn.functional.cross_entropy(
                    logits[:bs], src_labels
                )
                self.assertTrue(np.isclose(losses["c_loss"], correct_loss.item()))
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
                [x.zero_grad() for x in original_opts.values()]
                total_loss.backward()
                [x.step() for x in original_opts.values()]
                for x, y in [(G, originalG), (C, originalC), (D, originalD)]:
                    self.assertTrue(
                        c_f.state_dicts_are_equal(
                            x.state_dict(), y.state_dict(), rtol=1e-6
                        )
                    )

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
        models = {"G": G, "C": C, "D": D}
        d_opts = get_opts({"D": D})
        g_opts = get_opts({"G": G, "C": C})
        h = GVBGANHook(d_opts=list(d_opts.keys()), g_opts=list(g_opts.keys()))
        data = {
            "src_imgs": src_imgs,
            "src_labels": src_labels,
            "target_imgs": target_imgs,
            "src_domain": src_domain,
            "target_domain": target_domain,
        }

        model_counts = validate_hook(h, list(data.keys()))

        losses, outputs = h({}, {**models, **d_opts, **g_opts, **data})

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
                "c_loss",
                "total",
            }
        )
        assertRequiresGrad(self, outputs)
        self.assertTrue(
            G.count == C.model.count == 2 == model_counts["G"] == model_counts["C"]
        )
        self.assertTrue(D.model.count == 4 == model_counts["D"])

        d_opts = get_opts({"D": originalD})["D_opt"]
        g_opts = get_opts({"G": originalG, "C": originalC})
        features = originalG(torch.cat([src_imgs, target_imgs], dim=0))
        logits, gbridge = originalC(features, return_bridge=True)
        dlogits, dbridge = originalD(
            torch.nn.functional.softmax(logits.detach(), dim=1), return_bridge=True
        )
        self.assertTrue(torch.equal(logits[:bs], outputs["src_imgs_features_logits"]))
        self.assertTrue(
            torch.equal(logits[bs:], outputs["target_imgs_features_logits"])
        )
        self.assertTrue(torch.equal(gbridge[:bs], outputs["src_imgs_features_gbridge"]))
        self.assertTrue(
            torch.equal(gbridge[bs:], outputs["target_imgs_features_gbridge"])
        )
        self.assertTrue(
            torch.equal(
                dlogits[:bs], outputs["src_imgs_features_logits_detached_dlogits"]
            )
        )
        self.assertTrue(
            torch.equal(
                dlogits[bs:], outputs["target_imgs_features_logits_detached_dlogits"]
            )
        )
        self.assertTrue(
            torch.equal(
                dbridge[:bs], outputs["src_imgs_features_logits_detached_dbridge"]
            )
        )
        self.assertTrue(
            torch.equal(
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

        d_opts.zero_grad()
        total_loss.backward()
        d_opts.step()

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
        self.assertTrue(losses["g_loss"]["c_loss"] == correct_loss)
        total_loss += correct_loss

        correct_loss = torch.mean(torch.abs(gbridge[:bs]))
        self.assertTrue(losses["g_loss"]["g_src_bridge_loss"] == correct_loss)
        total_loss += correct_loss

        correct_loss = torch.mean(torch.abs(gbridge[bs:]))
        self.assertTrue(losses["g_loss"]["g_target_bridge_loss"] == correct_loss)
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
        [x.zero_grad() for x in g_opts.values()]
        total_loss.backward()
        [x.step() for x in g_opts.values()]

        for x, y in [(G, originalG), (C, originalC), (D, originalD)]:
            self.assertTrue(
                c_f.state_dicts_are_equal(x.state_dict(), y.state_dict(), rtol=1e-6)
            )
