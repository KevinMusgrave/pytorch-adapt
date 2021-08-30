import copy
import unittest

import torch

from pytorch_adapt.hooks import GVBGANHook, GVBHook, validate_hook
from pytorch_adapt.layers import GradientReversal, ModelWithBridge
from pytorch_adapt.utils import common_functions as c_f

from .utils import Net, assertRequiresGrad, get_opts


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
        h = GVBHook(opts=opts)
        model_counts = validate_hook(
            h, ["src_imgs", "src_labels", "target_imgs", "src_domain", "target_domain"]
        )
        losses, outputs = h({}, locals())
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

        original_opts = get_opts(originalG, originalC, originalD)
        grl = GradientReversal()
        features = originalG(torch.cat([src_imgs, target_imgs], dim=0))
        logits, gbridge = originalC(features, return_bridge=True)
        dlogits, dbridge = originalD(
            grl(torch.nn.functional.softmax(logits, dim=1)), return_bridge=True
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
            torch.equal(dlogits[:bs], outputs["src_imgs_features_logits_dlogits"])
        )
        self.assertTrue(
            torch.equal(dlogits[bs:], outputs["target_imgs_features_logits_dlogits"])
        )
        self.assertTrue(
            torch.equal(dbridge[:bs], outputs["src_imgs_features_logits_dbridge"])
        )
        self.assertTrue(
            torch.equal(dbridge[bs:], outputs["target_imgs_features_logits_dbridge"])
        )

        total_loss = 0
        correct_loss = torch.nn.functional.cross_entropy(logits[:bs], src_labels)
        self.assertTrue(losses["c_loss"] == correct_loss)
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

        correct_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            dlogits[:bs], src_domain
        )
        self.assertTrue(losses["src_domain_loss"] == correct_loss)
        total_loss += correct_loss

        correct_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            dlogits[bs:], target_domain
        )
        self.assertTrue(losses["target_domain_loss"] == correct_loss)
        total_loss += correct_loss

        total_loss /= 7
        [x.zero_grad() for x in original_opts]
        total_loss.backward()
        [x.step() for x in original_opts]
        for x, y in [(G, originalG), (C, originalC), (D, originalD)]:
            self.assertTrue(
                c_f.state_dicts_are_equal(x.state_dict(), y.state_dict(), rtol=1e-6)
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
        d_opts = get_opts(D)
        g_opts = get_opts(G, C)
        h = GVBGANHook(d_opts=d_opts, g_opts=g_opts)
        model_counts = validate_hook(
            h, ["src_imgs", "src_labels", "target_imgs", "src_domain", "target_domain"]
        )
        losses, outputs = h({}, locals())

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

        d_opts = get_opts(originalD)
        g_opts = get_opts(originalG, originalC)
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
        g_opts[0].zero_grad()
        g_opts[1].zero_grad()
        total_loss.backward()
        g_opts[0].step()
        g_opts[1].step()

        for x, y in [(G, originalG), (C, originalC), (D, originalD)]:
            self.assertTrue(
                c_f.state_dicts_are_equal(x.state_dict(), y.state_dict(), rtol=1e-6)
            )
