import copy
import unittest

import numpy as np
import torch

from pytorch_adapt.hooks import (
    SymNetsCategoryLossHook,
    SymNetsCHook,
    SymNetsDomainLossHook,
    SymNetsEntropyHook,
    SymNetsGHook,
    SymNetsHook,
    validate_hook,
)
from pytorch_adapt.layers import MultipleModels
from pytorch_adapt.utils import common_functions as c_f

from .utils import Net, assertRequiresGrad, get_opts


def get_model_and_data():
    batch_size = 100
    num_classes = 10
    src_imgs = torch.randn(batch_size, 32)
    target_imgs = torch.randn(batch_size, 32)
    G = Net(32, 16)
    Cs = Net(16, num_classes)
    Ct = Net(16, num_classes)
    C = MultipleModels(Cs, Ct)
    return G, C, src_imgs, target_imgs, batch_size, num_classes


def get_correct_entropy(G, C, target_imgs, num_classes):
    logits = C(G(target_imgs))
    logits = torch.nn.functional.softmax(torch.cat(logits, dim=1), dim=1)
    logits = logits[:, :num_classes] + logits[:, num_classes:]
    return -torch.mean(torch.sum(logits * torch.log(logits), dim=1))


def get_correct_category_loss(G, C, src_imgs, src_labels, num_classes):
    logits = C(G(src_imgs))
    logits = torch.nn.functional.softmax(torch.cat(logits, dim=1), dim=1)
    loss = torch.nn.functional.cross_entropy(logits[:, :num_classes], src_labels)
    loss += torch.nn.functional.cross_entropy(logits[:, num_classes:], src_labels)
    return loss


def get_correct_domain_loss(G, C, src_imgs, target_imgs, num_classes, domain, half_idx):
    imgs = src_imgs if domain == "src" else target_imgs
    logits = C(G(imgs))
    logits = torch.nn.functional.softmax(torch.cat(logits, dim=1), dim=1)
    if half_idx == 0:
        logits = logits[:, :num_classes]
    else:
        logits = logits[:, num_classes:]
    return -torch.mean(torch.log(torch.sum(logits, dim=1)))


class TestSymNets(unittest.TestCase):
    def test_symnets_entropy_hook(self):
        G, C, src_imgs, target_imgs, batch_size, num_classes = get_model_and_data()
        h = SymNetsEntropyHook()

        losses, outputs = h({}, locals())

        self.assertTrue(G.count == C.models[0].count == C.models[1].count == 1)
        assertRequiresGrad(self, outputs)

        self.assertTrue(
            outputs.keys() == {"target_imgs_features", "target_imgs_features_logits"}
        )
        self.assertTrue(losses.keys() == {"symnets_entropy_loss"})
        correct_entropy = get_correct_entropy(G, C, target_imgs, num_classes)
        self.assertTrue(torch.isclose(losses["symnets_entropy_loss"], correct_entropy))

    def test_symnets_domain_loss_hook(self):
        G, C, src_imgs, target_imgs, batch_size, num_classes = get_model_and_data()
        TestG = copy.deepcopy(G)
        TestC = copy.deepcopy(C)
        models = {"G": G, "C": C}
        data = {"src_imgs": src_imgs, "target_imgs": target_imgs}
        outputs = {}
        for domain in ["src", "target"]:
            for half_idx in [0, 1]:
                h = SymNetsDomainLossHook(domain, half_idx)

                losses, new_outputs = h({}, {**models, **data, **outputs})
                outputs.update(new_outputs)
                correct_count = 1 if domain == "src" else 2
                self.assertTrue(
                    G.count == C.models[0].count == C.models[1].count == correct_count
                )
                assertRequiresGrad(self, outputs)

                if half_idx == 0:
                    correct_outputs = {
                        f"{domain}_imgs_features",
                        f"{domain}_imgs_features_logits",
                    }
                    self.assertTrue(new_outputs.keys() == correct_outputs)
                else:
                    self.assertTrue(len(new_outputs.keys()) == 0)
                self.assertTrue(
                    losses.keys() == {f"symnets_{domain}_domain_loss_{half_idx}"}
                )

                correct_loss = get_correct_domain_loss(
                    TestG, TestC, src_imgs, target_imgs, num_classes, domain, half_idx
                )
                self.assertTrue(torch.isclose(losses[h.loss_keys[0]], correct_loss))

    def test_symnets_category_loss_hook(self):
        G, C, src_imgs, target_imgs, batch_size, num_classes = get_model_and_data()
        src_labels = torch.randint(0, num_classes, size=(batch_size,))
        h = SymNetsCategoryLossHook()

        losses, outputs = h({}, locals())

        self.assertTrue(G.count == C.models[0].count == C.models[1].count == 1)
        assertRequiresGrad(self, outputs)

        self.assertTrue(
            outputs.keys() == {"src_imgs_features", "src_imgs_features_logits"}
        )
        self.assertTrue(losses.keys() == {"symnets_category_loss"})

        loss = get_correct_category_loss(G, C, src_imgs, src_labels, num_classes)
        self.assertTrue(torch.isclose(losses["symnets_category_loss"], loss))

    def test_symnets(self):
        G, C, src_imgs, target_imgs, batch_size, num_classes = get_model_and_data()

        originalG = copy.deepcopy(G)
        originalC = copy.deepcopy(C)

        c_opts = get_opts(C)
        g_opts = get_opts(G)

        src_labels = torch.randint(0, num_classes, size=(batch_size,))
        models = {"G": G, "C": C}
        data = {
            "src_imgs": src_imgs,
            "target_imgs": target_imgs,
            "src_labels": src_labels,
        }
        outputs = {}
        h = SymNetsHook(c_opts, g_opts)
        model_counts = validate_hook(h, list(data.keys()))

        losses, outputs = h({}, {**models, **data})
        self.assertTrue(G.count == model_counts["G"] == 2)
        self.assertTrue(
            C.models[0].count == C.models[1].count == model_counts["C"] == 4
        )
        assertRequiresGrad(self, outputs)
        self.assertTrue(
            outputs.keys()
            == {
                "src_imgs_features",
                "target_imgs_features",
                "src_imgs_features_detached",
                "target_imgs_features_detached",
                "src_imgs_features_detached_logits",
                "target_imgs_features_detached_logits",
                "src_imgs_features_logits",
                "target_imgs_features_logits",
            }
        )
        self.assertTrue(
            losses["c_loss"].keys()
            == {
                "c_loss0",
                "c_loss1",
                "c_symnets_src_domain_loss_0",
                "c_symnets_target_domain_loss_1",
                "total",
            }
        )

        self.assertTrue(
            losses["g_loss"].keys()
            == {
                "symnets_entropy_loss",
                "symnets_category_loss",
                "g_symnets_target_domain_loss_0",
                "g_symnets_target_domain_loss_1",
                "total",
            }
        )

        c_opts = get_opts(originalC)
        g_opts = get_opts(originalG)

        src_features = originalG(src_imgs)
        src_logits = originalC(src_features)
        c0_loss = torch.nn.functional.cross_entropy(src_logits[0], src_labels)
        c1_loss = torch.nn.functional.cross_entropy(src_logits[1], src_labels)

        d1_loss = get_correct_domain_loss(
            originalG, originalC, src_imgs, target_imgs, num_classes, "src", 0
        )
        d2_loss = get_correct_domain_loss(
            originalG, originalC, src_imgs, target_imgs, num_classes, "target", 1
        )

        total_loss = (c0_loss + c1_loss + d1_loss + d2_loss) / 4
        self.assertTrue(
            np.isclose(total_loss.item(), losses["c_loss"]["total"], rtol=1e-4)
        )

        c_opts[0].zero_grad()
        total_loss.backward()
        c_opts[0].step()

        d1_loss = get_correct_domain_loss(
            originalG, originalC, src_imgs, target_imgs, num_classes, "target", 1
        )

        d2_loss = get_correct_domain_loss(
            originalG, originalC, src_imgs, target_imgs, num_classes, "target", 0
        )

        correct_entropy = get_correct_entropy(
            originalG, originalC, target_imgs, num_classes
        )
        correct_category_loss = get_correct_category_loss(
            originalG, originalC, src_imgs, src_labels, num_classes
        )

        total_loss = (correct_entropy + correct_category_loss + d1_loss + d2_loss) / 4
        self.assertTrue(
            np.isclose(total_loss.item(), losses["g_loss"]["total"], rtol=1e-4)
        )
        g_opts[0].zero_grad()
        total_loss.backward()
        g_opts[0].step()

        for x, y in [(G, originalG), (C, originalC)]:
            self.assertTrue(
                c_f.state_dicts_are_equal(x.state_dict(), y.state_dict(), rtol=1e-6)
            )
