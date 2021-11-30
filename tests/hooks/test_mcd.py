import copy
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_adapt.hooks import MCDHook, MCDLossHook, MultipleCLossHook, validate_hook
from pytorch_adapt.layers import (
    MCDLoss,
    MultipleModels,
    SlicedWasserstein,
    StochasticLinear,
)
from pytorch_adapt.utils import common_functions as c_f

from .utils import Net, assertRequiresGrad, get_models_and_data, get_opts


def get_disc_loss(loss_fn, target_logits):
    if loss_fn is None:
        return torch.mean(
            torch.abs(
                F.softmax(target_logits[0], dim=1) - F.softmax(target_logits[1], dim=1)
            )
        )
    return loss_fn(*target_logits)


class TestMCD(unittest.TestCase):
    def test_multiple_c_loss_hook(self):
        for detach in [False, True]:
            h = MultipleCLossHook(num_c=3, detach_features=detach)
            batch_size = 32
            emb_size = 128
            num_classes = 10
            if not detach:
                out_keys = ["src_imgs_features_logits"]
                logits_key = out_keys[0]
            else:
                out_keys = [
                    "src_imgs_features_detached",
                    "src_imgs_features_detached_logits",
                ]
                logits_key = out_keys[1]

            C0 = Net(emb_size, num_classes)
            C1 = Net(emb_size, num_classes)
            C2 = Net(emb_size, num_classes)
            C = MultipleModels(C0, C1, C2)

            src_imgs_features = torch.randn(batch_size, emb_size)
            src_labels = torch.randint(0, num_classes, size=(batch_size,))
            loss, outputs = h({}, locals())
            assertRequiresGrad(self, outputs)

            self.assertTrue(
                outputs[logits_key][i].requires_grad == True for i in range(3)
            )
            self.assertTrue(all(x.count == 1 for x in [C0, C1, C2]))
            self.assertTrue(outputs.keys() == {*out_keys})
            self.assertTrue(loss.keys() == {"c_loss0", "c_loss1", "c_loss2"})

            loss_fn = torch.nn.CrossEntropyLoss()
            losses = [loss_fn(x, src_labels) for x in outputs[logits_key]]
            self.assertTrue(
                all(
                    torch.isclose(losses[i], torch.mean(loss[f"c_loss{i}"]))
                    for i in range(len(losses))
                )
            )

    def test_mcd_loss_hook(self):
        for detach in [False, True]:
            h = MCDLossHook(detach_features=detach)
            batch_size = 32
            emb_size = 128
            num_classes = 10
            if not detach:
                out_keys = ["target_imgs_features_logits"]
                logits_key = out_keys[0]
            else:
                out_keys = [
                    "target_imgs_features_detached",
                    "target_imgs_features_detached_logits",
                ]
                logits_key = out_keys[1]

            C0 = Net(emb_size, num_classes)
            C1 = Net(emb_size, num_classes)
            C = MultipleModels(C0, C1)

            target_imgs_features = torch.randn(batch_size, emb_size)
            loss, outputs = h({}, locals())
            assertRequiresGrad(self, outputs)
            self.assertTrue(
                outputs[logits_key][i].requires_grad == True for i in range(3)
            )
            self.assertTrue(all(x.count == 1 for x in [C0, C1]))
            self.assertTrue(outputs.keys() == {*out_keys})
            self.assertTrue(loss.keys() == {"discrepancy_loss"})

            loss_fn = MCDLoss()
            correct_loss = torch.mean(
                torch.abs(
                    torch.nn.functional.softmax(outputs[logits_key][0], dim=1)
                    - torch.nn.functional.softmax(outputs[logits_key][1], dim=1)
                )
            )
            self.assertTrue(
                loss_fn(outputs[logits_key][0], outputs[logits_key][1]) == correct_loss
            )
            self.assertTrue(correct_loss == loss["discrepancy_loss"])

    def test_mcd_hook(self):
        seed = 2021
        for stochastic_linear in [False, True]:
            for loss_fn in [None, SlicedWasserstein()]:
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

                C0 = Net(16, 10)
                C1 = Net(16, 10)
                if stochastic_linear:
                    C0.net = StochasticLinear(16, 10)
                    C1.net = StochasticLinear(16, 10)

                C = MultipleModels(C0, C1)

                originalG = copy.deepcopy(G)
                originalC = copy.deepcopy(C)

                opts = get_opts(G, C)

                repeat = 5

                h = MCDHook(
                    [opts[0]], [opts[1]], repeat=repeat, discrepancy_loss_fn=loss_fn
                )
                models = {
                    "G": G,
                    "C": C,
                }
                data = {
                    "src_imgs": src_imgs,
                    "target_imgs": target_imgs,
                    "src_labels": src_labels,
                }
                model_counts = validate_hook(h, list(data.keys()))

                torch.manual_seed(seed)
                losses, outputs = h({}, {**models, **data})
                assertRequiresGrad(self, outputs)
                self.assertTrue(
                    outputs.keys()
                    == {
                        "src_imgs_features",
                        "src_imgs_features_logits",
                        "target_imgs_features",
                        "target_imgs_features_logits",
                        "src_imgs_features_detached",
                        "src_imgs_features_detached_logits",
                        "target_imgs_features_detached",
                        "target_imgs_features_detached_logits",
                    }
                )

                self.assertTrue(
                    losses["x_loss"].keys() == {"c_loss0", "c_loss1", "total"}
                )
                self.assertTrue(
                    losses["y_loss"].keys()
                    == {"c_loss0", "c_loss1", "discrepancy_loss", "total"}
                )
                self.assertTrue(
                    losses["z_loss"].keys() == {"discrepancy_loss", "total"}
                )
                self.assertTrue(G.count == model_counts["G"])
                self.assertTrue(C.models[0].count == model_counts["C"])
                self.assertTrue(C.models[1].count == model_counts["C"])

                opts = get_opts(originalG, originalC)

                torch.manual_seed(seed)
                ## x ##
                src_features = originalG(src_imgs)
                src_logits = originalC(src_features)
                c_loss0 = F.cross_entropy(src_logits[0], src_labels)
                c_loss1 = F.cross_entropy(src_logits[1], src_labels)
                total_loss = (c_loss0 + c_loss1) / 2
                correct_losses = [c_loss0, c_loss1, total_loss]
                computed_losses = [
                    losses["x_loss"][k] for k in ["c_loss0", "c_loss1", "total"]
                ]
                self.assertTrue(
                    all(
                        np.isclose(x.item(), y)
                        for x, y in zip(correct_losses, computed_losses)
                    )
                )

                [x.zero_grad() for x in opts]
                total_loss.backward()
                [x.step() for x in opts]

                ## y ##
                src_features = originalG(src_imgs)
                src_logits = originalC(src_features.detach())
                c_loss0 = F.cross_entropy(src_logits[0], src_labels)
                c_loss1 = F.cross_entropy(src_logits[1], src_labels)
                target_features = originalG(target_imgs)
                target_logits = originalC(target_features.detach())
                disc_loss = -get_disc_loss(loss_fn, target_logits)
                total_loss = (c_loss0 + c_loss1 + disc_loss) / 3
                correct_losses = [c_loss0, c_loss1, disc_loss, total_loss]
                computed_losses = [
                    losses["y_loss"][k]
                    for k in ["c_loss0", "c_loss1", "discrepancy_loss", "total"]
                ]
                self.assertTrue(
                    all(
                        np.isclose(x.item(), y)
                        for x, y in zip(correct_losses, computed_losses)
                    )
                )

                opts[1].zero_grad()
                total_loss.backward()
                opts[1].step()

                ## z ##
                for _ in range(repeat):
                    target_features = originalG(target_imgs)
                    target_logits = originalC(target_features)
                    disc_loss = get_disc_loss(loss_fn, target_logits)
                    total_loss = disc_loss
                    correct_losses = [disc_loss, total_loss]
                    computed_losses = [
                        losses["z_loss"][k] for k in ["discrepancy_loss", "total"]
                    ]

                    opts[0].zero_grad()
                    total_loss.backward()
                    opts[0].step()

                self.assertTrue(
                    all(x == y for x, y in zip(correct_losses, computed_losses))
                )

                for x, y in [(G, originalG), (C, originalC)]:
                    self.assertTrue(
                        c_f.state_dicts_are_equal(
                            x.state_dict(), y.state_dict(), rtol=1e-6
                        )
                    )
