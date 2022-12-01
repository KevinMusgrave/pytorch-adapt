import copy
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_adapt.adapters import Aligner
from pytorch_adapt.containers import Models, Optimizers
from pytorch_adapt.hooks import AlignerPlusCHook, JointAlignerHook, validate_hook
from pytorch_adapt.layers import CORALLoss, MMDLoss

from .utils import (
    assert_equal_models,
    assertRequiresGrad,
    get_models_and_data,
    get_opt_tuple,
    get_opts,
)


def test_equivalent_adapter(G, C, data, aligner_hook, loss_fn):
    models = Models({"G": copy.deepcopy(G), "C": copy.deepcopy(C)})
    optimizers = Optimizers(get_opt_tuple())
    adapter = Aligner(
        models,
        optimizers,
        hook_kwargs={"aligner_hook": aligner_hook, "loss_fn": loss_fn},
    )
    adapter.training_step(data)
    return models


class TestAligners(unittest.TestCase):
    def test_aligner_plus_classifier_hook(self):
        torch.manual_seed(10)
        for loss_fn in [MMDLoss, CORALLoss]:
            for joint in [False, True]:
                if loss_fn == CORALLoss and joint:
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

                opts = get_opts(G, C)
                aligner_hook = JointAlignerHook() if joint else None
                h = AlignerPlusCHook(opts, aligner_hook=aligner_hook, loss_fn=loss_fn())
                models = {"G": G, "D": D, "C": C}
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
                    }
                )

                loss_keys = {
                    "src_c_loss",
                    "total",
                }

                if joint:
                    loss_keys.update({"joint_confusion_loss"})
                else:
                    loss_keys.update(
                        {
                            "features_confusion_loss",
                            "logits_confusion_loss",
                        }
                    )

                self.assertTrue(losses["total_loss"].keys() == loss_keys)
                self.assertTrue(
                    G.count == model_counts["G"] == C.count == model_counts["C"] == 2
                )

                adapter_models = test_equivalent_adapter(
                    originalG, originalC, data, aligner_hook, loss_fn()
                )

                opts = get_opts(originalG, originalC)

                src_features = originalG(src_imgs)
                target_features = originalG(target_imgs)
                src_logits = originalC(src_features)
                target_logits = originalC(target_features)

                # logits shouldn't have softmaxed applied
                self.assertTrue(
                    torch.equal(src_logits, outputs["src_imgs_features_logits"])
                )
                self.assertTrue(
                    torch.equal(target_logits, outputs["target_imgs_features_logits"])
                )

                c_loss = F.cross_entropy(src_logits, src_labels)
                if joint:
                    f_loss = loss_fn()(
                        [F.softmax(src_logits, dim=1), src_features],
                        [F.softmax(target_logits, dim=1), target_features],
                    )
                    total_loss = (f_loss + c_loss) / 2
                    correct_losses = [f_loss, c_loss, total_loss]
                else:
                    f_loss = loss_fn()(src_features, target_features)
                    l_loss = loss_fn()(
                        F.softmax(src_logits, dim=1), F.softmax(target_logits, dim=1)
                    )

                    total_loss = (f_loss + l_loss + c_loss) / 3
                    correct_losses = [f_loss, l_loss, c_loss, total_loss]

                computed_losses = [
                    losses["total_loss"][k] for k in sorted(list(loss_keys))
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

                assert_equal_models(
                    self, (G, adapter_models["G"], originalG), rtol=1e-6
                )
                assert_equal_models(
                    self, (C, adapter_models["C"], originalC), rtol=1e-6
                )
