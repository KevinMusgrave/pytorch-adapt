import copy
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_adapt.adapters import ADDA
from pytorch_adapt.containers import Models, Optimizers
from pytorch_adapt.hooks import ADDAHook, BSPHook, validate_hook
from pytorch_adapt.utils import common_functions as c_f

from .utils import (
    Net,
    assert_equal_models,
    assertRequiresGrad,
    get_opt_tuple,
    get_opts,
    post_g_hook_update_keys,
    post_g_hook_update_total_loss,
)


def test_equivalent_adapter(G, D, data, post_g, threshold):
    models = Models(
        {"G": copy.deepcopy(G), "D": copy.deepcopy(D), "C": torch.nn.Identity()}
    )
    optimizers = Optimizers(get_opt_tuple())
    adapter = ADDA(
        models, optimizers, hook_kwargs={"post_g": post_g, "threshold": threshold}
    )
    adapter.training_step(data)
    return models


def get_models_and_data():
    src_domain = torch.randint(0, 2, size=(100,)).float()
    target_domain = torch.randint(0, 2, size=(100,)).float()
    src_imgs = torch.randn(100, 32)
    target_imgs = torch.randn(100, 32)
    G = Net(32, 16, with_batch_norm=True)
    D = Net(16, 1, with_batch_norm=True)
    return G, D, src_imgs, target_imgs, src_domain, target_domain


class TestADDA(unittest.TestCase):
    def test_adda(self):
        torch.manual_seed(922)
        for post_g in [None, [BSPHook(domains=["target"])]]:
            for threshold in np.linspace(0, 1, 10):
                (
                    G,
                    D,
                    src_imgs,
                    target_imgs,
                    src_domain,
                    target_domain,
                ) = get_models_and_data()
                T = copy.deepcopy(G)
                originalG = copy.deepcopy(G)
                originalD = copy.deepcopy(D)
                originalT = copy.deepcopy(T)
                d_opts = get_opts(D)
                g_opts = get_opts(T)
                h = ADDAHook(
                    d_opts=d_opts, g_opts=g_opts, threshold=threshold, post_g=post_g
                )
                models = {"G": G, "D": D, "T": T}
                data = {
                    "src_imgs": src_imgs,
                    "target_imgs": target_imgs,
                    "src_domain": src_domain,
                    "target_domain": target_domain,
                }
                model_counts = validate_hook(h, list(data.keys()))
                outputs, losses = h({**models, **data})
                assertRequiresGrad(self, outputs)
                output_keys = {
                    "src_imgs_features_detached",
                    "src_imgs_features_detached_dlogits",
                    "target_imgs_features_detached",
                    "target_imgs_features_detached_dlogits",
                    "target_imgs_features",
                    "target_imgs_features_dlogits",
                }

                g_loss_keys = {"g_target_domain_loss", "total"}
                post_g_hook_update_keys(post_g, g_loss_keys, output_keys)
                self.assertTrue(outputs.keys() == output_keys)
                self.assertTrue(
                    losses["d_loss"].keys()
                    == {"d_src_domain_loss", "d_target_domain_loss", "total"}
                )
                self.assertTrue(losses["g_loss"].keys() == g_loss_keys)

                adapter_models = test_equivalent_adapter(
                    originalG, originalD, data, post_g, threshold
                )

                d_opts = get_opts(originalD)[0]
                g_opts = get_opts(originalT)[0]
                originalG.eval()
                with torch.no_grad():
                    src_features = originalG(src_imgs)

                target_features = originalT(target_imgs)

                src_features_dlogits = originalD(src_features)
                target_features_dlogits = originalD(target_features.detach())

                d_src_loss = F.binary_cross_entropy_with_logits(
                    src_features_dlogits, src_domain
                )
                d_target_loss = F.binary_cross_entropy_with_logits(
                    target_features_dlogits, target_domain
                )
                self.assertTrue(d_src_loss == losses["d_loss"]["d_src_domain_loss"])
                self.assertTrue(
                    d_target_loss == losses["d_loss"]["d_target_domain_loss"]
                )
                total_loss = (d_src_loss + d_target_loss) / 2
                self.assertTrue(total_loss == losses["d_loss"]["total"])
                d_opts.zero_grad()
                total_loss.backward()
                d_opts.step()

                with torch.no_grad():
                    features = torch.cat([src_features, target_features], dim=0)
                    logits = torch.cat(
                        [src_features_dlogits, target_features_dlogits], dim=0
                    )
                    preds = torch.round(torch.sigmoid(logits))
                    labels = torch.cat([src_domain, target_domain], dim=0)
                    accuracy = torch.mean((preds == labels).float()).item()
                d_count = 2  # detached src & target
                total_loss = []
                if accuracy >= threshold:
                    d_count += 1  # undetached target
                    target_features_dlogits = originalD(target_features)
                    total_loss.append(
                        F.binary_cross_entropy_with_logits(
                            target_features_dlogits, src_domain
                        )
                    )
                else:
                    total_loss.append(c_f.zero_loss())

                post_g_hook_update_total_loss(
                    post_g, total_loss, None, target_features, None
                )

                total_loss = sum(total_loss) / len(total_loss)
                self.assertAlmostEqual(
                    total_loss.item(), losses["g_loss"]["total"], places=6
                )
                g_opts.zero_grad()
                total_loss.backward()
                g_opts.step()

                self.assertTrue(
                    G.count == model_counts["G"] == T.count == model_counts["T"] == 1
                )
                # can't use model_counts for conditional part
                self.assertTrue(D.count == d_count)

                assert_equal_models(
                    self, (G, adapter_models["G"], originalG), rtol=1e-3
                )
                assert_equal_models(
                    self, (T, adapter_models["T"], originalT), rtol=1e-3
                )
                assert_equal_models(
                    self, (D, adapter_models["D"], originalD), rtol=1e-3
                )
