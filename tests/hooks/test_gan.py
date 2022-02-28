import copy
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_adapt.hooks import (
    AFNHook,
    BNMHook,
    BSPHook,
    DomainConfusionHook,
    GANEHook,
    GANHook,
    MCCHook,
    VADAHook,
    validate_hook,
)
from pytorch_adapt.layers import EntropyLoss, UniformDistributionLoss, VATLoss
from pytorch_adapt.utils import common_functions as c_f

from .utils import (
    assertRequiresGrad,
    get_entropy_weights,
    get_models_and_data,
    get_opts,
    post_g_hook_update_keys,
    post_g_hook_update_total_loss,
)


class TestGAN(unittest.TestCase):
    def test_gan(self):
        seed = 1234
        for post_g in [None, BSPHook(), BNMHook(), MCCHook(), AFNHook()]:
            for hook_cls in [GANHook, DomainConfusionHook, VADAHook, GANEHook]:
                for detach_reducer in [False, True]:
                    if detach_reducer and hook_cls is not GANEHook:
                        continue
                    if hook_cls is GANEHook and post_g is not None:
                        continue
                    d_out = 2 if hook_cls == DomainConfusionHook else 1
                    (
                        G,
                        C,
                        D,
                        src_imgs,
                        src_labels,
                        target_imgs,
                        src_domain,
                        target_domain,
                    ) = get_models_and_data(d_out=d_out)
                    models = {"G": G, "D": D, "C": C}
                    if hook_cls == VADAHook:
                        combined_model = torch.nn.Sequential(G, C)
                        models["combined_model"] = combined_model

                    originalG = copy.deepcopy(G)
                    originalC = copy.deepcopy(C)
                    originalD = copy.deepcopy(D)

                    d_opts = get_opts(D)
                    g_opts = get_opts(G, C)
                    post_g_ = [post_g] if post_g is not None else post_g
                    hook_kwargs = {
                        "d_opts": d_opts,
                        "g_opts": g_opts,
                        "post_g": post_g_,
                    }
                    if hook_cls is GANEHook:
                        hook_kwargs["detach_entropy_reducer"] = detach_reducer
                    h = hook_cls(**hook_kwargs)

                    data = {
                        "src_imgs": src_imgs,
                        "target_imgs": target_imgs,
                        "src_labels": src_labels,
                        "src_domain": src_domain,
                        "target_domain": target_domain,
                    }
                    model_counts = validate_hook(h, list(data.keys()))

                    torch.manual_seed(seed)
                    outputs, losses = h({**models, **data})
                    assertRequiresGrad(self, outputs)

                    output_keys = {
                        "src_imgs_features",
                        "src_imgs_features_dlogits",
                        "target_imgs_features",
                        "target_imgs_features_dlogits",
                        "src_imgs_features_detached",
                        "src_imgs_features_detached_dlogits",
                        "target_imgs_features_detached",
                        "target_imgs_features_detached_dlogits",
                        "src_imgs_features_logits",
                    }

                    g_loss_keys = {
                        "g_src_domain_loss",
                        "g_target_domain_loss",
                        "c_loss",
                        "total",
                    }

                    if hook_cls == GANEHook:
                        output_keys.update(
                            {
                                "src_imgs_features_logits_detached",
                                "target_imgs_features_logits_detached",
                            }
                        )
                        if not detach_reducer:
                            output_keys.update({"target_imgs_features_logits"})

                    if hook_cls == VADAHook:
                        output_keys.update({"target_imgs_features_logits"})
                        g_loss_keys.update(
                            {"entropy_loss", "src_vat_loss", "target_vat_loss"}
                        )
                    post_g_hook_update_keys(post_g, g_loss_keys, output_keys)

                    self.assertTrue(outputs.keys() == output_keys)

                    self.assertTrue(losses["g_loss"].keys() == g_loss_keys)
                    self.assertTrue(
                        losses["d_loss"].keys()
                        == {"d_src_domain_loss", "d_target_domain_loss", "total"}
                    )

                    if hook_cls is GANEHook:
                        correct_c_count = 3 if detach_reducer else 4
                    elif isinstance(post_g, (BNMHook, MCCHook)) or hook_cls is VADAHook:
                        correct_c_count = 2
                    else:
                        correct_c_count = 1
                    self.assertTrue(model_counts["C"] == correct_c_count)
                    self.assertTrue(model_counts["G"] == 2)
                    self.assertTrue(model_counts["D"] == 4)
                    if hook_cls == VADAHook:
                        # inside VATLoss, 2*2 domains
                        model_counts["C"] += 4
                        model_counts["G"] += 4
                        # also need target logits, but they are computed during VATLoss

                    self.assertTrue(C.count == model_counts["C"])
                    self.assertTrue(G.count == model_counts["G"])
                    self.assertTrue(D.count == model_counts["D"])

                    d_opts = get_opts(originalD)
                    g_opts = get_opts(originalG, originalC)

                    torch.manual_seed(seed)
                    src_features = originalG(src_imgs)
                    target_features = originalG(target_imgs)
                    src_logits = originalC(src_features)
                    src_dlogits = originalD(src_features.detach())
                    target_dlogits = originalD(target_features.detach())

                    domain_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
                    if hook_cls == DomainConfusionHook:
                        src_domain = src_domain.long()
                        target_domain = target_domain.long()
                        domain_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

                    d_src_domain_loss = domain_loss_fn(src_dlogits, src_domain)
                    d_target_domain_loss = domain_loss_fn(target_dlogits, target_domain)
                    if hook_cls is GANEHook:
                        target_logits = originalC(target_features)
                        c_logits = torch.cat(
                            [src_logits.detach(), target_logits.detach()], dim=0
                        )
                        (
                            src_entropy_weights,
                            target_entropy_weights,
                        ) = get_entropy_weights(
                            c_logits, len(target_logits), detach_reducer
                        )
                        d_src_domain_loss = torch.mean(
                            d_src_domain_loss * src_entropy_weights
                        )
                        d_target_domain_loss = torch.mean(
                            d_target_domain_loss * target_entropy_weights
                        )
                    else:
                        d_src_domain_loss = torch.mean(d_src_domain_loss)
                        d_target_domain_loss = torch.mean(d_target_domain_loss)

                    self.assertTrue(
                        d_src_domain_loss == losses["d_loss"]["d_src_domain_loss"]
                    )
                    self.assertTrue(
                        d_target_domain_loss == losses["d_loss"]["d_target_domain_loss"]
                    )
                    total_loss = (d_src_domain_loss + d_target_domain_loss) / 2
                    self.assertTrue(total_loss == losses["d_loss"]["total"])
                    d_opts[0].zero_grad()
                    total_loss.backward()
                    d_opts[0].step()

                    src_dlogits = originalD(src_features)
                    target_dlogits = originalD(target_features)

                    domain_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
                    if hook_cls == DomainConfusionHook:
                        domain_loss_fn = UniformDistributionLoss()
                    g_src_domain_loss = domain_loss_fn(src_dlogits, target_domain)
                    g_target_domain_loss = domain_loss_fn(target_dlogits, src_domain)

                    if hook_cls is GANEHook:
                        c_logits = torch.cat([src_logits, target_logits], dim=0)
                        (
                            src_entropy_weights,
                            target_entropy_weights,
                        ) = get_entropy_weights(
                            c_logits, len(target_logits), detach_reducer
                        )
                        g_src_domain_loss = torch.mean(
                            g_src_domain_loss * src_entropy_weights
                        )
                        g_target_domain_loss = torch.mean(
                            g_target_domain_loss * target_entropy_weights
                        )
                    else:
                        g_src_domain_loss = torch.mean(g_src_domain_loss)
                        g_target_domain_loss = torch.mean(g_target_domain_loss)

                    self.assertTrue(
                        g_src_domain_loss == losses["g_loss"]["g_src_domain_loss"]
                    )
                    self.assertTrue(
                        g_target_domain_loss == losses["g_loss"]["g_target_domain_loss"]
                    )

                    c_loss = F.cross_entropy(src_logits, src_labels)
                    self.assertTrue(
                        np.isclose(c_loss.item(), losses["g_loss"]["c_loss"])
                    )

                    total_loss = [g_src_domain_loss, g_target_domain_loss, c_loss]
                    target_logits = originalC(target_features)
                    if hook_cls == VADAHook:
                        entropy_loss = EntropyLoss()(target_logits)
                        self.assertTrue(
                            entropy_loss == losses["g_loss"]["entropy_loss"]
                        )
                        combined_model = torch.nn.Sequential(originalG, originalC)
                        vat_loss_fn = VATLoss()
                        src_vat_loss = vat_loss_fn(src_imgs, src_logits, combined_model)
                        self.assertTrue(
                            src_vat_loss == losses["g_loss"]["src_vat_loss"]
                        )
                        target_vat_loss = vat_loss_fn(
                            target_imgs, target_logits, combined_model
                        )
                        self.assertTrue(
                            target_vat_loss == losses["g_loss"]["target_vat_loss"]
                        )
                        total_loss.extend([entropy_loss, src_vat_loss, target_vat_loss])

                    post_g_hook_update_total_loss(
                        post_g, total_loss, src_features, target_features, target_logits
                    )

                    total_loss = sum(total_loss) / len(total_loss)

                    self.assertTrue(
                        np.isclose(total_loss.item(), losses["g_loss"]["total"])
                    )
                    g_opts[0].zero_grad()
                    g_opts[1].zero_grad()
                    total_loss.backward()
                    g_opts[0].step()
                    g_opts[1].step()

                    for x, y in [(G, originalG), (C, originalC), (D, originalD)]:
                        self.assertTrue(
                            c_f.state_dicts_are_equal(
                                x.state_dict(), y.state_dict(), rtol=1e-6
                            )
                        )
