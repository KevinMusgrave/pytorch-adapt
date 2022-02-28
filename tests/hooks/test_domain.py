import unittest

import torch

from pytorch_adapt.hooks import DomainLossHook, FeaturesForDomainLossHook

from .utils import Net, assertRequiresGrad


class TestDomain(unittest.TestCase):
    def test_domain_loss_hook(self):
        for detach_features in [False, True]:
            for use_logits in [True, False]:
                if use_logits:
                    f_hook = FeaturesForDomainLossHook(
                        detach=detach_features, use_logits=use_logits
                    )
                else:
                    f_hook = None
                for reverse_labels in [True, False]:
                    h = DomainLossHook(
                        f_hook=f_hook,
                        detach_features=detach_features,
                        reverse_labels=reverse_labels,
                    )
                    src_domain = torch.zeros(10)
                    target_domain = torch.ones(10)
                    src_imgs = torch.randn(10, 32)
                    target_imgs = torch.randn(10, 32)
                    G = Net(32, 16)
                    C = Net(16, 16)
                    D = Net(16, 1)

                    output_keys = ["src_imgs_features", "target_imgs_features"]
                    if use_logits:
                        output_keys += [
                            "src_imgs_features_logits",
                            "target_imgs_features_logits",
                            "src_imgs_features_logits_dlogits",
                            "target_imgs_features_logits_dlogits",
                        ]
                    else:
                        output_keys += [
                            "src_imgs_features_dlogits",
                            "target_imgs_features_dlogits",
                        ]

                    output_keys = [x.replace("_detached", "") for x in output_keys]
                    if detach_features:
                        original_str, new_str = "features", "features_detached"
                        if use_logits:
                            original_str, new_str = "_logits", "_logits_detached"
                        output_keys = [
                            x.replace(original_str, new_str) for x in output_keys
                        ]
                        output_keys = [
                            f"{x}_detached"
                            if not x.endswith("dlogits") and not x.endswith("detached")
                            else x
                            for x in output_keys
                        ]

                    outputs, losses = h(locals())
                    self.assertTrue(all(x.count == 2 for x in [G, D]))
                    self.assertTrue(C.count == (2 if use_logits else 0))
                    self.assertTrue(outputs.keys() == {*output_keys})
                    assertRequiresGrad(self, outputs)

                    if reverse_labels:
                        src_domain, target_domain = target_domain, src_domain
                    loss_fn = torch.nn.BCEWithLogitsLoss()
                    correct_loss = {}
                    for domain, imgs, labels in [
                        ("src", src_imgs, src_domain),
                        ("target", target_imgs, target_domain),
                    ]:
                        features = G(imgs)
                        if use_logits:
                            features = C(features)
                        dlogits = D(features)
                        correct_loss[f"{domain}_domain_loss"] = loss_fn(dlogits, labels)

                    self.assertTrue(
                        all(
                            correct_loss[f"{domain}_domain_loss"]
                            == torch.mean(losses[f"{domain}_domain_loss"])
                            for domain in ["src", "target"]
                        )
                    )
