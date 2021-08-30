import copy
import unittest

import numpy as np
import torch

from pytorch_adapt.hooks import DomainConfusionHook, validate_hook
from pytorch_adapt.utils import common_functions as c_f

from .utils import Net, assertRequiresGrad, get_models_and_data


class TestDomainConfusion(unittest.TestCase):
    def test_domain_confusion_hook(self):
        (
            G,
            C,
            D,
            src_imgs,
            src_labels,
            target_imgs,
            src_domain,
            target_domain,
        ) = get_models_and_data(d_out=2)
        d_opts = [torch.optim.SGD(D.parameters(), lr=0.1)]
        g_opts = [
            torch.optim.SGD(G.parameters(), lr=0.1),
            torch.optim.SGD(C.parameters(), lr=0.1),
        ]
        h = DomainConfusionHook(d_opts=d_opts, g_opts=g_opts)
        models = {"G": G, "D": D, "C": C}
        data = {
            "src_imgs": src_imgs,
            "target_imgs": target_imgs,
            "src_labels": src_labels,
            "src_domain": src_domain,
            "target_domain": target_domain,
        }
        model_counts = validate_hook(h, list(data.keys()))

        for _ in range(10):
            losses, outputs = h({}, {**models, **data})
            assertRequiresGrad(self, outputs)
            self.assertTrue(
                outputs.keys()
                == {
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
            )
            self.assertTrue(
                losses["g_loss"].keys()
                == {"g_src_domain_loss", "g_target_domain_loss", "c_loss", "total"}
            )
            self.assertTrue(
                losses["d_loss"].keys()
                == {"d_src_domain_loss", "d_target_domain_loss", "total"}
            )

        self.assertTrue(C.count == 10)
        self.assertTrue(G.count == 20)
        self.assertTrue(D.count == 40)
