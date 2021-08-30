import copy
import unittest

import torch

from pytorch_adapt.hooks import AdaBNHook, validate_hook
from pytorch_adapt.layers import AdaptiveBatchNorm2d
from pytorch_adapt.layers.adaptive_batch_norm import (
    finalize_bn,
    set_bn_layer_to_train,
    set_curr_domain,
)
from pytorch_adapt.utils import common_functions as c_f

from .utils import assertRequiresGrad, get_models_and_data


class Net(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.count = 0
        self.net = torch.nn.Sequential(
            AdaptiveBatchNorm2d(in_size), torch.nn.Conv2d(in_size, out_size, 3)
        )

    def forward(self, x, domain):
        self.count += 1
        set_curr_domain(self, domain[0].int(), AdaptiveBatchNorm2d)
        x = self.net(x)
        return x


class TestAdaBN(unittest.TestCase):
    def test_adabn_hook(self):
        src_imgs = torch.randn(100, 32, 8, 8)
        target_imgs = torch.randn(100, 32, 8, 8)
        src_domain = torch.zeros(100)
        target_domain = torch.ones(100)
        G = Net(32, 16)
        C = Net(16, 10)
        originalG = copy.deepcopy(G)
        originalC = copy.deepcopy(C)

        models = {"G": G, "C": C}
        data = {
            "src_imgs": src_imgs,
            "target_imgs": target_imgs,
            "src_domain": src_domain,
            "target_domain": target_domain,
        }

        h = AdaBNHook()
        model_counts = validate_hook(h, list(data.keys()))
        losses, outputs = h({}, {**models, **data})
        self.assertTrue(
            G.count == C.count == model_counts["G"] == model_counts["C"] == 2
        )
        assertRequiresGrad(self, outputs)
        self.assertTrue(len(losses) == 0)

        originalG.net[0].bns[0](src_imgs)
        originalG.net[0].bns[1](target_imgs)

        for i in range(2):
            self.assertTrue(
                torch.equal(
                    G.net[0].bns[i].running_mean, originalG.net[0].bns[i].running_mean
                )
            )
            self.assertTrue(
                torch.equal(
                    G.net[0].bns[i].running_var, originalG.net[0].bns[i].running_var
                )
            )
