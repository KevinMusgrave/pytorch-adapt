import copy
import unittest

import torch

from pytorch_adapt.layers import PopulationBatchNorm2d
from pytorch_adapt.layers.adaptive_batch_norm import finalize_bn

from .. import TEST_DEVICE


class TestModel(torch.nn.Module):
    def __init__(self, first_layer, batchnorm_layer):
        super().__init__()
        self.net = torch.nn.Sequential(
            first_layer,
            batchnorm_layer(32),
        )

    def forward(self, x):
        return self.net(x)


def check(cls, model1, model2, should_be_close):
    for attr in ["final_mean", "final_var"]:
        x = getattr(model1.net[1], attr)
        y = getattr(model2.net[1], attr)
        all_close = torch.allclose(x, y, rtol=1e-1)
        if should_be_close:
            cls.assertTrue(all_close)
        else:
            cls.assertTrue(not all_close)


class TestPopulationBatchNorm2d(unittest.TestCase):
    def test_population_batch_norm_2d(self):
        first_layer = torch.nn.Conv2d(3, 32, 5, 1)
        model1 = TestModel(first_layer, PopulationBatchNorm2d).to(TEST_DEVICE)
        model2 = copy.deepcopy(model1)
        imgs = torch.randn(1000, 3, 32, 32, device=TEST_DEVICE)

        for s, e in [(0, 10), (10, 1000)]:
            for i in range(s, e):
                model1(imgs[i].unsqueeze(0))
        finalize_bn(model1, PopulationBatchNorm2d)

        model2(imgs)

        check(self, model1, model2, False)

        finalize_bn(model2, PopulationBatchNorm2d)

        check(self, model1, model2, True)
