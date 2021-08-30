import copy
import unittest

import torch

from pytorch_adapt.layers import AdaptiveBatchNorm2d, PopulationBatchNorm2d
from pytorch_adapt.layers.adaptive_batch_norm import (
    convert_bn_to_adabn,
    finalize_bn,
    set_curr_domain,
)

from .. import TEST_DEVICE


class TestModel(torch.nn.Module):
    def __init__(self, batchnorm_layer, nested_batchnorm_layer):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5, 1),
            batchnorm_layer,
            torch.nn.Conv2d(32, 3, 5, 1),
            nested_batchnorm_layer,
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(1728, 10),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x, domain):
        set_curr_domain(self, domain, AdaptiveBatchNorm2d)
        return self.net(x)


def optimize(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(10):
        for domain in [0, 1]:
            imgs = torch.randn(64, 3, 32, 32, device=TEST_DEVICE)
            labels = torch.randint(low=0, high=10, size=(64,), device=TEST_DEVICE)
            logits = model(imgs, domain)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()


def check(cls, model, affine_domain):
    for attr in ["running_mean", "running_var", "weight", "bias"]:
        for B in [model.net[1], model.net[3][0]]:
            x = getattr(B.bns[0], attr)
            y = getattr(B.bns[1], attr)
            if attr in ["weight", "bias"] and affine_domain is not None:
                cls.assertTrue(torch.allclose(x, y))
            else:
                cls.assertTrue(not torch.allclose(x, y))


def check2(cls, model, bn_clones, should_be_equal):
    for attr1, attr2 in [("running_mean", "final_mean"), ("running_var", "final_var")]:
        for i, B in enumerate([model.net[1], model.net[3][0]]):
            x = getattr(B.bns[0], attr1).to(TEST_DEVICE)
            y = getattr(B.bns[1], attr2).to(TEST_DEVICE)
            z = getattr(bn_clones[i], attr1).to(TEST_DEVICE)
            if should_be_equal:
                cls.assertTrue(torch.equal(y, z))
            else:
                cls.assertTrue(not torch.equal(y, z))
            cls.assertTrue(not torch.equal(x, y))


def random_init(bn, size):
    bn.running_mean = torch.randn(size)
    bn.running_var = torch.abs(torch.randn(size))
    return bn


class TestAdaptiveBatchNorm2d(unittest.TestCase):
    def test_adaptive_batch_norm_2d(self):
        for affine_domain in [None, 0, 1]:
            batchnorm_layer = AdaptiveBatchNorm2d(
                num_features=32, num_domains=2, affine_domain=affine_domain
            )
            nested_batchnorm_layer = torch.nn.Sequential(
                AdaptiveBatchNorm2d(
                    num_features=3, num_domains=2, affine_domain=affine_domain
                )
            )
            model = TestModel(batchnorm_layer, nested_batchnorm_layer).to(TEST_DEVICE)
            optimize(model)
            check(self, model, affine_domain)


class TestConversionToAdaBN(unittest.TestCase):
    def test_conversion1(self):
        bn1 = random_init(torch.nn.BatchNorm2d(32), 32)
        bn1_clone = copy.deepcopy(bn1)
        bn2 = random_init(torch.nn.BatchNorm2d(3), 3)
        bn2_clone = copy.deepcopy(bn2)
        nested_batchnorm_layer = torch.nn.Sequential(bn2)
        model = TestModel(bn1, nested_batchnorm_layer).to(TEST_DEVICE)
        convert_bn_to_adabn(model, affine_domain=0, bn_type=PopulationBatchNorm2d)
        model.to(TEST_DEVICE)
        optimize(model)
        check2(self, model, [bn1_clone, bn2_clone], True)
        finalize_bn(model, PopulationBatchNorm2d)
        check2(self, model, [bn1_clone, bn2_clone], False)
        check(self, model, 0)

    def test_conversion2(self):
        for affine_domain in [None, 0, 1]:
            batchnorm_layer = torch.nn.BatchNorm2d(32)
            nested_batchnorm_layer = torch.nn.Sequential(torch.nn.BatchNorm2d(3))
            model = TestModel(batchnorm_layer, nested_batchnorm_layer)
            convert_bn_to_adabn(
                model, affine_domain=affine_domain, bn_type=torch.nn.BatchNorm2d
            )
            model.to(TEST_DEVICE)
            optimize(model)
            check(self, model, affine_domain)

    def test_bad_conversion(self):
        for affine_domain in [None, 0, 1]:
            batchnorm_layer = torch.nn.BatchNorm1d(32)
            nested_batchnorm_layer = torch.nn.Sequential(torch.nn.BatchNorm1d(3))
            model = TestModel(batchnorm_layer, nested_batchnorm_layer)
            with self.assertRaises(TypeError):
                convert_bn_to_adabn(
                    model, affine_domain=affine_domain, bn_type=torch.nn.BatchNorm2d
                )
