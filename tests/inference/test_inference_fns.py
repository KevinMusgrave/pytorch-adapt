import copy
import unittest

import torch

from pytorch_adapt import inference
from pytorch_adapt.containers import Models
from pytorch_adapt.layers import AdaBNModel

from .. import TEST_DEVICE


def get_models_and_data():
    models = {
        "G": torch.nn.Sequential(
            torch.nn.BatchNorm2d(3),
            torch.nn.Conv2d(3, 1, 2),
            torch.nn.Flatten(),
        ).to(TEST_DEVICE),
        "C": torch.nn.Linear(225, 10, device=TEST_DEVICE),
    }
    data = torch.randn(32, 3, 16, 16, device=TEST_DEVICE)
    src_domain = torch.tensor([0], device=TEST_DEVICE)
    target_domain = torch.tensor([1], device=TEST_DEVICE)
    return models, data, src_domain, target_domain


def compare_with_default_fn(cls, data, G, C, in_features, in_logits, should_match):
    features = G(data)
    logits = C(features)
    eq1 = torch.equal(in_features, features)
    eq2 = torch.equal(in_logits, logits)
    for eq in [eq1, eq2]:
        cls.assertTrue(eq if should_match else not eq)


class TestInferenceFns(unittest.TestCase):
    def test_default_fn(self):
        models, data, src_domain, target_domain = get_models_and_data()
        for domain in [src_domain, target_domain]:
            output = inference.default_fn(x=data, models=models)
            compare_with_default_fn(
                self,
                data,
                models["G"],
                models["C"],
                output["features"],
                output["logits"],
                True,
            )

    def test_adabn_fn(self):
        models, data, src_domain, target_domain = get_models_and_data()
        models["G_original"] = copy.deepcopy(models["G"])
        models["C_original"] = copy.deepcopy(models["C"])
        models["G"] = AdaBNModel(models["G"]).to(TEST_DEVICE)
        models["C"] = AdaBNModel(models["C"]).to(TEST_DEVICE)
        models = Models(models)
        models.train()

        # update batchnorm parameters
        for domain in [src_domain, target_domain]:
            for _ in range(100):
                inference.adabn_fn(x=data, domain=domain, models=models)

        models.eval()
        for domain in [src_domain, target_domain]:
            output = inference.adabn_fn(x=data, domain=domain, models=models)

            # original parameters should be different
            compare_with_default_fn(
                self,
                data,
                models["G_original"],
                models["C_original"],
                output["features"],
                output["logits"],
                False,
            )

            features = models["G"](data, domain=domain)
            logits = models["C"](features, domain=domain)
            self.assertTrue(torch.equal(features, output["features"]))
            self.assertTrue(torch.equal(logits, output["logits"]))
