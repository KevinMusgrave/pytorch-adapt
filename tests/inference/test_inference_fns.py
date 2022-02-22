import copy
import unittest

import torch

from pytorch_adapt import inference
from pytorch_adapt.containers import Models
from pytorch_adapt.layers import AdaBNModel, MultipleModels
from pytorch_adapt.utils import common_functions as c_f

from .. import TEST_DEVICE


def get_models_and_data(with_D=False):
    models = {
        "G": torch.nn.Sequential(
            torch.nn.BatchNorm2d(3),
            torch.nn.Conv2d(3, 1, 2),
            torch.nn.Flatten(),
        ).to(TEST_DEVICE),
        "C": torch.nn.Linear(225, 10, device=TEST_DEVICE),
    }
    if with_D:
        models["D"] = torch.nn.Linear(225, 1, device=TEST_DEVICE)
    data = torch.randn(32, 3, 16, 16, device=TEST_DEVICE)
    src_domain = torch.tensor([0], device=TEST_DEVICE)
    target_domain = torch.tensor([1], device=TEST_DEVICE)
    return models, data, src_domain, target_domain


def compare_with_gc_fn(cls, output, default_output, should_match=False):
    eq1 = torch.equal(output["features"], default_output["features"])
    eq2 = torch.equal(output["logits"], default_output["logits"])
    for i, eq in enumerate([eq1, eq2]):
        should_match_ = should_match
        if isinstance(should_match, list):
            should_match_ = should_match[i]
        cls.assertTrue(eq if should_match_ else not eq)


class TestInferenceFns(unittest.TestCase):
    def test_gc_fn(self):
        models, data, src_domain, target_domain = get_models_and_data()
        for domain in [src_domain, target_domain]:
            output = inference.gc_fn(x=data, models=models)
            features = models["G"](data)
            logits = models["C"](features)
            compare_with_gc_fn(
                self, output, {"features": features, "logits": logits}, True
            )

    def test_gcd_fn(self):
        models, data, src_domain, target_domain = get_models_and_data(with_D=True)
        for domain in [src_domain, target_domain]:
            output = inference.gcd_fn(x=data, models=models)
            compare_with_gc_fn(
                self, output, inference.gc_fn(x=data, models=models), True
            )
            d_logits = models["D"](models["G"](data))
            self.assertTrue(torch.equal(d_logits, output["d_logits"]))

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
            default_output = inference.gc_fn(
                x=data, models={"G": models["G_original"], "C": models["C_original"]}
            )
            compare_with_gc_fn(self, output, default_output, False)

            features = models["G"](data, domain=domain)
            logits = models["C"](features, domain=domain)
            self.assertTrue(torch.equal(features, output["features"]))
            self.assertTrue(torch.equal(logits, output["logits"]))

    def test_adda_fn(self):
        models, data, src_domain, target_domain = get_models_and_data()
        models["T"] = c_f.reinit(copy.deepcopy(models["G"]))

        for domain in [src_domain, target_domain]:
            output = inference.adda_fn(x=data, domain=domain, models=models)
            default_output = inference.gc_fn(x=data, models=models)
            compare_with_gc_fn(self, output, default_output, domain == src_domain)

            features = models["G" if domain == src_domain else "T"](data)
            logits = models["C"](features)
            self.assertTrue(torch.equal(features, output["features"]))
            self.assertTrue(torch.equal(logits, output["logits"]))

    def test_rtn_fn(self):
        models, data, src_domain, target_domain = get_models_and_data()
        models["residual_model"] = torch.nn.Linear(10, 10).to(TEST_DEVICE)

        for domain in [src_domain, target_domain]:
            output = inference.rtn_fn(x=data, domain=domain, models=models)
            default_output = inference.gc_fn(x=data, models=models)
            compare_with_gc_fn(
                self, output, default_output, [True, domain == target_domain]
            )

            features = models["G"](data)
            target_logits = models["C"](features)
            src_logits = models["residual_model"](target_logits)
            self.assertTrue(torch.equal(features, output["features"]))
            logits = src_logits if domain == src_domain else target_logits
            self.assertTrue(torch.equal(logits, output["logits"]))

    def test_mcd_fn(self):
        models, data, src_domain, target_domain = get_models_and_data()
        models["C"] = MultipleModels(
            models["C"], c_f.reinit(copy.deepcopy(models["C"]))
        )

        for domain in [src_domain, target_domain]:
            output = inference.mcd_fn(x=data, models=models)
            default_output = inference.gc_fn(x=data, models=models)

            # logits output will be list, not Tensor
            with self.assertRaises(TypeError):
                compare_with_gc_fn(
                    self,
                    output,
                    default_output,
                )

            features = models["G"](data)
            [logits1, logits2] = models["C"](features)
            self.assertTrue(torch.equal(features, output["features"]))
            self.assertTrue(torch.equal(logits1 + logits2, output["logits"]))

    def test_symnets_fn(self):
        models, data, src_domain, target_domain = get_models_and_data()
        models["C"] = MultipleModels(
            models["C"], c_f.reinit(copy.deepcopy(models["C"]))
        )

        for domain in [src_domain, target_domain]:
            output = inference.symnets_fn(x=data, domain=domain, models=models)
            default_output = inference.gc_fn(x=data, models=models)

            # logits output will be list, not Tensor
            with self.assertRaises(TypeError):
                compare_with_gc_fn(
                    self,
                    output,
                    default_output,
                )

            features = models["G"](data)
            logits = models["C"](features)[0 if domain == src_domain else 1]
            self.assertTrue(torch.equal(features, output["features"]))
            self.assertTrue(torch.equal(logits, output["logits"]))