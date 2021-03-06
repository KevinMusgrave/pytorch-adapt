import copy
import unittest

import torch

from pytorch_adapt import inference
from pytorch_adapt.containers import Models
from pytorch_adapt.layers import (
    AdaBNModel,
    ModelWithBridge,
    MultipleModels,
    RandomizedDotProduct,
)
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


def compare_with_default_fn(cls, output, default_output, should_match=False):
    eq1 = torch.equal(output["features"], default_output["features"])
    eq2 = torch.equal(output["logits"], default_output["logits"])
    for i, eq in enumerate([eq1, eq2]):
        should_match_ = should_match
        if isinstance(should_match, list):
            should_match_ = should_match[i]
        cls.assertTrue(eq if should_match_ else not eq)


class TestInferenceFns(unittest.TestCase):
    def test_default_fn(self):
        models, data, src_domain, target_domain = get_models_and_data()
        for domain in [src_domain, target_domain]:
            output = inference.default_fn(x=data, models=models)
            self.assertTrue(output.keys() == {"features", "logits"})
            features = models["G"](data)
            logits = models["C"](features)
            compare_with_default_fn(
                self, output, {"features": features, "logits": logits}, True
            )

    def test_with_d(self):
        models, data, src_domain, target_domain = get_models_and_data(with_D=True)
        for domain in [src_domain, target_domain]:
            output1 = inference.with_d(x=data, models=models, fn=inference.default_fn)
            output2 = inference.default_with_d(x=data, models=models)
            for output in [output1, output2]:
                self.assertTrue(output.keys() == {"features", "logits", "d_logits"})
                compare_with_default_fn(
                    self, output, inference.default_fn(x=data, models=models), True
                )
                d_logits = models["D"](models["G"](data))
                self.assertTrue(torch.equal(d_logits, output["d_logits"]))

    def test_default_with_d_logits_layer(self):
        models, data, src_domain, target_domain = get_models_and_data()
        models["D"] = torch.nn.Linear(10, 1, device=TEST_DEVICE)
        for domain in [src_domain, target_domain]:
            output = inference.default_with_d_logits_layer(x=data, models=models)
            self.assertTrue(output.keys() == {"features", "logits", "d_logits"})
            compare_with_default_fn(
                self, output, inference.default_fn(x=data, models=models), True
            )
            d_logits = models["D"](models["C"](models["G"](data)))
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
            self.assertTrue(output.keys() == {"features", "logits"})

            # original parameters should be different
            default_output = inference.default_fn(
                x=data, models={"G": models["G_original"], "C": models["C_original"]}
            )
            compare_with_default_fn(self, output, default_output, False)

            features = models["G"](data, domain=domain)
            logits = models["C"](features, domain=domain)
            self.assertTrue(torch.equal(features, output["features"]))
            self.assertTrue(torch.equal(logits, output["logits"]))

    def test_adda_fn(self):
        models, data, src_domain, target_domain = get_models_and_data(with_D=True)
        models["T"] = c_f.reinit(copy.deepcopy(models["G"]))

        for domain in [src_domain, target_domain]:
            output = inference.adda_fn(x=data, domain=domain, models=models)
            self.assertTrue(output.keys() == {"features", "logits"})
            default_output = inference.default_fn(x=data, models=models)
            compare_with_default_fn(self, output, default_output, domain == src_domain)

            gen_model = "G" if domain == src_domain else "T"
            features = models[gen_model](data)
            logits = models["C"](features)
            self.assertTrue(torch.equal(features, output["features"]))
            self.assertTrue(torch.equal(logits, output["logits"]))

            output1 = inference.with_d(
                x=data, domain=domain, models=models, fn=inference.adda_fn
            )
            output2 = inference.adda_with_d(x=data, domain=domain, models=models)
            output3 = inference.adda_full_fn(x=data, domain=domain, models=models)
            for i, output in enumerate([output1, output2, output3]):
                d_logits = models["D"](features)
                self.assertTrue(torch.equal(d_logits, output["d_logits"]))
                if i < 2:
                    self.assertTrue(output.keys() == {"features", "logits", "d_logits"})
                else:
                    self.assertTrue(
                        output.keys()
                        == {
                            "features",
                            "logits",
                            "d_logits",
                            "other_features",
                            "other_logits",
                            "other_d_logits",
                        }
                    )
                    gen_model = "T" if domain == src_domain else "G"
                    other_features = models[gen_model](data)
                    other_logits = models["C"](other_features)
                    other_d_logits = models["D"](other_features)
                    self.assertTrue(
                        torch.equal(other_features, output["other_features"])
                    )
                    self.assertTrue(torch.equal(other_logits, output["other_logits"]))
                    self.assertTrue(
                        torch.equal(other_d_logits, output["other_d_logits"])
                    )

    def test_rtn_fn(self):
        models, data, src_domain, target_domain = get_models_and_data()
        models["residual_model"] = torch.nn.Linear(10, 10).to(TEST_DEVICE)
        misc = {
            "feature_combiner": RandomizedDotProduct(in_dims=[225, 10], out_dim=225)
        }
        for domain in [src_domain, target_domain]:
            output1 = inference.rtn_fn(x=data, domain=domain, models=models)
            output2 = inference.rtn_full_fn(
                x=data, domain=domain, models=models, misc=misc
            )
            self.assertTrue(output1.keys() == {"features", "logits"})
            self.assertTrue(
                output2.keys()
                == {"features", "logits", "other_logits", "features_logits_combined"}
            )
            for i, output in enumerate([output1, output2]):
                default_output = inference.default_fn(x=data, models=models)
                compare_with_default_fn(
                    self, output, default_output, [True, domain == target_domain]
                )

                features = models["G"](data)
                target_logits = models["C"](features)
                src_logits = models["residual_model"](target_logits)
                self.assertTrue(torch.equal(features, output["features"]))
                logits = src_logits if domain == src_domain else target_logits
                self.assertTrue(torch.equal(logits, output["logits"]))
                if i == 1:
                    other_logits = target_logits if domain == src_domain else src_logits
                    self.assertTrue(torch.equal(other_logits, output["other_logits"]))

    def test_mcd_fn(self):
        models, data, src_domain, target_domain = get_models_and_data()
        models["C"] = MultipleModels(
            models["C"], c_f.reinit(copy.deepcopy(models["C"]))
        )

        for domain in [src_domain, target_domain]:
            output1 = inference.mcd_fn(x=data, models=models)
            output2 = inference.mcd_full_fn(x=data, models=models)
            self.assertTrue(output1.keys() == {"features", "logits"})
            self.assertTrue(
                output2.keys() == {"features", "logits", "logits0", "logits1"}
            )
            for i, output in enumerate([output1, output2]):
                default_output = inference.default_fn(x=data, models=models)

                # logits output will be list, not Tensor
                with self.assertRaises(TypeError):
                    compare_with_default_fn(
                        self,
                        output,
                        default_output,
                    )

                features = models["G"](data)
                [logits0, logits1] = models["C"](features)
                self.assertTrue(torch.equal(features, output["features"]))
                self.assertTrue(torch.equal(logits0 + logits1, output["logits"]))
                if i == 1:
                    self.assertTrue(torch.equal(logits0, output["logits0"]))
                    self.assertTrue(torch.equal(logits1, output["logits1"]))

    def test_symnets_fn(self):
        models, data, src_domain, target_domain = get_models_and_data()
        models["C"] = MultipleModels(
            models["C"], c_f.reinit(copy.deepcopy(models["C"]))
        )

        for domain in [src_domain, target_domain]:
            output1 = inference.symnets_fn(x=data, domain=domain, models=models)
            output2 = inference.symnets_full_fn(x=data, domain=domain, models=models)
            self.assertTrue(output1.keys() == {"features", "logits"})
            self.assertTrue(output2.keys() == {"features", "logits", "other_logits"})
            for i, output in enumerate([output1, output2]):
                default_output = inference.default_fn(x=data, models=models)

                # logits output will be list, not Tensor
                with self.assertRaises(TypeError):
                    compare_with_default_fn(
                        self,
                        output,
                        default_output,
                    )

                features = models["G"](data)
                logits = models["C"](features)[0 if domain == src_domain else 1]
                self.assertTrue(torch.equal(features, output["features"]))
                self.assertTrue(torch.equal(logits, output["logits"]))
                if i == 1:
                    other_logits = models["C"](features)[
                        1 if domain == src_domain else 0
                    ]
                    self.assertTrue(torch.equal(other_logits, output["other_logits"]))

    def test_with_feature_combiner_fn(self):
        models, data, src_domain, target_domain = get_models_and_data(with_D=True)
        models["residual_model"] = torch.nn.Linear(10, 10).to(TEST_DEVICE)
        misc = {
            "feature_combiner": RandomizedDotProduct(in_dims=[225, 10], out_dim=225)
        }
        for domain in [src_domain, target_domain]:
            for softmax in [True, False]:
                for fn in [
                    inference.default_fn,
                    inference.default_with_d,
                    inference.rtn_fn,
                    inference.cdan_full_fn,
                    inference.rtn_with_feature_combiner,
                ]:
                    using_rtn = fn in [
                        inference.rtn_fn,
                        inference.rtn_with_feature_combiner,
                    ]

                    kwargs = {
                        "x": data,
                        "models": models,
                        "misc": misc,
                        "softmax": softmax,
                        "domain": domain,
                    }

                    full_fn = fn
                    if full_fn not in [
                        inference.cdan_full_fn,
                        inference.rtn_with_feature_combiner,
                    ]:
                        full_fn = inference.with_feature_combiner
                        kwargs["fn"] = fn

                    output = full_fn(**kwargs)

                    correct_keys = {"features", "logits", "features_logits_combined"}
                    if fn in [inference.default_with_d, inference.cdan_full_fn]:
                        correct_keys.add("d_logits")

                    self.assertTrue(output.keys() == correct_keys)

                    should_match = (
                        [True, domain == target_domain] if using_rtn else True
                    )
                    compare_with_default_fn(
                        self,
                        output,
                        inference.default_fn(x=data, models=models),
                        should_match,
                    )

                    features = models["G"](data)
                    logits = models["C"](features)
                    if using_rtn and domain == src_domain:
                        logits = models["residual_model"](logits)
                    if softmax:
                        logits = torch.nn.Softmax(dim=1)(logits)
                    combined = misc["feature_combiner"](features, logits)
                    self.assertTrue(
                        torch.equal(combined, output["features_logits_combined"])
                    )

    def test_gvb_fn(self):
        models, data, src_domain, target_domain = get_models_and_data()
        models["C"] = ModelWithBridge(models["C"], torch.nn.Linear(225, 10)).to(
            TEST_DEVICE
        )
        models["D"] = ModelWithBridge(
            torch.nn.Linear(10, 1), torch.nn.Linear(10, 1)
        ).to(TEST_DEVICE)

        for domain in [src_domain, target_domain]:
            output1 = inference.with_d_bridge(
                x=data, models=models, fn=inference.gvb_with_g_bridge
            )
            output2 = inference.gvb_full_fn(x=data, models=models)
            for output in [output1, output2]:
                self.assertTrue(
                    output.keys()
                    == {"features", "logits", "g_bridge", "d_bridge", "d_logits"}
                )
                compare_with_default_fn(
                    self, output, inference.default_fn(x=data, models=models), True
                )

                features = models["G"](data)
                [logits, g_bridge] = models["C"](features, return_bridge=True)
                [d_logits, d_bridge] = models["D"](
                    torch.softmax(logits, dim=1), return_bridge=True
                )
                self.assertTrue(torch.equal(d_logits, output["d_logits"]))
                self.assertTrue(torch.equal(g_bridge, output["g_bridge"]))
                self.assertTrue(torch.equal(d_bridge, output["d_bridge"]))
