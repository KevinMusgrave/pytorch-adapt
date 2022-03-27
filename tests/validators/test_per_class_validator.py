import unittest

import torch

from pytorch_adapt.validators import (
    ClusterValidator,
    KNNValidator,
    MMDValidator,
    PerClassValidator,
    SNDValidator,
)

from .. import TEST_DEVICE


def get_data(dataset_size, domain):
    features = torch.randn(dataset_size, 128, device=TEST_DEVICE)
    labels = torch.randint(0, 7, size=(dataset_size,), device=TEST_DEVICE)
    logits = torch.randn(dataset_size, 7, device=TEST_DEVICE)
    preds = torch.softmax(logits, dim=1)
    domain = torch.ones(dataset_size, device=TEST_DEVICE) * domain
    return {
        "features": features,
        "labels": labels,
        "logits": logits,
        "preds": preds,
        "domain": domain,
    }


class TestPerClassValidator(unittest.TestCase):
    def test_per_class_validator(self):
        dataset_size = 2048
        inner_validators = [
            ClusterValidator(),
            KNNValidator(),
            MMDValidator(num_samples=1024),
            SNDValidator(),
        ]
        for v in inner_validators:
            validator = PerClassValidator(v)
            kwargs = {
                "src_train": get_data(dataset_size, 0),
                "target_train": get_data(dataset_size, 1),
            }
            kwargs = {k: v for k, v in kwargs.items() if k in validator.required_data}
            score = validator(**kwargs)
            print(score)
