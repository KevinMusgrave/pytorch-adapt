import unittest

import numpy as np
import torch

from pytorch_adapt.utils import common_functions as c_f
from pytorch_adapt.validators import (
    KNNValidator,
    MMDValidator,
    PerClassValidator,
    SNDValidator,
)
from pytorch_adapt.validators.per_class_validator import get_common_labels

from .. import TEST_DEVICE
from .utils import get_knn_func


def get_data(dataset_size, domain, label_range=None):
    label_range = c_f.default(label_range, [0, 7])
    features = torch.randn(dataset_size, 128, device=TEST_DEVICE)
    labels = torch.randint(*label_range, size=(dataset_size,), device=TEST_DEVICE)
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


def map_keys(kwargs, validator):
    for k, v in validator.validator.key_map.items():
        kwargs[k] = kwargs.pop(v)
    return {k: v for k, v in kwargs.items() if k in validator.required_data}


def get_correct_score(data, validator):
    labels = torch.argmax(data["target_train"]["logits"], dim=1)
    unique_labels = torch.unique(labels)
    score = 0
    for L in unique_labels:
        curr_data = {}
        curr_data["target_train"] = {
            k: v[labels == L] for k, v in data["target_train"].items()
        }
        curr_data["src_train"] = {
            k: v[data["src_train"]["labels"] == L] for k, v in data["src_train"].items()
        }
        curr_data = map_keys(curr_data, validator)
        score += validator(**curr_data)
    return score / len(unique_labels)


class TestPerClassValidator(unittest.TestCase):
    def test_per_class_validator(self):
        torch.manual_seed(204)
        dataset_size = 256
        knn_func = get_knn_func()
        inner_validators = [
            KNNValidator(
                key_map={"src_val": "src_train"}, metric="AMI", knn_func=knn_func
            ),
            KNNValidator(
                key_map={"src_val": "src_train", "target_val": "target_train"},
                knn_func=knn_func,
            ),
            MMDValidator(mmd_kwargs={"mmd_type": "quadratic"}),
            SNDValidator(),
        ]
        for v in inner_validators:
            validator = PerClassValidator(v)
            kwargs = {
                "src_train": get_data(dataset_size, 0),
                "target_train": get_data(dataset_size, 1),
            }
            correct_score = get_correct_score(kwargs, validator)
            kwargs = map_keys(kwargs, validator)

            score = validator(**kwargs)
            if isinstance(v, MMDValidator):
                self.assertTrue(np.isclose(score, correct_score, rtol=0.05))
            else:
                self.assertTrue(score == correct_score)

    def test_common_labels(self):
        def label_fn(x):
            return x["labels"]

        kwargs = {
            "src_train": get_data(100, 0, label_range=[0, 5]),
            "target_train": get_data(100, 1, label_range=[3, 6]),
        }
        _, common_labels = get_common_labels(
            kwargs, {"src_train": label_fn, "target_train": label_fn}
        )
        self.assertTrue(common_labels == {3, 4})
