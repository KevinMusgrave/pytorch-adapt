import unittest

import torch

from pytorch_adapt.validators import TargetClusterValidator, TargetKNNValidator

from .. import TEST_DEVICE


def target_label_fn(x):
    return x["labels"]


class TestTargetKNNValidator(unittest.TestCase):
    def test_target_knn_validator(self):
        src_features = torch.randn(5, 32, device=TEST_DEVICE)
        src_labels = torch.arange(0, 5, device=TEST_DEVICE)
        target_features = torch.randn(1, 32, device=TEST_DEVICE).repeat(3, 1)
        target_labels = torch.arange(0, 3, device=TEST_DEVICE)

        for i in range(5):
            src_features[i] += i * 100

        data = {
            "src_train": {"features": src_features, "labels": src_labels},
            "target_train": {"features": target_features, "labels": target_labels},
        }

        for add_target_to_ref in [False, True]:
            validator = TargetKNNValidator(
                add_target_to_ref=add_target_to_ref, target_label_fn=target_label_fn
            )

            score = validator(**data)

            # Case add_target_to_ref == False
            # All 3 target features are identical
            # src_features[0] is near all 3 target features
            # So the only correct match is for target_features[0]

            # Case add_target_to_ref == True
            # All target_features match with each other
            # because they are co-located.
            # Since they have different labels, accuracy is 0

            self.assertTrue(score == 0 if add_target_to_ref else 1.0 / 3)

    def test_target_cluster_validator(self):
        features, labels = [], []
        num_classes = 5
        for i in range(num_classes):
            features.append(torch.randn(32, 128, device=TEST_DEVICE) + i * 100)
            labels.append(torch.ones(32, device=TEST_DEVICE) * i)

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        validator = TargetClusterValidator(target_label_fn=target_label_fn)

        args = {"features": features, "labels": labels}
        score = validator(src_train=args, target_train=args)
        self.assertTrue(score > 0.8)

        args["labels"] = torch.randint(0, num_classes, size=args["labels"].shape)
        score = validator(src_train=args, target_train=args)
        self.assertTrue(score < 0.1)
