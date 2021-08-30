import unittest

import numpy as np
import torch

from pytorch_adapt.validators import ClusterValidator, KNNValidator

from .. import TEST_DEVICE


class TestKNNValidator(unittest.TestCase):
    def test_knn_validator(self):
        knn_validator = KNNValidator()
        cluster_validator = ClusterValidator()
        for epoch in [1, 2]:
            for validator in [knn_validator, cluster_validator]:
                dataset_size = 10000
                if epoch == 1:
                    src_features = torch.randn(dataset_size, 64, device=TEST_DEVICE)
                    target_features = torch.randn(dataset_size, 64, device=TEST_DEVICE)
                else:
                    src_features = torch.ones(dataset_size, 64, device=TEST_DEVICE)
                    target_features = torch.zeros(dataset_size, 64, device=TEST_DEVICE)
                src_domain = torch.zeros(dataset_size, device=TEST_DEVICE)
                target_domain = torch.ones(dataset_size, device=TEST_DEVICE)

                score = validator.score(
                    epoch=epoch,
                    src_train={"features": src_features, "domain": src_domain},
                    target_train={"features": target_features, "domain": target_domain},
                )
                if epoch == 1:
                    self.assertTrue(
                        score < (0.51 if validator is knn_validator else 0.001)
                    )
                else:
                    self.assertTrue(score == 1)

        self.assertTrue(knn_validator.best_epoch == 1)
        self.assertTrue(knn_validator.best_score < 0.51)

        self.assertTrue(cluster_validator.best_epoch == 1)
        self.assertTrue(cluster_validator.best_score < 0.001)
