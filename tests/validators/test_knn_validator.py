import unittest

import torch
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from pytorch_adapt.validators import ClusterValidator, KNNValidator, ScoreHistory
from pytorch_adapt.validators.knn_validator import BatchedAccuracyCalculator

from .. import TEST_DEVICE


class TestKNNValidator(unittest.TestCase):
    def test_knn_validator(self):
        torch.cuda.empty_cache()
        knn_validator = ScoreHistory(KNNValidator())
        cluster_validator = ScoreHistory(KNNValidator(metric="AMI"))
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

                score = validator(
                    epoch=epoch,
                    src_train={"features": src_features, "domain": src_domain},
                    target_train={"features": target_features, "domain": target_domain},
                )
                if epoch == 1:
                    self.assertTrue(
                        score > (-0.51 if validator is knn_validator else -0.001)
                    )
                else:
                    self.assertTrue(score == -1)

        self.assertTrue(knn_validator.best_epoch == 1)
        self.assertTrue(knn_validator.best_score > -0.51)

        self.assertTrue(cluster_validator.best_epoch == 1)
        self.assertTrue(cluster_validator.best_score > -0.001)

    def test_batched_vs_regular(self):
        metric = "mean_average_precision"
        for batch_size in [None, 10, 99, 128, 500, 512]:
            for dataset_size in [10, 100, 1000, 10000]:
                src_features = torch.randn(dataset_size, 64, device=TEST_DEVICE)
                target_features = torch.randn(dataset_size, 64, device=TEST_DEVICE)
                src_domain = torch.zeros(dataset_size, device=TEST_DEVICE)
                target_domain = torch.ones(dataset_size, device=TEST_DEVICE)

                src_train = {"features": src_features, "domain": src_domain}
                target_train = {"features": target_features, "domain": target_domain}

                v1 = KNNValidator(metric=metric)
                v2 = KNNValidator(batch_size=batch_size, metric=metric)

                if batch_size is not None:
                    self.assertTrue(isinstance(v1.acc_fn, AccuracyCalculator))
                    self.assertTrue(isinstance(v2.acc_fn, BatchedAccuracyCalculator))

                scores = [
                    v(src_train=src_train, target_train=target_train) for v in [v1, v2]
                ]
                self.assertTrue(scores[0] == scores[1])

    def test_cluster_validator(self):
        def target_label_fn(x):
            return x["labels"]

        features, labels = [], []
        num_classes = 5
        for i in range(num_classes):
            features.append(torch.randn(32, 128, device=TEST_DEVICE) + i * 100)
            labels.append(torch.ones(32, device=TEST_DEVICE) * i)

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        validator = ClusterValidator(target_label_fn=target_label_fn)

        args = {"features": features, "labels": labels}
        score = validator(src_train=args, target_train=args)
        self.assertTrue(score > 0.8)

        args["labels"] = torch.randint(0, num_classes, size=args["labels"].shape)
        score = validator(src_train=args, target_train=args)
        self.assertTrue(score < 0.1)
