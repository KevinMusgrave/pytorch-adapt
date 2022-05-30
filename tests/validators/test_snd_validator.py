import unittest

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_metric_learning.utils import common_functions as pml_cf
from tqdm import tqdm

from pytorch_adapt.adapters import Classifier
from pytorch_adapt.containers import Models, Optimizers
from pytorch_adapt.datasets import (
    CombinedSourceAndTargetDataset,
    SourceDataset,
    TargetDataset,
)
from pytorch_adapt.frameworks.ignite import Ignite, IgnitePredsAsFeatures
from pytorch_adapt.validators import ScoreHistory, SNDValidator

from .. import TEST_DEVICE


def simple_compute_snd(logits, T):
    logits = F.softmax(logits, dim=1)
    logits = F.normalize(logits, dim=1)
    sim_mat = torch.mm(logits, logits.t())
    n = sim_mat.shape[0]
    mask = torch.eye(n, dtype=torch.bool)
    sim_mat = sim_mat[~mask].view(n, n - 1)
    sim_mat = F.softmax(sim_mat / T, dim=1)
    entropies = -torch.sum(sim_mat * torch.log(sim_mat), dim=1)
    return torch.mean(entropies).item()


class TestSNDValidator(unittest.TestCase):
    def test_snd_validator(self):
        ignore_epoch = 0
        for T in tqdm([0.01, 0.05, 0.5, 1]):
            for batch_size in [33, 257, 999, 1000, 1001]:
                for dataset_size in [10, 100, 1000, 2000]:
                    for num_classes in [13, 23, 98]:
                        validator = SNDValidator(T=T, batch_size=batch_size)
                        validator = ScoreHistory(validator, ignore_epoch=ignore_epoch)
                        all_scores = []
                        for epoch in [0, 1, 2]:
                            logits = torch.randn(
                                dataset_size, num_classes, device=TEST_DEVICE
                            )
                            target_train = {"preds": F.softmax(logits, dim=1)}
                            score = validator(epoch=epoch, target_train=target_train)
                            correct_score = simple_compute_snd(logits, T)
                            self.assertTrue(np.isclose(score, correct_score))

                            if epoch != ignore_epoch:
                                all_scores.append(correct_score)

                        self.assertTrue(validator.best_epoch != ignore_epoch)
                        self.assertTrue(
                            np.isclose(validator.best_score, max(all_scores))
                        )

    def test_snd_validator_with_framework(self):
        for wrapper_type in [Ignite, IgnitePredsAsFeatures]:
            dataset_size = 9999

            train_datasets = []
            all_features = []
            for _ in range(3):
                features = torch.randn(dataset_size, 128)
                labels = torch.randint(0, 10, size=(dataset_size,))
                all_features.append(features)
                train_datasets.append(pml_cf.EmbeddingDataset(features, labels))

            train_dataset = CombinedSourceAndTargetDataset(
                SourceDataset(train_datasets[0]), TargetDataset(train_datasets[1])
            )
            target_train = TargetDataset(train_datasets[2])

            C = torch.nn.Sequential(torch.nn.Linear(128, 10), torch.nn.Softmax(dim=1))
            models = Models({"G": C, "C": torch.nn.Identity()})
            optimizers = Optimizers((torch.optim.Adam, {"lr": 0}))
            adapter = wrapper_type(
                Classifier(models=models, optimizers=optimizers),
                validator=ScoreHistory(SNDValidator()),
                device=TEST_DEVICE,
            )
            score, _ = adapter.run(
                {"train": train_dataset, "target_train": target_train},
                epoch_length=1,
            )

            with torch.no_grad():
                if wrapper_type is Ignite:
                    logits = C(all_features[2].to(TEST_DEVICE))
                else:
                    logits = C[0](all_features[2].to(TEST_DEVICE))

            correct_score = simple_compute_snd(logits, T=0.05)
            self.assertTrue(np.isclose(score, correct_score))
