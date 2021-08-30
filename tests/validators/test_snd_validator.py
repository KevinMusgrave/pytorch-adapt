import unittest

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_adapt.validators import SNDValidator

from .. import TEST_DEVICE


class TestSNDValidator(unittest.TestCase):
    def test_snd_validator(self):
        ignore_epoch = 0
        for T in [0.01, 0.05, 0.5, 1]:
            for batch_size in [33, 257, 999, 1000, 1001]:
                for dataset_size in [100, 1000, 10000]:
                    for num_classes in [13, 23, 98]:
                        validator = SNDValidator(
                            T=T, batch_size=batch_size, ignore_epoch=ignore_epoch
                        )
                        all_scores = []
                        for epoch in [0, 1, 2]:
                            logits = torch.randn(
                                dataset_size, num_classes, device=TEST_DEVICE
                            )
                            logits = F.softmax(logits, dim=1)
                            target_train = {"preds": logits}

                            score = validator.score(
                                epoch=epoch, target_train=target_train
                            )

                            logits = F.normalize(logits, dim=1)
                            sim_mat = torch.mm(logits, logits.t())
                            n = sim_mat.shape[0]
                            mask = torch.eye(n, dtype=torch.bool)
                            sim_mat = sim_mat[~mask].view(n, n - 1)
                            sim_mat = F.softmax(sim_mat / T, dim=1)
                            entropies = -torch.sum(sim_mat * torch.log(sim_mat), dim=1)
                            correct_score = torch.mean(entropies).item()

                            self.assertTrue(np.isclose(score, correct_score))

                            if epoch != ignore_epoch:
                                all_scores.append(correct_score)

                        self.assertTrue(validator.best_epoch != ignore_epoch)
                        self.assertTrue(
                            np.isclose(validator.best_score, max(all_scores))
                        )
