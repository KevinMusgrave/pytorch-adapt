import unittest

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from pytorch_adapt.validators import ScoreHistory, SNDValidator

from .. import TEST_DEVICE
from .utils import test_with_ignite_framework


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
        def assertion_fn(logits, labels, score):
            correct_score = simple_compute_snd(logits["target_train"], T=0.05)
            self.assertTrue(np.isclose(score, correct_score))

        test_with_ignite_framework(SNDValidator(), assertion_fn)
