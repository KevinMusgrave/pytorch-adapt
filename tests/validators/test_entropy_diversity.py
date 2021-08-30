import unittest

import numpy as np
import torch

from pytorch_adapt.validators import (
    DiversityValidator,
    EntropyValidator,
    MultipleValidators,
)


class TestEntropyDiversity(unittest.TestCase):
    def test_entropy_diversity(self):
        for ignore_epoch in [None, 2]:
            validator = MultipleValidators(
                validators={
                    "entropy": EntropyValidator(),
                    "diversity": DiversityValidator(),
                },
                ignore_epoch=ignore_epoch,
            )
            logits1 = torch.tensor([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float)
            logits2 = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
            logits3 = torch.tensor(
                [[100, -100, -100], [-100, 100, -100], [-100, -100, 100]], dtype=float
            )
            score1 = validator.score(epoch=0, target_train={"logits": logits1})
            score2 = validator.score(epoch=1, target_train={"logits": logits2})
            score3 = validator.score(epoch=2, target_train={"logits": logits3})
            self.assertTrue(score3 > score2 > score1)
            if ignore_epoch is None:
                self.assertTrue(validator.best_epoch == 2)
                self.assertTrue(validator.best_score == score3)
            else:
                self.assertTrue(validator.best_epoch == 1)
                self.assertTrue(validator.best_score == score2)
