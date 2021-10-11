import unittest

import numpy as np
import torch

from pytorch_adapt.layers import DiversityLoss, EntropyLoss
from pytorch_adapt.validators import (
    DiversityValidator,
    EntropyValidator,
    MultipleValidators,
)


class TestEntropyDiversity(unittest.TestCase):
    def test_entropy_diversity(self):
        results = []
        _logits1 = torch.tensor([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float)
        _logits2 = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        _logits3 = torch.tensor(
            [[100, -100, -100], [-100, 100, -100], [-100, -100, 100]],
            dtype=float,
        )
        correct_entropies = [
            -torch.mean(
                torch.sum(torch.softmax(x, dim=1) * torch.log_softmax(x, dim=1), dim=1)
            )
            for x in [_logits1, _logits2, _logits3]
        ]
        correct_diversities = [
            torch.mean(
                torch.sum(
                    torch.softmax(torch.mean(x, dim=0, keepdims=True), dim=1)
                    * torch.log_softmax(torch.mean(x, dim=0, keepdims=True), dim=1),
                    dim=1,
                )
            )
            for x in [_logits1, _logits2, _logits3]
        ]

        for layer in ["logits", "preds"]:
            for ignore_epoch in [None, 2]:
                validator = MultipleValidators(
                    validators={
                        "entropy": EntropyValidator(layer=layer),
                        "diversity": DiversityValidator(layer=layer),
                    },
                    ignore_epoch=ignore_epoch,
                )

                if layer == "preds":
                    [logits1, logits2, logits3] = [
                        torch.softmax(x, dim=1) for x in [_logits1, _logits2, _logits3]
                    ]
                else:
                    [logits1, logits2, logits3] = [_logits1, _logits2, _logits3]
                score1 = validator.score(epoch=0, target_train={layer: logits1})
                score2 = validator.score(epoch=1, target_train={layer: logits2})
                score3 = validator.score(epoch=2, target_train={layer: logits3})
                results.append([score1, score2, score3])
                self.assertTrue(score3 > score2 > score1)
                if ignore_epoch is None:
                    self.assertTrue(validator.best_epoch == 2)
                    self.assertTrue(validator.best_score == score3)
                else:
                    self.assertTrue(validator.best_epoch == 1)
                    self.assertTrue(validator.best_score == score2)

                for idx, (score, logits) in enumerate(
                    [
                        (score1, logits1),
                        (score2, logits2),
                        (score3, logits3),
                    ]
                ):
                    after_softmax = layer == "preds"
                    self.assertTrue(
                        score
                        == (
                            -EntropyLoss(after_softmax=after_softmax)(logits)
                            - DiversityLoss(after_softmax=after_softmax)(logits)
                        )
                    )
                    self.assertTrue(
                        np.isclose(
                            score,
                            -correct_entropies[idx].item()
                            - correct_diversities[idx].item(),
                        )
                    )

        # assert that all results are the same
        self.assertTrue(results.count(results[0]) == len(results))
