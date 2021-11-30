import unittest

import numpy as np
import torch

from pytorch_adapt.layers import DiversityLoss, EntropyLoss
from pytorch_adapt.validators import (
    DiversityValidator,
    EntropyValidator,
    IMValidator,
    MultipleValidators,
    WithHistory,
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
                    key_map={"target_val": "target_train"},
                )
                validator = WithHistory(validator, ignore_epoch=ignore_epoch)
                validator2 = WithHistory(
                    IMValidator(layer=layer, key_map={"target_val": "target_train"}),
                    ignore_epoch=ignore_epoch,
                )

                if layer == "preds":
                    [logits1, logits2, logits3] = [
                        torch.softmax(x, dim=1) for x in [_logits1, _logits2, _logits3]
                    ]
                else:
                    [logits1, logits2, logits3] = [_logits1, _logits2, _logits3]

                score1, score2, score3 = [], [], []
                for v in [validator, validator2]:
                    score1.append(v.score(epoch=0, target_val={layer: logits1}))
                    score2.append(v.score(epoch=1, target_val={layer: logits2}))
                    score3.append(v.score(epoch=2, target_val={layer: logits3}))

                v1_scores, v2_scores = list(zip(score1, score2, score3))

                results.append(v1_scores)
                self.assertTrue(v1_scores[2] > v1_scores[1] > v1_scores[0])
                if ignore_epoch is None:
                    self.assertTrue(validator.best_epoch == 2)
                    self.assertTrue(validator.best_score == v1_scores[2])
                else:
                    self.assertTrue(validator.best_epoch == 1)
                    self.assertTrue(validator.best_score == v1_scores[1])

                self.assertTrue(all(v1 == v2 for v1, v2 in zip(v1_scores, v2_scores)))

                for idx, (score, logits) in enumerate(
                    [
                        (v1_scores[0], logits1),
                        (v1_scores[1], logits2),
                        (v1_scores[2], logits3),
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
