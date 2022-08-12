import unittest

import numpy as np
import torch
from sklearn.metrics import accuracy_score

from pytorch_adapt.validators import AccuracyValidator, ScoreHistory


class TestAccuracyValidator(unittest.TestCase):
    def test_accuracy_validator(self):
        dataset_size = 1000
        ignore_epoch = 0

        for start in [-1, 0, 1]:
            for step in [1, 2]:
                validator1 = ScoreHistory(
                    AccuracyValidator(), ignore_epoch=ignore_epoch
                )
                validator2 = ScoreHistory(
                    AccuracyValidator(layer="logits"), ignore_epoch=ignore_epoch
                )
                correct_scores = []
                for i, epoch in enumerate(range(start, 5, step)):
                    labels = torch.randint(0, 10, (dataset_size,))
                    logits = torch.randn(dataset_size, 10)
                    preds = torch.softmax(logits, dim=1)
                    score1 = validator1(
                        epoch=epoch, src_val={"labels": labels, "preds": preds}
                    )
                    score2 = validator2(
                        epoch=epoch, src_val={"labels": labels, "logits": logits}
                    )
                    correct_score = accuracy_score(
                        labels.numpy(), np.argmax(logits.numpy(), axis=1)
                    )

                    if epoch != ignore_epoch:
                        correct_scores.append(correct_score)

                    for validator, score in [
                        (validator1, score1),
                        (validator2, score2),
                    ]:
                        if epoch != ignore_epoch:
                            self.assertTrue(
                                validator.score_history_ignore_epoch[validator.best_idx]
                                == validator.best_score
                            )
                            self.assertTrue(
                                validator.epochs_ignore_epoch[validator.best_idx]
                                == validator.best_epoch
                            )
                            self.assertTrue(
                                np.isclose(validator.best_score, max(correct_scores))
                            )
                            self.assertTrue(
                                np.isclose(
                                    validator.best_idx, np.argmax(correct_scores)
                                )
                            )
                        elif i == 0 and epoch == ignore_epoch:
                            self.assertTrue(validator.best_epoch is None)
                            self.assertTrue(validator.best_score is None)

                        self.assertTrue(np.isclose(score, correct_score))
                        self.assertTrue(validator.latest_score == score)
