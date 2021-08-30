import unittest

import numpy as np

from pytorch_adapt.validators import AccuracyValidator


class TestAllNanScoreHistory(unittest.TestCase):
    def test_all_nan_score_history(self):
        ignore_epoch = 10
        se = AccuracyValidator(ignore_epoch=ignore_epoch)
        se.score_history = np.array([float("nan"), float("nan")])
        se.epochs = np.array([0, 5])
        self.assertTrue(se.best_epoch is None)
        self.assertTrue(se.best_score is None)

        for epoch in [10, 11]:
            se.score_history = np.append(se.score_history, 3)
            se.epochs = np.append(se.epochs, epoch)
            if epoch == ignore_epoch:
                self.assertTrue(se.best_epoch is None)
                self.assertTrue(se.best_score is None)
            else:
                self.assertTrue(se.best_epoch == 11)
                self.assertTrue(se.best_score == 3)
