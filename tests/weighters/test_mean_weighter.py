import unittest

import torch

from pytorch_adapt.weighters import MeanWeighter


class TestMeanWeighter(unittest.TestCase):
    def test_mean_weighter(self):
        losses = {"A": torch.tensor(1), "B": torch.tensor(2), "C": torch.tensor(12)}

        total_loss, _ = MeanWeighter()(losses)
        self.assertTrue(total_loss == 5)

        total_loss, _ = MeanWeighter(weights={"A": 4})(losses)
        self.assertTrue(total_loss == 6)

        total_loss, _ = MeanWeighter(weights={"C": 0.5})(losses)
        self.assertTrue(total_loss == 3)

        self.assertRaises(KeyError, lambda: MeanWeighter(weights={"X": 0.5})(losses))
