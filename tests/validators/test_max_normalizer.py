import unittest

import numpy as np
import torch

from pytorch_adapt.validators import ErrorValidator, ScoreHistory
from pytorch_adapt.validators.utils import max_normalizer


class TestMaxNormalizer(unittest.TestCase):
    def test_max_normalizer(self):
        se = ScoreHistory(ErrorValidator(), normalizer=max_normalizer)
        labels = torch.ones(10, dtype=int)
        logits = torch.randn(10, 2)
        logits[0, 1] -= 2000
        logits[1:, 1] += 1000
        for i in range(1, 10):
            preds = torch.softmax(logits, dim=1)
            kwargs = {"src_val": {"labels": labels, "preds": preds}}
            score2 = se.score(epoch=i, **kwargs)
            self.assertTrue(np.isclose(se.raw_score_history[-1], -i / 10.0))
            self.assertTrue(np.isclose(se.score_history[-1], -i))
            self.assertTrue(np.isclose(score2, -i))
            logits[i, 1] -= 2000
