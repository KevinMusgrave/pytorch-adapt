import unittest

import numpy as np
import torch

from pytorch_adapt.layers import SufficientAccuracy

from .. import TEST_DEVICE, TEST_DTYPES


def correct_fn(x, labels, threshold):
    pred = torch.round(torch.sigmoid(x))
    d_accuracy = torch.mean((pred == labels).float()).item()
    return d_accuracy > threshold, d_accuracy


class TestSufficientAccuracy(unittest.TestCase):
    def test_sufficient_accuracy(self):
        batch_size = 32
        embedding_size = 1
        for threshold in np.linspace(0, 1, 100):
            loss_fn = SufficientAccuracy(
                threshold, to_probs_func=torch.nn.Sigmoid()
            ).to(TEST_DEVICE)
            for dtype in TEST_DTYPES:
                if dtype == torch.float16:
                    continue
                x = torch.randn(batch_size, device=TEST_DEVICE, dtype=dtype)
                labels = torch.randint(0, 2, size=(batch_size,), device=TEST_DEVICE)
                result = loss_fn(x, labels)
                acc = loss_fn.accuracy

                correct_result, correct_acc = correct_fn(x, labels, threshold)
                self.assertTrue(np.isclose(result, correct_result))
                self.assertTrue(np.isclose(acc, correct_acc, rtol=1e-2))
