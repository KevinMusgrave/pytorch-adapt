import unittest

import torch

from pytorch_adapt.layers import EntropyWeights, MinMaxNormalizer, SumNormalizer
from pytorch_adapt.layers.entropy_loss import entropy

from .. import TEST_DEVICE, TEST_DTYPES


class TestEntropyWeights(unittest.TestCase):
    def test_entropy_weights(self):
        torch.manual_seed(340)
        softmax = torch.nn.functional.softmax
        fn1 = EntropyWeights(normalizer=SumNormalizer())
        fn2 = EntropyWeights(normalizer=MinMaxNormalizer())
        fn1_s = EntropyWeights(after_softmax=True, normalizer=SumNormalizer())
        fn2_s = EntropyWeights(after_softmax=True, normalizer=MinMaxNormalizer())
        for dtype in TEST_DTYPES:
            batch_size = 32
            embedding_size = 100
            x = (
                torch.randn(batch_size, embedding_size, device=TEST_DEVICE, dtype=dtype)
                ** 3
            )
            w1 = fn1(x)
            w2 = fn2(x)

            unnormalized = 1 + torch.exp(-entropy(x))
            correct_w1 = unnormalized / torch.sum(unnormalized)
            correct_w2 = (unnormalized - torch.min(unnormalized)) / (
                torch.max(unnormalized) - torch.min(unnormalized)
            )

            self.assertTrue(torch.allclose(w1, correct_w1))
            self.assertTrue(torch.allclose(w2, correct_w2))

            w1_s = fn1_s(softmax(x, dim=1))
            w2_s = fn2_s(softmax(x, dim=1))

            self.assertTrue(torch.allclose(w1, w1_s, rtol=1e-2))
            self.assertTrue(torch.allclose(w2, w2_s, rtol=1e-2))
