import unittest

import torch

from pytorch_adapt.layers import (
    EntropyWeights,
    MaxNormalizer,
    MinMaxNormalizer,
    SumNormalizer,
)
from pytorch_adapt.layers.entropy_loss import entropy, entropy_after_softmax

from .. import TEST_DEVICE, TEST_DTYPES


class TestEntropyWeights(unittest.TestCase):
    def test_entropy_weights(self):
        torch.manual_seed(340)
        softmax = torch.nn.functional.softmax
        fn1 = EntropyWeights(normalizer=SumNormalizer())
        fn2 = EntropyWeights(normalizer=MinMaxNormalizer())
        fn3 = EntropyWeights(normalizer=MaxNormalizer())
        fn1_s = EntropyWeights(after_softmax=True, normalizer=SumNormalizer())
        fn2_s = EntropyWeights(after_softmax=True, normalizer=MinMaxNormalizer())
        fn3_s = EntropyWeights(after_softmax=True, normalizer=MaxNormalizer())
        for dtype in TEST_DTYPES:
            batch_size = 32
            embedding_size = 100
            x = (
                torch.randn(batch_size, embedding_size, device=TEST_DEVICE, dtype=dtype)
                ** 3
            )
            w1 = fn1(x)
            w2 = fn2(x)
            w3 = fn3(x)

            unnormalized = 1 + torch.exp(-entropy(x))
            correct_w1 = unnormalized / torch.sum(unnormalized)
            correct_w2 = (unnormalized - torch.min(unnormalized)) / (
                torch.max(unnormalized) - torch.min(unnormalized)
            )
            correct_w3 = unnormalized / torch.max(unnormalized)

            self.assertTrue(torch.allclose(w1, correct_w1))
            self.assertTrue(torch.allclose(w2, correct_w2))
            self.assertTrue(torch.allclose(w3, correct_w3))

            w1_s = fn1_s(softmax(x, dim=1))
            w2_s = fn2_s(softmax(x, dim=1))
            w3_s = fn3_s(softmax(x, dim=1))

            unnormalized = 1 + torch.exp(-entropy_after_softmax(softmax(x, dim=1)))
            correct_w1 = unnormalized / torch.sum(unnormalized)
            correct_w2 = (unnormalized - torch.min(unnormalized)) / (
                torch.max(unnormalized) - torch.min(unnormalized)
            )
            correct_w3 = unnormalized / torch.max(unnormalized)

            self.assertTrue(torch.allclose(w1_s, correct_w1))
            self.assertTrue(torch.allclose(w2_s, correct_w2))
            self.assertTrue(torch.allclose(w3_s, correct_w3))
