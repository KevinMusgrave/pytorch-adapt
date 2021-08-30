import unittest
from contextlib import nullcontext

import numpy as np
import torch
from sklearn.metrics import silhouette_score

from pytorch_adapt.layers import SilhouetteScore

from .. import TEST_DEVICE


class TestSilhouetteScore(unittest.TestCase):
    def test_silhouette_score(self):
        fn = SilhouetteScore()
        for embedding_size in [10, 100]:
            for num_embeddings in [1000, 10000]:
                for num_classes in [1, 12, 123, num_embeddings]:
                    features = torch.randn(
                        num_embeddings, embedding_size, device=TEST_DEVICE
                    )
                    if num_classes == num_embeddings:
                        labels = torch.arange(num_embeddings, device=TEST_DEVICE)
                    else:
                        labels = torch.randint(
                            0, num_classes, size=(num_embeddings,), device=TEST_DEVICE
                        )
                    context = (
                        self.assertRaises(ValueError)
                        if num_classes in [1, num_embeddings]
                        else nullcontext()
                    )
                    with context:
                        score = fn(features, labels)
                    with context:
                        correct_score = silhouette_score(features.cpu(), labels.cpu())
                    if isinstance(context, nullcontext):
                        self.assertTrue(np.isclose(score, correct_score, rtol=1e-2))
