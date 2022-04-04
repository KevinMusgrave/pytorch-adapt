import unittest

import torch

from pytorch_adapt.validators import SilhouetteScoreValidator

from .. import TEST_DEVICE


class TestSilhouetteScoreValidator(unittest.TestCase):
    def test_silhouette_score_validator(self):
        torch.cuda.empty_cache()
        for pca_size in [32, 64, None]:
            validator = SilhouetteScoreValidator(pca_size=pca_size)
            dataset_size = 10000
            num_classes = 2
            half = dataset_size // 2
            features = torch.ones(dataset_size, 64, device=TEST_DEVICE)
            features[:half] = 0
            logits = torch.zeros(dataset_size, num_classes, device=TEST_DEVICE)
            logits[:half, 0] = 1
            logits[half:, 1] = 1
            score = validator(target_train={"features": features, "logits": logits})
            self.assertTrue(score == 1)

            dataset_size = 10000
            num_classes = 100
            features = torch.randn(dataset_size, 64, device=TEST_DEVICE)
            logits = torch.randn(dataset_size, num_classes, device=TEST_DEVICE)
            score = validator(target_train={"features": features, "logits": logits})
            self.assertTrue(score < 0.03)
