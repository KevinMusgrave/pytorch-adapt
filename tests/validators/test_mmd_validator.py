import random
import unittest

import numpy as np
import torch

from pytorch_adapt.layers import MMDLoss, utils
from pytorch_adapt.validators import MMDValidator

from .. import TEST_DEVICE


class TestMMDValidator(unittest.TestCase):
    def test_mmd_validator(self):
        embedding_size = 128
        for kernel_scales in [1, utils.get_kernel_scales()]:
            for dataset_size in [100, 1000, 2000]:
                for num_samples in ["max", 128, 256, 512]:
                    num_trials = (
                        1 if num_samples == "max" or num_samples > dataset_size else 100
                    )
                    loss_fn = MMDLoss(kernel_scales=kernel_scales)
                    validator = MMDValidator(
                        num_samples=num_samples,
                        num_trials=num_trials,
                        mmd_kwargs={"kernel_scales": kernel_scales},
                    )
                    for _ in range(10):
                        src_features = torch.randn(dataset_size, embedding_size).to(
                            TEST_DEVICE
                        )
                        target_features = torch.randn(dataset_size, embedding_size).to(
                            TEST_DEVICE
                        )
                        target_offset = random.uniform(0.5, 2)
                        target_features += target_offset

                        score = validator(
                            src_train={"features": src_features},
                            target_train={"features": target_features},
                        )
                        correct_score = -loss_fn(src_features, target_features).item()

                        if num_samples == "max" or num_samples > dataset_size:
                            self.assertTrue(score == correct_score)
                        else:
                            self.assertTrue(np.isclose(score, correct_score, rtol=0.1))
