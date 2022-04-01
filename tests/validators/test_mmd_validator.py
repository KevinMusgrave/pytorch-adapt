import random
import unittest

import numpy as np
import torch

from pytorch_adapt.layers import MMDLoss, utils
from pytorch_adapt.validators import MMDValidator

from .. import TEST_DEVICE


class TestMMDValidator(unittest.TestCase):
    def test_mmd_validator(self):
        torch.manual_seed(3531)
        embedding_size = 128
        for kernel_scales in [1, utils.get_kernel_scales()]:
            for src_dataset_size in [100, 500]:
                for target_dataset_size in [100, 500]:
                    for batch_size in [32, 128, 256, 512]:
                        for bandwidth in [None, 1]:
                            loss_fn = MMDLoss(
                                kernel_scales=kernel_scales,
                                bandwidth=bandwidth,
                                mmd_type="quadratic",
                            )
                            validator = MMDValidator(
                                batch_size=batch_size,
                                mmd_kwargs={
                                    "kernel_scales": kernel_scales,
                                    "mmd_type": "quadratic",
                                    "bandwidth": bandwidth,
                                },
                            )
                            src_features = torch.randn(
                                src_dataset_size, embedding_size
                            ).to(TEST_DEVICE)
                            target_features = torch.randn(
                                target_dataset_size, embedding_size
                            ).to(TEST_DEVICE)
                            target_offset = random.uniform(0.5, 2)
                            target_features += target_offset

                            score = validator(
                                src_train={"features": src_features},
                                target_train={"features": target_features},
                            )
                            correct_score = -loss_fn(
                                src_features, target_features
                            ).item()
                            rtol = 1e-2 if bandwidth is None else 1e-6
                            self.assertTrue(np.isclose(score, correct_score, rtol=rtol))
