import random
import unittest

import numpy as np
import torch

from pytorch_adapt.layers import MMDLoss, utils
from pytorch_adapt.validators import MMDValidator
from pytorch_adapt.validators.mmd_validator import randomly_sample

from .. import TEST_DEVICE


class TestMMDValidator(unittest.TestCase):
    def test_mmd_validator(self):
        torch.manual_seed(3531)
        embedding_size = 128
        for kernel_scales in [1, utils.get_kernel_scales()]:
            for src_dataset_size in [100, 500]:
                for target_dataset_size in [100, 500]:
                    for num_samples in ["max", 128, 256, 512]:
                        max_dataset_size = max(src_dataset_size, target_dataset_size)
                        num_trials = 200
                        loss_fn = MMDLoss(kernel_scales=kernel_scales)
                        validator = MMDValidator(
                            num_samples=num_samples,
                            num_trials=num_trials,
                            mmd_kwargs={"kernel_scales": kernel_scales},
                        )
                        src_features = torch.randn(src_dataset_size, embedding_size).to(
                            TEST_DEVICE
                        )
                        target_features = torch.randn(
                            target_dataset_size, embedding_size
                        ).to(TEST_DEVICE)
                        target_offset = random.uniform(0.5, 2)
                        target_features += target_offset

                        score = validator(
                            src_train={"features": src_features},
                            target_train={"features": target_features},
                        )

                        _num_samples = (
                            max_dataset_size if num_samples == "max" else num_samples
                        )
                        _src_features = randomly_sample(src_features, _num_samples)
                        _target_features = randomly_sample(
                            target_features, _num_samples
                        )
                        self.assertTrue(
                            len(_src_features) == len(_target_features) == _num_samples
                        )
                        if src_dataset_size == target_dataset_size:
                            correct_score = -loss_fn(
                                src_features, target_features
                            ).item()
                            if num_samples == "max":
                                self.assertTrue(score == correct_score)
                            elif src_dataset_size > 100:
                                self.assertTrue(
                                    np.isclose(score, correct_score, rtol=0.1)
                                )
