import unittest

import torch

from pytorch_adapt.validators import (
    DeepEmbeddedValidator,
    DiversityValidator,
    EntropyValidator,
    MultipleValidators,
)

from .. import TEST_FOLDER


class TestKeyMap(unittest.TestCase):
    def test_key_map(self):
        validator = MultipleValidators(
            validators={
                "entropy": EntropyValidator(),
                "diversity": DiversityValidator(),
            },
        )
        self.assertTrue(validator.required_data == ["target_train"])

        validator = MultipleValidators(
            validators={
                "entropy": EntropyValidator(),
                "diversity": DiversityValidator(),
            },
            key_map={"baaa": "target_train"},
        )
        self.assertTrue(validator.required_data == ["baaa"])

        validator.score(epoch=0, baaa={"logits": torch.randn(100, 100)})

        with self.assertRaises(ValueError):
            validator.score(epoch=0, target_train={"logits": torch.randn(100, 100)})

        validator = MultipleValidators(
            validators={
                "entropy": EntropyValidator(key_map={"goo": "target_train"}),
                "diversity": DiversityValidator(),
            },
            key_map={"baaa": "target_train"},
        )
        self.assertTrue(validator.required_data == ["goo", "baaa"])
        validator.score(
            epoch=0,
            baaa={"logits": torch.randn(100, 100)},
            goo={"logits": torch.randn(100, 100)},
        )

        with self.assertRaises(ValueError):
            validator.score(epoch=0, baaa={"logits": torch.randn(100, 100)})

        validator = MultipleValidators(
            validators={
                "entropy": EntropyValidator(key_map={"goo": "target_train"}),
                "diversity": DiversityValidator(),
                "dev": DeepEmbeddedValidator(temp_folder=TEST_FOLDER),
            },
            key_map={"baaa": "target_train"},
        )

        self.assertTrue(
            set(validator.required_data) == set(["goo", "baaa", "src_train", "src_val"])
        )
