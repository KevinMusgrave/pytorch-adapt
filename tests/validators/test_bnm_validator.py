import unittest

import torch

from pytorch_adapt.layers import BNMLoss
from pytorch_adapt.validators import BNMValidator

from .. import TEST_DEVICE


class TestBNMValidator(unittest.TestCase):
    def test_bnm_validator(self):
        v = BNMValidator()
        logits = torch.randn(1000, 13, device=TEST_DEVICE)
        score = v(target_train={"logits": logits})
        self.assertEqual(score, -BNMLoss()(logits))
