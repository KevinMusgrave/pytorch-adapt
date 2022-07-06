import unittest

import torch

from pytorch_adapt.validators import NearestSourceValidator

from .. import TEST_DEVICE


class TestNearestSourceValidator(unittest.TestCase):
    def test_nearest_source_validator(self):
        src_size, target_size = 1000, 2000
        emb_size, num_classes = 128, 5
        for layer in ["features", "preds"]:
            for threshold in [-2, -0.5, 0, 0.5, 1]:
                for weighted in [False, True]:
                    validator = NearestSourceValidator(
                        layer=layer, threshold=threshold, weighted=weighted
                    )
                    src_preds = torch.randn(src_size, num_classes, device=TEST_DEVICE)
                    src_labels = torch.randint(
                        0, num_classes, size=(src_size,), device=TEST_DEVICE
                    )
                    if layer == "preds":
                        src_emb = src_preds
                        target_emb = torch.randn(
                            target_size, num_classes, device=TEST_DEVICE
                        )
                    else:
                        src_emb = torch.randn(src_size, emb_size, device=TEST_DEVICE)
                        target_emb = torch.randn(
                            target_size, emb_size, device=TEST_DEVICE
                        )
                    src_val = {layer: src_emb, "preds": src_preds, "labels": src_labels}
                    target_train = {layer: target_emb}
                    score = validator(src_val=src_val, target_train=target_train)
                    # TODO: actually test that the score is correct
