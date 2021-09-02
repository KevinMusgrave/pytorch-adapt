import copy
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_adapt.hooks import ATDOCHook
from pytorch_adapt.layers import ConfidenceWeights, NeighborhoodAggregation
from pytorch_adapt.utils import common_functions as c_f

from .utils import Net, assertRequiresGrad


def get_data(dataset_size, feature_dim, num_classes, batch_size):
    target_imgs_features = torch.randn(batch_size, feature_dim)
    target_imgs_features_logits = torch.randn(batch_size, num_classes)
    target_sample_idx = torch.randint(0, dataset_size, size=(batch_size,))

    return {
        "target_imgs_features": target_imgs_features,
        "target_imgs_features_logits": target_imgs_features_logits,
        "target_sample_idx": target_sample_idx,
    }


class TestATDOC(unittest.TestCase):
    def test_atdoc_hook(self):
        torch.manual_seed(4308)
        dataset_size = 10000
        feature_dim = 128
        num_classes = 10
        batch_size = 64
        iters = 10

        seed = 545
        torch.manual_seed(seed)
        h = ATDOCHook(dataset_size, feature_dim, num_classes)
        all_losses = []
        for i in range(iters):
            data = get_data(dataset_size, feature_dim, num_classes, batch_size)
            losses, outputs = h({}, data)
            all_losses.append(losses["pseudo_label_loss"])

        torch.manual_seed(seed)
        na = NeighborhoodAggregation(dataset_size, feature_dim, num_classes)
        all_correct_losses = []
        for i in range(iters):
            data = get_data(dataset_size, feature_dim, num_classes, batch_size)
            pseudo_labels, neighbor_logits = na(
                data["target_imgs_features"],
                data["target_imgs_features_logits"],
                update=True,
                idx=data["target_sample_idx"],
            )
            loss = F.cross_entropy(
                data["target_imgs_features_logits"], pseudo_labels, reduction="none"
            )
            loss = torch.mean(loss * ConfidenceWeights()(neighbor_logits))
            all_correct_losses.append(loss)

        self.assertTrue(
            all(
                np.isclose(x, y.item(), rtol=1e-6)
                for x, y in zip(all_losses, all_correct_losses)
            )
        )
