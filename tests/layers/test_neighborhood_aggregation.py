import unittest

import torch
import torch.nn.functional as F

from pytorch_adapt.layers import ConfidenceWeights, NeighborhoodAggregation

from .. import TEST_DEVICE, TEST_DTYPES


# from https://github.com/tim-learn/ATDOC/blob/main/demo_uda.py
def get_correct_pseudo_labels(features, feat_memory, pred_memory, idx, k):
    dis = -torch.mm(features, feat_memory.t())
    for di in range(dis.size(0)):
        dis[di, idx[di]] = torch.max(dis)
    _, p1 = torch.sort(dis, dim=1)

    w = torch.zeros(features.size(0), feat_memory.size(0)).cuda()
    for wi in range(w.size(0)):
        for wj in range(k):
            w[wi][p1[wi, wj]] = 1 / k

    return torch.max(w.mm(pred_memory), 1)


class TestNeighborhoodAggregation(unittest.TestCase):
    def test_neighborhood_aggregation(self):
        dataset_size = 100000
        feature_dim = 2048
        num_classes = 123
        k = 5

        na = NeighborhoodAggregation(dataset_size, feature_dim, num_classes, k=k).to(
            TEST_DEVICE
        )
        weighter = ConfidenceWeights()

        batch_size = 64
        features = torch.randn(batch_size, feature_dim, device=TEST_DEVICE)
        logits = torch.randn(batch_size, num_classes, device=TEST_DEVICE)
        idx = torch.randint(0, dataset_size, size=(64,))
        for i in range(10):
            pseudo_labels, neighbor_logits = na(features, logits, update=True, idx=idx)
            weights = weighter(neighbor_logits)
            if i > 0:
                correct_weights, correct_pseudo_labels = get_correct_pseudo_labels(
                    features, na.feat_memory, na.pred_memory, idx, k
                )
                self.assertTrue(torch.equal(pseudo_labels, correct_pseudo_labels))
                self.assertTrue(torch.allclose(weights, correct_weights))
