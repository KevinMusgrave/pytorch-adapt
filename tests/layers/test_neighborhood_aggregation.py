import unittest

import torch

from pytorch_adapt.layers import ConfidenceWeights, NeighborhoodAggregation

from .. import TEST_DEVICE


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


# from https://github.com/tim-learn/ATDOC/blob/main/demo_uda.py
def get_correct_memory_update(
    features_target, outputs_target, mem_fea, mem_cls, idx, momentum=1
):
    features_target = features_target / torch.norm(
        features_target, p=2, dim=1, keepdim=True
    )
    softmax_out = torch.nn.Softmax(dim=1)(outputs_target)

    # https://github.com/tim-learn/ATDOC/issues/1
    # https://github.com/KevinMusgrave/pytorch-adapt/issues/10
    outputs_target = softmax_out ** 2 / ((softmax_out ** 2).sum(dim=0))

    mem_fea[idx] = (1.0 - momentum) * mem_fea[idx] + momentum * features_target.clone()
    mem_cls[idx] = (1.0 - momentum) * mem_cls[idx] + momentum * outputs_target.clone()
    return mem_fea, mem_cls


class TestNeighborhoodAggregation(unittest.TestCase):
    def test_neighborhood_aggregation(self):
        dataset_size = 100000
        feature_dim = 2048
        num_classes = 123
        for k in [5, 11, 29]:
            na = NeighborhoodAggregation(
                dataset_size, feature_dim, num_classes, k=k
            ).to(TEST_DEVICE)
            weighter = ConfidenceWeights()

            batch_size = 64

            for i in range(10):
                features = torch.randn(batch_size, feature_dim, device=TEST_DEVICE)
                logits = torch.randn(batch_size, num_classes, device=TEST_DEVICE)
                idx = torch.randint(0, dataset_size, size=(64,))

                curr_feat_memory = na.feat_memory.clone()
                curr_pred_memory = na.pred_memory.clone()

                pseudo_labels, neighbor_logits = na(
                    features, logits, update=True, idx=idx
                )
                weights = weighter(neighbor_logits)

                self.assertTrue(not torch.allclose(curr_feat_memory, na.feat_memory))
                self.assertTrue(not torch.allclose(curr_pred_memory, na.pred_memory))

                correct_weights, correct_pseudo_labels = get_correct_pseudo_labels(
                    features, curr_feat_memory, curr_pred_memory, idx, k
                )

                self.assertTrue(torch.equal(pseudo_labels, correct_pseudo_labels))
                self.assertTrue(torch.allclose(weights, correct_weights))

                curr_feat_memory, curr_pred_memory = get_correct_memory_update(
                    features, logits, curr_feat_memory, curr_pred_memory, idx
                )

                self.assertTrue(torch.allclose(curr_feat_memory, na.feat_memory))
                self.assertTrue(torch.allclose(curr_pred_memory, na.pred_memory))
