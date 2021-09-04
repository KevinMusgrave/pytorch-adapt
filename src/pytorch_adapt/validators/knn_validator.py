import torch
import torch.nn.functional as F
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from .base_validator import BaseValidator


class KNNValidator(BaseValidator):
    def __init__(
        self,
        layer="features",
        l2_normalize=False,
        metric="precision_at_1",
        k=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.acc_fn = AccuracyCalculator(k=k, include=(metric,), avg_of_avgs=True)
        self.layer = layer
        self.l2_normalize = l2_normalize
        self.metric = metric
        self.k = k

    def compute_score(self, src_train, target_train):
        features = torch.cat([src_train[self.layer], target_train[self.layer]], dim=0)
        if self.l2_normalize:
            features = F.normalize(features, dim=1)
        labels = torch.cat([src_train["domain"], target_train["domain"]], dim=0)
        accuracies = self.acc_fn.get_accuracy(features, features, labels, labels, True)
        return -accuracies[self.metric]


class ClusterValidator(KNNValidator):
    def __init__(self, layer="features", metric="AMI", **kwargs):
        if metric not in ["AMI", "NMI"]:
            raise ValueError("metric must be 'AMI' or 'NMI'")
        super().__init__(layer=layer, metric=metric, **kwargs)
