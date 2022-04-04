from collections import defaultdict

import torch
from pytorch_metric_learning.utils.accuracy_calculator import (
    AccuracyCalculator,
    maybe_get_avg_of_avgs,
    try_getting_not_lone_labels,
    zero_accuracy,
)

from ..utils import common_functions as c_f
from . import utils as val_utils
from .base_validator import BaseValidator


# temporary
def mean_average_precision(
    knn_labels,
    gt_labels,
    embeddings_come_from_same_source,
    avg_of_avgs,
    return_per_class,
    label_comparison_fn,
    relevance_mask=None,
    at_r=False,
):
    device = gt_labels.device
    num_samples, num_k = knn_labels.shape[:2]
    relevance_mask = (
        torch.ones((num_samples, num_k), dtype=torch.bool, device=device)
        if relevance_mask is None
        else relevance_mask
    )
    is_same_label = label_comparison_fn(gt_labels, knn_labels)
    equality = is_same_label * relevance_mask
    cumulative_correct = torch.cumsum(equality, dim=1)
    k_idx = torch.arange(1, num_k + 1, device=device).repeat(num_samples, 1)
    precision_at_ks = (cumulative_correct * equality).type(torch.float64) / k_idx
    summed_precision_per_row = torch.sum(precision_at_ks * relevance_mask, dim=1)
    if at_r:
        max_possible_matches_per_row = torch.sum(relevance_mask, dim=1)
    else:
        max_possible_matches_per_row = torch.sum(equality, dim=1)
        max_possible_matches_per_row[max_possible_matches_per_row == 0] = 1
    accuracy_per_sample = summed_precision_per_row / max_possible_matches_per_row
    return accuracy_per_sample


# temporary, eventaully move to PML
class BatchedAccuracyCalculator(AccuracyCalculator):
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def _get_accuracy(self, function_dict, **kwargs):
        output = defaultdict(list)

        for s in range(0, len(kwargs["knn_labels"]), self.batch_size):
            curr_kwargs = {
                k: kwargs[k][s : s + self.batch_size]
                for k in ["knn_labels", "query_labels", "not_lone_query_mask"]
            }
            curr_kwargs["embeddings_come_from_same_source"] = kwargs[
                "embeddings_come_from_same_source"
            ]
            curr_kwargs["label_counts"] = kwargs["label_counts"]
            curr_results = {k: v(**curr_kwargs) for k, v in function_dict.items()}
            for k, v in curr_results.items():
                output[k].append(v)

        for k, v in output.items():
            output[k] = maybe_get_avg_of_avgs(
                torch.cat(v, dim=0),
                kwargs["query_labels"],
                self.avg_of_avgs,
                self.return_per_class,
            )
        return output

    def calculate_mean_average_precision(
        self,
        knn_labels,
        query_labels,
        not_lone_query_mask,
        embeddings_come_from_same_source,
        label_counts,
        **kwargs,
    ):
        knn_labels, query_labels = try_getting_not_lone_labels(
            knn_labels, query_labels, not_lone_query_mask
        )
        if knn_labels is None:
            return zero_accuracy(label_counts[0], self.return_per_class)

        return mean_average_precision(
            knn_labels,
            query_labels[:, None],
            embeddings_come_from_same_source,
            self.avg_of_avgs,
            self.return_per_class,
            self.label_comparison_fn,
        )


class KNNValidator(BaseValidator):
    def __init__(
        self,
        layer="features",
        metric="precision_at_1",
        k=1,
        knn_func=None,
        kmeans_func=None,
        batch_size=None,
        device=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        acc_kwargs = {
            "k": k,
            "include": (metric,),
            "avg_of_avgs": True,
            "knn_func": knn_func,
            "kmeans_func": kmeans_func,
            "device": device,
        }
        if batch_size is not None:
            acc_kwargs["batch_size"] = batch_size
            acc_fn = BatchedAccuracyCalculator
        else:
            acc_fn = AccuracyCalculator
        self.acc_fn = acc_fn(**acc_kwargs)
        self.layer = layer
        self.metric = metric

    def compute_score(self, src_train, target_train):
        features = torch.cat([src_train[self.layer], target_train[self.layer]], dim=0)
        labels = torch.cat([src_train["domain"], target_train["domain"]], dim=0)
        accuracies = self.acc_fn.get_accuracy(features, features, labels, labels, True)
        return -accuracies[self.metric]


class ClusterValidator(KNNValidator):
    def __init__(self, src_label_fn=None, target_label_fn=None, metric="AMI", **kwargs):
        if metric not in ["AMI", "NMI"]:
            raise ValueError("metric must be 'AMI' or 'NMI'")
        super().__init__(metric=metric, **kwargs)
        self.src_label_fn = c_f.default(src_label_fn, val_utils.src_label_fn)
        self.target_label_fn = c_f.default(target_label_fn, val_utils.target_label_fn)

    def compute_score(self, src_train, target_train):
        query = torch.cat([src_train[self.layer], target_train[self.layer]], dim=0)
        labels = torch.cat(
            [self.src_label_fn(src_train), self.target_label_fn(target_train)], dim=0
        )

        acc = self.acc_fn.get_accuracy(query, query, labels, labels, True)
        return acc[self.metric]
