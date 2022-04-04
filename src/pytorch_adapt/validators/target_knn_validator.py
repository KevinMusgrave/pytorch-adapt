import numpy as np
import torch

from ..utils import common_functions as c_f
from . import utils as val_utils
from .knn_validator import KNNValidator


def get_target_in_ref(query, ref, query_labels, ref_labels):
    data = []
    for L in torch.unique(query_labels):
        mask = query_labels == L
        curr_query = query[mask]
        curr_ref = torch.cat([ref, query[~mask]], dim=0)
        curr_query_labels = query_labels[mask]
        curr_ref_labels = torch.cat([ref_labels, query_labels[~mask]], dim=0)
        if len(curr_query) != len(curr_query_labels) or len(curr_ref) != len(
            curr_ref_labels
        ):
            raise ValueError("lengths should match")
        data.append((curr_query, curr_ref, curr_query_labels, curr_ref_labels))
    return data


class TargetKNNValidator(KNNValidator):
    def __init__(
        self, add_target_to_ref=False, src_label_fn=None, target_label_fn=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.add_target_to_ref = add_target_to_ref
        self.src_label_fn = c_f.default(src_label_fn, val_utils.src_label_fn)
        self.target_label_fn = c_f.default(target_label_fn, val_utils.target_label_fn)

    def compute_score(self, src_train, target_train):
        query = target_train[self.layer]
        query_labels = self.target_label_fn(target_train)
        ref = src_train[self.layer]
        ref_labels = self.src_label_fn(src_train)

        if self.add_target_to_ref:
            data = get_target_in_ref(query, ref, query_labels, ref_labels)
        else:
            data = [(query, ref, query_labels, ref_labels)]

        scores = []
        for d in data:
            accuracies = self.acc_fn.get_accuracy(
                *d, embeddings_come_from_same_source=False
            )
            scores.append(accuracies[self.metric])
        return np.mean(scores)


class TargetClusterValidator(KNNValidator):
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
