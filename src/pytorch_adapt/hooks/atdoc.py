import torch

from ..layers import ConfidenceWeights, NeighborhoodAggregation
from ..utils import common_functions as c_f
from .base import BaseHook
from .features import FeaturesAndLogitsHook


class ATDOCHook(BaseHook):
    """
    Creates pseudo labels for the target domain
    using k-nearest neighbors. Then computes a
    classification loss based on these pseudo labels.

    Implementation of
    [Domain Adaptation with Auxiliary Target Domain-Oriented Classifier](https://arxiv.org/abs/2007.04171).
    """

    def __init__(
        self, dataset_size, feature_dim, num_classes, k=5, loss_fn=None, **kwargs
    ):
        """
        Arguments:
            dataset_size: The number of samples in the target dataset.
            feature_dim: The feature dimensionality, i.e at each iteration
                the features should be size ```(N, D)``` where N is batch size and
                D is ```feature_dim```.
            num_classes: The number of class labels in the target dataset.
            k: The number of nearest neighbors used to determine each
                sample's pseudolabel
            loss_fn: The classification loss function.
                If ```None``` it defaults to
                ```torch.nn.CrossEntropyLoss```.
        """
        super().__init__(**kwargs)
        self.labeler = NeighborhoodAggregation(
            dataset_size, feature_dim, num_classes, k=k
        )
        self.weighter = ConfidenceWeights()
        self.loss_fn = c_f.default(
            loss_fn, torch.nn.CrossEntropyLoss, {"reduction": "none"}
        )
        self.hook = FeaturesAndLogitsHook(domains=["target"])

    def call(self, losses, inputs):
        """"""
        outputs = self.hook(losses, inputs)[1]
        [features, logits] = c_f.extract(
            [outputs, inputs],
            c_f.filter(self.hook.out_keys, "", ["_features$", "_logits$"]),
        )
        pseudo_labels, neighbor_preds = self.labeler(
            features, logits, update=True, idx=inputs["target_sample_idx"]
        )
        loss = self.loss_fn(logits, pseudo_labels)
        weights = self.weighter(neighbor_preds)
        loss = torch.mean(weights * loss)
        return {"pseudo_label_loss": loss}, outputs

    def _loss_keys(self):
        """"""
        return ["pseudo_label_loss"]

    def _out_keys(self):
        """"""
        return self.hook.out_keys
