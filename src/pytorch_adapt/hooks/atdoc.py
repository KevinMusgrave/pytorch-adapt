import torch

from ..layers import ConfidenceWeights, NeighborhoodAggregation
from ..utils import common_functions as c_f
from .base import BaseHook
from .features import FeaturesAndLogitsHook


class ATDOCHook(BaseHook):
    def __init__(
        self, dataset_size, feature_dim, num_classes, k=5, loss_fn=None, **kwargs
    ):
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
        outputs = self.hook(losses, inputs)[1]
        [features, logits] = c_f.extract(
            [outputs, inputs],
            c_f.filter(self.hook.out_keys, "", ["_features$", "_logits$"]),
        )
        pseudo_labels, neighbor_logits = self.labeler(
            features, logits, update=True, idx=inputs["target_sample_idx"]
        )
        loss = self.loss_fn(logits, pseudo_labels)
        weights = self.weighter(neighbor_logits)
        loss = torch.mean(weights * loss)
        return {"pseudo_label_loss": loss}, outputs

    def _loss_keys(self):
        return ["pseudo_label_loss"]

    def _out_keys(self):
        return self.hook.out_keys
