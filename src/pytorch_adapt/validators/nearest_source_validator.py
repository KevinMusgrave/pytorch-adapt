import torch
from pytorch_metric_learning.distances import CosineSimilarity, LpDistance
from pytorch_metric_learning.utils.inference import CustomKNN

from .base_validator import BaseValidator


def acc(preds, labels):
    if max(labels) != preds.shape[1] - 1:
        raise ValueError(
            f"Max label {max(labels)} should be equal to preds.shape[1] {preds.shape[1]}"
        )
    preds = torch.argmax(preds, dim=1)
    return (preds == labels).float()


class NearestSourceValidator(BaseValidator):
    def __init__(self, layer="preds", threshold=0, weighted=False, **kwargs):
        super().__init__(**kwargs)
        self.layer = layer
        self.threshold = threshold
        self.weighted = weighted
        self.knn_fn = CustomKNN(CosineSimilarity())

    def compute_score(self, src_val, target_train):
        nearest_src_acc, sims = self.get_nearest_src_acc(src_val, target_train)

        if self.weighted:
            sims = (sims - self.threshold) / (max(sims) - self.threshold)
            sims[sims <= 0] = 0
            nearest_src_acc *= sims
        else:
            nearest_src_acc[sims <= self.threshold] = 0

        return torch.mean(nearest_src_acc).item()

    def get_nearest_src_acc(self, src_val, target_train):
        src_acc = acc(src_val["preds"], src_val["labels"])

        sims, idx = self.knn_fn(
            target_train[self.layer],
            k=1,
            reference=src_val[self.layer],
            embeddings_come_from_same_source=False,
        )
        sims, idx = sims.squeeze(1), idx.squeeze(1)
        nearest_src_acc = src_acc[idx]
        return nearest_src_acc, sims


class NearestSourceL2Validator(NearestSourceValidator):
    def __init__(self, layer="preds", **kwargs):
        super().__init__(layer=layer, threshold=float("inf"), weighted=True, **kwargs)
        self.knn_fn = CustomKNN(LpDistance(normalize_embeddings=False))

    def compute_score(self, src_val, target_train):
        nearest_src_acc, dists = self.get_nearest_src_acc(src_val, target_train)
        dists = (dists - min(dists)) / (max(dists) - min(dists))
        nearest_src_acc *= 1 - dists
        return torch.mean(nearest_src_acc).item()
