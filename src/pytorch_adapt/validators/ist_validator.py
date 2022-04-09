import torch
from pytorch_metric_learning.distances import BatchedDistance, CosineSimilarity

from ..layers import DiversityLoss, EntropyLoss
from ..layers.ist_loss import get_loss, get_probs
from ..utils.common_functions import mask_out_self
from .base_validator import BaseValidator


def get_iter_fn(probs, y, dist_is_inverted):
    def fn(mat, s, *_):
        mat, mask = mask_out_self(mat, s, return_mask=True)
        p = get_probs(mat, mask, y, dist_is_inverted)
        probs.append(p)

    return fn


class ISTValidator(BaseValidator):
    def __init__(
        self, layer="features", batch_size=1024, with_ent=True, with_div=True, **kwargs
    ):
        super().__init__(**kwargs)
        self.layer = layer
        self.dist_fn = BatchedDistance(CosineSimilarity(), batch_size=batch_size)
        self.with_ent = with_ent
        self.with_div = with_div
        self.ent_fn = EntropyLoss(after_softmax=True)
        self.div_fn = DiversityLoss(after_softmax=True)

    def compute_score(self, src_train, target_train):
        features = torch.cat([src_train[self.layer], target_train[self.layer]], dim=0)
        labels = torch.cat([src_train["domain"], target_train["domain"]], dim=0)

        # probs is modified via self.iter_fn
        probs = []
        self.dist_fn.iter_fn = get_iter_fn(
            probs, labels, self.dist_fn.distance.is_inverted
        )
        self.dist_fn(features)
        probs = torch.cat(probs, dim=0)
        if len(probs) != len(features):
            raise ValueError("probs should have same length as features")
        return get_loss(probs, self.ent_fn, self.div_fn, self.with_ent, self.with_div)
