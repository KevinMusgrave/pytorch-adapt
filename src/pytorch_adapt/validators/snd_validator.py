import torch
import torch.nn.functional as F
from pytorch_metric_learning.distances import BatchedDistance, CosineSimilarity

from ..layers import EntropyLoss
from ..utils.common_functions import mask_out_self
from .base_validator import BaseValidator


def get_iter_fn(all_entropies, entropy_fn, T):
    def fn(sim_mat, s, *_):
        sim_mat = mask_out_self(sim_mat, s)
        sim_mat = F.softmax(sim_mat / T, dim=1)
        all_entropies.append(entropy_fn(sim_mat))

    return fn


# https://arxiv.org/pdf/2108.10860.pdf
class SNDValidator(BaseValidator):
    """
    Implementation of
    [Tune it the Right Way: Unsupervised Validation of Domain Adaptation via Soft Neighborhood Density](https://arxiv.org/abs/2108.10860)
    """

    def __init__(self, layer="preds", T=0.05, batch_size=1024, **kwargs):
        super().__init__(**kwargs)
        self.layer = layer
        self.T = T
        self.entropy_fn = EntropyLoss(after_softmax=True, return_mean=False)
        self.dist_fn = BatchedDistance(CosineSimilarity(), batch_size=batch_size)

    def compute_score(self, target_train):
        features = target_train[self.layer]
        # all_entropies is modified via self.iter_fn
        all_entropies = []
        self.dist_fn.iter_fn = get_iter_fn(all_entropies, self.entropy_fn, self.T)
        self.dist_fn(features)
        all_entropies = torch.cat(all_entropies, dim=0)
        if len(all_entropies) != len(features):
            raise ValueError("all_entropies should have same length as input features")
        return torch.mean(all_entropies).item()
