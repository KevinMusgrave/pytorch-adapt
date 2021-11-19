import torch
import torch.nn.functional as F
from pytorch_metric_learning.distances import CosineSimilarity

from ..layers import EntropyLoss
from .base_validator import BaseValidator


# https://arxiv.org/pdf/2108.10860.pdf
class SNDValidator(BaseValidator):
    """
    Implementation of
    [Tune it the Right Way: Unsupervised Validation of Domain Adaptation via Soft Neighborhood Density](https://arxiv.org/abs/2108.10860)
    """

    def __init__(self, layer="preds", T=0.05, batch_size=1000, **kwargs):
        super().__init__(**kwargs)
        self.layer = layer
        self.T = T
        self.dist_fn = CosineSimilarity()
        self.entropy_fn = EntropyLoss(after_softmax=True, return_mean=False)
        self.batch_size = batch_size

    def compute_score(self, target_train):
        features = target_train[self.layer]
        n = features.shape[0]
        all_entropies = []
        for s in range(0, n, self.batch_size):
            e = s + self.batch_size
            L = features[s:e]
            sim_mat = self.dist_fn(L, features)
            sim_mat = self.mask_out_self(sim_mat, s)
            sim_mat = F.softmax(sim_mat / self.T, dim=1)
            all_entropies.append(self.entropy_fn(sim_mat))
        all_entropies = torch.cat(all_entropies, dim=0)
        if len(all_entropies) != len(features):
            raise ValueError("all_entropies should have same length as input features")
        return torch.mean(all_entropies).item()

    def mask_out_self(self, sim_mat, start_idx):
        num_rows, num_cols = sim_mat.shape
        mask = torch.ones(num_rows, num_cols, dtype=torch.bool)
        rows = torch.arange(num_rows)
        cols = rows + start_idx
        mask[rows, cols] = False
        return sim_mat[mask].view(num_rows, num_cols - 1)
