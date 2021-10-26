import torch
import torch.nn.functional as F
from pytorch_metric_learning.distances import CosineSimilarity

from ..utils import common_functions as c_f
from .diversity_loss import DiversityLoss
from .entropy_loss import EntropyLoss


class ISTLoss(torch.nn.Module):
    """
    Implementation of the I_st loss from
    [Information-Theoretical Learning of Discriminative Clusters for Unsupervised Domain Adaptation](https://icml.cc/2012/papers/566.pdf)
    """

    def __init__(self, distance=None, with_div=True):
        super().__init__()
        self.distance = c_f.default(distance, CosineSimilarity, {})
        self.with_div = with_div
        self.ent_loss_fn = EntropyLoss(after_softmax=True)
        if self.with_div:
            self.div_loss_fn = DiversityLoss(after_softmax=True)

    def forward(self, x, y):
        """
        Arguments:
            x: source and target features
            y: domain labels, i.e. 0 for source domain, 1 for target domain
        """
        if torch.min(y) < 0 or torch.max(y) > 1:
            raise ValueError("y must be in the range 0 and 1")

        mat = self.distance(x)
        n = mat.shape[0]

        # remove self comparisons
        mask = torch.eye(n, dtype=torch.bool)
        mat = mat[~mask].view(n, n - 1)
        if not self.distance.is_inverted:
            mat *= -1
        mat = F.softmax(mat, dim=1)

        y = y.repeat(n, 1)[~mask].view(n, n - 1)

        domain_probs = mat * y
        target_probs = torch.sum(domain_probs, dim=1, keepdims=True)
        src_probs = 1 - target_probs
        probs = torch.cat([src_probs, target_probs], dim=1)

        ent_loss = self.ent_loss_fn(probs)

        if self.with_div:
            return -self.div_loss_fn(probs) - ent_loss
        return -ent_loss

    def extra_repr(self):
        """"""
        return c_f.extra_repr(self, ["with_div"])
