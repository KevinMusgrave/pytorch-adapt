import torch
import torch.nn.functional as F
from pytorch_metric_learning.utils import common_functions as pml_cf

from ..utils import common_functions as c_f


# reference https://github.com/tim-learn/ATDOC/blob/main/demo_uda.py
class NeighborhoodAggregation(torch.nn.Module):
    def __init__(self, dataset_size, feature_dim, num_classes, k=5, T=0.5):
        super().__init__()
        self.register_buffer(
            "feat_memory", F.normalize(torch.rand(dataset_size, feature_dim))
        )
        self.register_buffer(
            "pred_memory", torch.ones(dataset_size, num_classes) / num_classes
        )
        self.k = k
        self.T = T

    def forward(self, features, logits=None, update=False, idx=None):
        # move to device if necessary
        self.feat_memory = pml_cf.to_device(self.feat_memory, features)
        self.pred_memory = pml_cf.to_device(self.pred_memory, features)
        with torch.no_grad():
            features = F.normalize(features)
            pseudo_labels, mean_logits = self.get_pseudo_labels(features, idx)
            if update:
                self.update_memory(features, logits, idx)
        return pseudo_labels, mean_logits

    def get_pseudo_labels(self, normalized_features, idx):
        dis = torch.mm(normalized_features, self.feat_memory.t())
        # set self-comparisons to min similarity
        for di in range(dis.size(0)):
            dis[di, idx[di]] = torch.min(dis)
        _, indices = torch.topk(dis, k=self.k, dim=1)
        logits = torch.mean(self.pred_memory[indices], dim=1)
        pseudo_labels = torch.argmax(logits, dim=1)
        return pseudo_labels, logits

    def update_memory(self, normalized_features, logits, idx):
        logits = F.softmax(logits, dim=1)
        p = 1.0 / self.T
        logits = (logits ** p) / torch.sum(logits ** p, dim=1, keepdims=True)
        self.feat_memory[idx] = normalized_features
        self.pred_memory[idx] = logits

    def extra_repr(self):
        return c_f.extra_repr(self, ["k", "T"])
