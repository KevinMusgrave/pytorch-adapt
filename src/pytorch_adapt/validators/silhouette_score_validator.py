import torch
from pytorch_metric_learning.utils import common_functions as pml_cf

from ..layers import SilhouetteScore
from .base_validator import BaseValidator


class SilhouetteScoreValidator(BaseValidator):
    def __init__(self, layer="features", **kwargs):
        super().__init__(**kwargs)
        self.layer = layer
        self.silhouette_fn = SilhouetteScore()

    def compute_score(self, target_train):
        return get_silhouette_score(
            target_train[self.layer],
            target_train["logits"],
            silhouette_fn=self.silhouette_fn,
        )


### https://github.com/lr94/abas/blob/master/model_selection.py ###
def get_centroids(data, labels, num_classes):
    centroids = torch.zeros((num_classes, data.shape[1]), device=data.device)
    for cid in range(num_classes):
        matches = labels == cid
        if torch.any(matches):
            centroids[cid] = torch.mean(data[matches], dim=0)

    return centroids


def get_silhouette_score(feats, logits, silhouette_fn):
    plabels = torch.argmax(logits, dim=1)
    num_classes = logits.shape[1]
    centroids = get_centroids(feats, plabels, num_classes)
    clabels = get_kmeans(feats, num_classes, centroids)
    return silhouette_fn(feats, clabels)


def get_kmeans(x, num_classes, init_centroids):
    import faiss

    device = x.device
    x = pml_cf.to_numpy(x)
    init_centroids = pml_cf.to_numpy(init_centroids)
    km = faiss.Kmeans(x.shape[1], num_classes, gpu=True)
    km.train(x, init_centroids=init_centroids)
    _, I = km.index.search(x, 1)
    I = torch.from_numpy(I).squeeze(1)
    return pml_cf.to_device(I, device=device)
