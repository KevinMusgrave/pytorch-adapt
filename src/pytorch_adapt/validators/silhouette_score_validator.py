import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score

from .base_validator import BaseValidator


class SilhouetteScoreValidator(BaseValidator):
    def __init__(self, layer="features", with_src=False, pca_size=64, **kwargs):
        super().__init__(**kwargs)
        self.layer = layer
        self.with_src = with_src
        self.pca_size = pca_size
        self.score_fn = silhouette_score

    def _required_data(self):
        x = ["target_train"]
        if self.with_src:
            x.append("src_train")
        return x

    def compute_score(self, target_train, src_train=None):
        logits = target_train["logits"]
        feats = target_train[self.layer]
        plabels = torch.argmax(logits, dim=1)
        num_classes = logits.shape[1]
        source_feats = src_train[self.layer].cpu().numpy() if self.with_src else None

        return get_clustering_performance(
            feats.cpu().numpy(),
            plabels.cpu().numpy(),
            num_classes,
            self.score_fn,
            source_feats=source_feats,
            pca_size=self.pca_size,
        )


class CHScoreValidator(SilhouetteScoreValidator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_fn = calinski_harabasz_score


# copied from https://github.com/lr94/abas/blob/master/model_selection.py
def get_centroids(data, labels, num_classes):
    centroids = np.zeros((num_classes, data.shape[1]))
    for cid in range(num_classes):
        # Since we are using pseudolabels to compute centroids, some classes might not have instances according to the
        # pseudolabels assigned by the current model. In that case .mean() would return NaN causing KMeans to fail.
        # We set to 0 the missing centroids
        if (labels == cid).any():
            centroids[cid] = data[labels == cid].mean(0)

    return centroids


# copied and modified from https://github.com/lr94/abas/blob/master/model_selection.py
def get_clustering_performance(
    feats, plabels, num_classes, score_fn, source_feats=None, pca_size=64
):
    """
    :param feats: N x out numpy vector
    :param plabels:  N numpy vector
    :param num_classes: int
    :param source_feats
    :param pca_size
    :return: silhouette and calinski harabasz scores
    """
    if pca_size is not None:
        pca = PCA(pca_size)
        n_samples = feats.shape[0]
        if source_feats is not None:
            feats = np.concatenate((feats, source_feats))
        x = pca.fit_transform(feats)[:n_samples]
    else:
        x = feats

    try:
        centroids = get_centroids(x, plabels, num_classes)
        clustering = KMeans(n_clusters=num_classes, init=centroids, n_init=1)
        clustering.fit(x)
        clabels = clustering.labels_
        score = score_fn(x, clabels)
    except ValueError:
        score = float("nan")

    return score


# class SilhouetteScoreValidator(BaseValidator):
#     def __init__(self, layer="features", **kwargs):
#         super().__init__(**kwargs)
#         self.layer = layer
#         self.silhouette_fn = SilhouetteScore()

#     def compute_score(self, target_train):
#         return get_silhouette_score(
#             target_train[self.layer],
#             target_train["logits"],
#             silhouette_fn=self.silhouette_fn,
#         )


# ### https://github.com/lr94/abas/blob/master/model_selection.py ###
# def get_centroids(data, labels, num_classes):
#     centroids = torch.zeros((num_classes, data.shape[1]), device=data.device)
#     for cid in range(num_classes):
#         matches = labels == cid
#         if torch.any(matches):
#             centroids[cid] = torch.mean(data[matches], dim=0)

#     return centroids


# def get_silhouette_score(feats, logits, silhouette_fn):
#     plabels = torch.argmax(logits, dim=1)
#     num_classes = logits.shape[1]
#     centroids = get_centroids(feats, plabels, num_classes)
#     clabels = get_kmeans(feats, num_classes, centroids)
#     return silhouette_fn(feats, clabels)


# def get_kmeans(x, num_classes, init_centroids):
#     import faiss

#     device = x.device
#     x = pml_cf.to_numpy(x)
#     init_centroids = pml_cf.to_numpy(init_centroids)
#     km = faiss.Kmeans(x.shape[1], num_classes, gpu=True)
#     km.train(x, init_centroids=init_centroids)
#     _, I = km.index.search(x, 1)
#     I = torch.from_numpy(I).squeeze(1)
#     return pml_cf.to_device(I, device=device)
