import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score

from ..utils import common_functions as c_f
from . import utils as val_utils
from .base_validator import BaseValidator


def check_centroid_init(centroid_init):
    if centroid_init not in ["label_centers", None]:
        raise ValueError("centroid_init should be 'label_centers' or None")


class ClassClusterValidator(BaseValidator):
    def __init__(
        self,
        layer="features",
        src_label_fn=None,
        target_label_fn=None,
        with_src=False,
        src_for_pca=False,
        pca_size=None,
        centroid_init=None,
        score_fn=None,
        score_fn_type="labels",
        feat_normalizer=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.layer = layer
        self.src_label_fn = c_f.default(src_label_fn, val_utils.src_label_fn)
        self.target_label_fn = c_f.default(target_label_fn, val_utils.target_label_fn)
        self.with_src = with_src
        self.src_for_pca = src_for_pca
        self.pca_size = pca_size
        check_centroid_init(centroid_init)
        self.centroid_init = centroid_init
        self.score_fn = c_f.default(score_fn, adjusted_mutual_info_score)
        self.score_fn_type = score_fn_type
        self.feat_normalizer = feat_normalizer

    def _required_data(self):
        x = ["target_train"]
        if self.requires_src:
            x.append("src_train")
        return x

    @property
    def requires_src(self):
        return self.with_src or self.src_for_pca

    def compute_score(self, target_train, src_train=None):
        feats = target_train[self.layer]
        labels = self.target_label_fn(target_train)
        num_classes = target_train["logits"].shape[1]

        src_feats, src_labels = None, None
        if self.requires_src:
            src_feats = src_train[self.layer]
        if self.with_src:
            src_labels = src_train["labels"]

        return get_clustering_performance(
            feats,
            labels,
            num_classes,
            self.score_fn,
            self.score_fn_type,
            src_feats=src_feats,
            src_labels=src_labels,
            pca_size=self.pca_size,
            centroid_init=self.centroid_init,
            feat_normalizer=self.feat_normalizer,
        )


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


# adapted from https://github.com/lr94/abas/blob/master/model_selection.py
def get_clustering_performance(
    feats,
    labels,
    num_classes,
    score_fn,
    score_fn_type,
    src_feats=None,
    src_labels=None,
    pca_size=None,
    centroid_init=None,
    feat_normalizer=None,
):
    num_target_feats = feats.shape[0]

    if src_feats is not None:
        feats = torch.cat((feats, src_feats), dim=0)

    if feat_normalizer:
        feats = feat_normalizer(feats)

    feats = feats.cpu().numpy()

    if pca_size is not None:
        pca = PCA(pca_size)
        feats = pca.fit_transform(feats)

    # no labels means src_feats is just for pca
    if src_labels is None:
        feats = feats[:num_target_feats]
    else:
        labels = torch.cat((labels, src_labels), dim=0)

    labels = labels.cpu().numpy()

    if centroid_init == "label_centers":
        centroids = get_centroids(feats, labels, num_classes)
        clustering = KMeans(n_clusters=num_classes, init=centroids, n_init=1)
    elif centroid_init is None:
        clustering = KMeans(n_clusters=num_classes)

    clustering.fit(feats)
    clabels = clustering.labels_

    if score_fn_type == "labels":
        score = score_fn(labels, clabels)
    elif score_fn_type == "features":
        score = score_fn(feats, clabels)

    return score
