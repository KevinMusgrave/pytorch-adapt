import numpy as np
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
        pca_size=64,
        centroid_init=None,
        score_fn=None,
        score_fn_type="labels",
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

    def _required_data(self):
        x = ["target_train"]
        if self.requires_src:
            x.append("src_train")
        return x

    @property
    def requires_src(self):
        return self.with_src or self.src_for_pca

    def compute_score(self, target_train, src_train=None):
        feats = target_train[self.layer].cpu().numpy()
        labels = self.target_label_fn(target_train).cpu().numpy()
        num_classes = target_train["logits"].shape[1]

        src_feats, src_labels = None, None
        if self.requires_src:
            src_feats = src_train[self.layer].cpu().numpy()
        if self.with_src:
            src_labels = src_train["labels"].cpu().numpy()

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
    pca_size=64,
    centroid_init="label_centers",
):
    """
    :param feats: N x out numpy vector
    :param labels:  N numpy vector
    :param num_classes: int
    :param src_feats
    :param pca_size
    :return: silhouette and calinski harabasz scores
    """
    num_target_feats = feats.shape[0]

    if src_feats is not None:
        feats = np.concatenate((feats, src_feats), axis=0)

    if pca_size is not None:
        pca = PCA(pca_size)
        feats = pca.fit_transform(feats)

    # no labels means src_feats is just for pca
    if src_labels is None:
        feats = feats[:num_target_feats]
    else:
        labels = np.concatenate((labels, src_labels), axis=0)

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
