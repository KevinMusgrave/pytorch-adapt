import unittest

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from pytorch_adapt.validators import SilhouetteScoreValidator

from .. import TEST_DEVICE

#### original implementation from ####
#### https://github.com/lr94/abas/blob/master/model_selection.py ####
#### Removed the pca step ####


def get_centroids(data, labels, num_classes):
    centroids = np.zeros((num_classes, data.shape[1]))
    for cid in range(num_classes):
        # Since we are using pseudolabels to compute centroids, some classes might not have instances according to the
        # pseudolabels assigned by the current model. In that case .mean() would return NaN causing KMeans to fail.
        # We set to 0 the missing centroids
        if (labels == cid).any():
            centroids[cid] = data[labels == cid].mean(0)

    return centroids


def get_clustering_performance(feats, plabels, num_classes):
    """
    :param feats: N x out numpy vector
    :param plabels:  N numpy vector
    :param num_classes: int
    :param source_feats
    :return: silhouette and calinski harabasz scores
    """

    try:
        x = feats
        centroids = get_centroids(x, plabels, num_classes)

        clustering = KMeans(n_clusters=num_classes, init=centroids, n_init=1)

        clustering.fit(x)
        clabels = clustering.labels_
        ss = silhouette_score(x, clabels)
    except ValueError:
        ss = float("nan")

    return ss


#### end of original ####
#### https://github.com/lr94/abas/blob/master/model_selection.py ####


class TestSilhouetteScoreValidator(unittest.TestCase):
    def test_silhouette_score_validator(self):
        torch.cuda.empty_cache()
        validator = SilhouetteScoreValidator()
        dataset_size = 10000
        num_classes = 2
        half = dataset_size // 2
        features = torch.ones(dataset_size, 64, device=TEST_DEVICE)
        features[:half] = 0
        logits = torch.zeros(dataset_size, num_classes, device=TEST_DEVICE)
        logits[:half, 0] = 1
        logits[half:, 1] = 1
        score = validator.score(target_train={"features": features, "logits": logits})
        correct_score = get_clustering_performance(
            features.cpu(), torch.argmax(logits, dim=1).cpu(), num_classes=num_classes
        )

        self.assertTrue(score == correct_score == 1)

        dataset_size = 10000
        num_classes = 100
        features = torch.randn(dataset_size, 64, device=TEST_DEVICE)
        logits = torch.randn(dataset_size, num_classes, device=TEST_DEVICE)
        score = validator.score(target_train={"features": features, "logits": logits})
        correct_score = get_clustering_performance(
            features.cpu(), torch.argmax(logits, dim=1).cpu(), num_classes=num_classes
        )
        self.assertTrue(score < 0.02 and correct_score < 0.02)
