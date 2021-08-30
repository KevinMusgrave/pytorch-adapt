import unittest

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from pytorch_adapt.validators import DeepEmbeddedValidator

from .. import TEST_DEVICE


### original implementation ###
# https://github.com/thuml/Deep-Embedded-Validation/blob/master/dev.py
def get_dev_risk(weight, error):
    """
    :param weight: shape [N, 1], the importance weight for N source samples in the validation set
    :param error: shape [N, 1], the error value for each source sample in the validation set
    (typically 0 for correct classification and 1 for wrong classification)
    """
    N, d = weight.shape
    _N, _d = error.shape
    assert N == _N and d == _d, "dimension mismatch!"
    weighted_error = weight * error
    cov = np.cov(np.concatenate((weighted_error, weight), axis=1), rowvar=False)[0][1]
    var_w = np.var(weight, ddof=1)
    eta = -cov / var_w
    return np.mean(weighted_error) + eta * np.mean(weight) - eta


def get_weight(source_feature, target_feature, validation_feature):
    """
    :param source_feature: shape [N_tr, d], features from training set
    :param target_feature: shape [N_te, d], features from test set
    :param validation_feature: shape [N_v, d], features from validation set
    :return:
    """
    N_s, d = source_feature.shape
    N_t, _d = target_feature.shape
    source_feature = source_feature.copy()
    target_feature = target_feature.copy()
    all_feature = np.concatenate((source_feature, target_feature))
    all_label = np.asarray([1] * N_s + [0] * N_t, dtype=np.int32)
    (
        feature_for_train,
        feature_for_test,
        label_for_train,
        label_for_test,
    ) = train_test_split(all_feature, all_label, train_size=0.8)

    decays = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
    val_acc = []
    domain_classifiers = []

    for decay in decays:
        domain_classifier = MLPClassifier(
            hidden_layer_sizes=(d, d, 2), activation="relu", alpha=decay
        )
        domain_classifier.fit(feature_for_train, label_for_train)
        output = domain_classifier.predict(feature_for_test)
        acc = np.mean((label_for_test == output).astype(np.float32))
        val_acc.append(acc)
        domain_classifiers.append(domain_classifier)

    index = val_acc.index(max(val_acc))

    domain_classifier = domain_classifiers[index]

    domain_out = domain_classifier.predict_proba(validation_feature)
    return domain_out[:, :1] / domain_out[:, 1:] * N_s * 1.0 / N_t


### end of original implementation ###


class TestDeepEmbeddedValidator(unittest.TestCase):
    def test_deep_embedded_validator(self):
        validator = DeepEmbeddedValidator()
        dataset_size = 1000
        features = torch.randn(dataset_size, 512, device=TEST_DEVICE)
        src_train = {"features": features}

        features = torch.randn(dataset_size, 512, device=TEST_DEVICE)
        labels = torch.arange(5, device=TEST_DEVICE).repeat(200)
        logits = torch.nn.functional.one_hot(labels).to(TEST_DEVICE)
        logits[800:] = torch.flip(
            logits[800:1000], dims=(0,)
        )  # make some of the logits incorrect
        logits = logits.float()
        src_val = {"features": features, "logits": logits, "labels": labels}

        shift_by = 0.1
        features = torch.randn(dataset_size, 512, device=TEST_DEVICE) + shift_by
        target_train = {"features": features}

        score = validator.score(
            epoch=0, src_train=src_train, src_val=src_val, target_train=target_train
        )
        print(score)

        # correct_weights = get_weight(
        #     src_train["features"].cpu().numpy(),
        #     target_train["features"].cpu().numpy(),
        #     src_val["features"].cpu().numpy(),
        # )

        # loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        # error_per_sample = loss_fn(src_val["logits"], src_val["labels"])
        # correct_dev = get_dev_risk(
        #     correct_weights, error_per_sample[:, None].cpu().numpy()
        # )
