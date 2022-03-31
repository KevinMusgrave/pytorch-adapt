import os
import shutil
import unittest

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from pytorch_adapt.validators import DeepEmbeddedValidator
from pytorch_adapt.validators.deep_embedded_validator import (
    get_dev_risk as pa_get_dev_risk,
)
from pytorch_adapt.validators.deep_embedded_validator import normalize_weights

from .. import TEST_DEVICE, TEST_FOLDER


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


def get_correct_score(cls, src_train, target_train, src_val):
    weights = get_weight(
        src_train["features"].cpu().numpy(),
        target_train["features"].cpu().numpy(),
        src_val["features"].cpu().numpy(),
    )

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    error_per_sample = loss_fn(src_val["logits"], src_val["labels"])
    correct_dev_risk = get_dev_risk(weights, error_per_sample[:, None].cpu().numpy())
    dev_risk = pa_get_dev_risk(
        torch.from_numpy(weights), error_per_sample[:, None], None
    )
    cls.assertTrue(np.isclose(correct_dev_risk, dev_risk, rtol=0.01))
    return correct_dev_risk


class TestDeepEmbeddedValidator(unittest.TestCase):
    def test_deep_embedded_validator(self):
        torch.manual_seed(1234)
        np.random.seed(1234)
        validator = DeepEmbeddedValidator(
            temp_folder=os.path.join(TEST_FOLDER, "deep_embedded_validation"),
            batch_size=256,
        )
        for epoch, dataset_size in enumerate([100, 350, 600]):
            features = torch.randn(dataset_size, 512, device=TEST_DEVICE)
            src_train = {"features": features}

            features = torch.randn(dataset_size, 512, device=TEST_DEVICE)
            labels = torch.arange(5, device=TEST_DEVICE).repeat(dataset_size // 5)
            logits = torch.nn.functional.one_hot(labels).to(TEST_DEVICE)
            s = int(dataset_size * (0.8))
            logits[s:] = torch.flip(
                logits[s:], dims=(0,)
            )  # make some of the logits incorrect
            logits = logits.float()
            src_val_features = torch.randn(dataset_size, 512, device=TEST_DEVICE)
            src_val = {"features": src_val_features, "logits": logits, "labels": labels}

            shift_by = 0.5
            features = torch.randn(dataset_size, 512, device=TEST_DEVICE) + shift_by
            target_train = {"features": features}

            score = validator(
                src_train=src_train,
                src_val=src_val,
                target_train=target_train,
            )

            correct_score = get_correct_score(self, src_train, target_train, src_val)
            correct_score = -correct_score
            self.assertTrue(np.isclose(score, correct_score, rtol=0.2))

        shutil.rmtree(TEST_FOLDER)

    def test_normalize_weights(self):
        N = 1000
        np.random.seed(15)

        def new_weights():
            weights = np.random.normal(loc=10, scale=0.1, size=(N, 1))
            weights[0] = 100000000000
            self.assertTrue(not np.isclose(np.mean(weights), 1))
            self.assertTrue(not np.isclose(np.std(weights), 1))
            return weights

        weights = normalize_weights(new_weights(), "max")
        self.assertTrue(np.isclose(np.mean(weights), 1))

        weights = normalize_weights(new_weights(), "standardize")
        self.assertTrue(np.isclose(np.mean(weights), 1))
        self.assertTrue(np.isclose(np.std(weights), 1))

        weights = normalize_weights(new_weights(), None)
        self.assertTrue(not np.isclose(np.mean(weights), 1))
        self.assertTrue(not np.isclose(np.std(weights), 1))

        with self.assertRaises(ValueError) as c:
            normalize_weights(new_weights(), "")
