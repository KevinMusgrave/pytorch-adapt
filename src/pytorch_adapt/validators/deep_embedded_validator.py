import logging

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_metric_learning.utils import common_functions as pml_cf
from sklearn.model_selection import train_test_split

from ..adapters import Finetuner
from ..containers import Models, Optimizers
from ..datasets import SourceDataset
from ..frameworks import Ignite
from ..models import Discriminator
from ..utils import common_functions as c_f
from .accuracy_validator import AccuracyValidator
from .base_validator import BaseValidator


class DeepEmbeddedValidator(BaseValidator):
    def __init__(self, layer="features", **kwargs):
        super().__init__(**kwargs)
        self.layer = layer

    def compute_score(self, src_train, src_val, target_train):
        init_logging_level = c_f.LOGGER.level
        c_f.LOGGER.setLevel(logging.WARNING)
        weights = get_weights(
            src_train[self.layer], src_val[self.layer], target_train[self.layer]
        )
        error_per_sample = F.cross_entropy(
            src_val["logits"], src_val["labels"], reduction="none"
        )
        output = get_dev_risk(weights, error_per_sample[:, None])
        c_f.LOGGER.setLevel(init_logging_level)
        return output

    @property
    def maximize(self):
        return False


#########################################################################
#### ADAPTED FROM https://github.com/thuml/Deep-Embedded-Validation #####
#########################################################################


def get_dev_risk(weight, error):
    """
    :param weight: shape [N, 1], the importance weight for N source samples in the validation set
    :param error: shape [N, 1], the error value for each source sample in the validation set
    (typically 0 for correct classification and 1 for wrong classification)
    """
    if torch.any(weight < 0) or torch.any(error < 0):
        raise ValueError("weights and errors must be positive")

    weight = pml_cf.to_numpy(weight)
    error = pml_cf.to_numpy(error)

    N, d = weight.shape
    _N, _d = error.shape
    assert N == _N and d == _d, "dimension mismatch!"
    weighted_error = weight * error
    cov = np.cov(np.concatenate((weighted_error, weight), axis=1), rowvar=False)[0][1]
    var_w = np.var(weight, ddof=1)
    eta = -cov / var_w
    return np.mean(weighted_error) + eta * np.mean(weight) - eta


def get_weights(source_feature, validation_feature, target_feature):
    """
    :param source_feature: shape [N_tr, d], features from training set
    :param validation_feature: shape [N_v, d], features from validation set
    :param target_feature: shape [N_te, d], features from test set
    :return:
    """

    device = source_feature.device
    source_feature = pml_cf.to_numpy(source_feature)
    validation_feature = pml_cf.to_numpy(validation_feature)
    target_feature = pml_cf.to_numpy(target_feature)

    N_s, d = source_feature.shape
    N_t, _d = target_feature.shape
    source_feature = source_feature.copy()
    target_feature = target_feature.copy()
    all_feature = np.concatenate((source_feature, target_feature))
    all_label = np.asarray([1] * N_s + [0] * N_t, dtype=np.int64)
    (
        feature_for_train,
        feature_for_test,
        label_for_train,
        label_for_test,
    ) = train_test_split(all_feature, all_label, train_size=0.8)

    train_set = SourceDataset(
        pml_cf.EmbeddingDataset(feature_for_train, label_for_train)
    )
    val_set = SourceDataset(pml_cf.EmbeddingDataset(feature_for_test, label_for_test))

    decays = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
    val_acc = []
    trainers = []
    epochs = 2

    for decay in decays:
        models = Models(
            {
                "G": torch.nn.Identity(),
                "C": Discriminator(d, h=d, out_size=2).to(device),
            }
        )
        optimizers = Optimizers(
            (torch.optim.Adam, {"lr": 0.001, "weight_decay": decay})
        )
        trainer = Finetuner(models=models, optimizers=optimizers)
        trainer = Ignite(trainer, with_pbars=False)
        datasets = {"train": train_set, "src_val": val_set}
        acc, _ = trainer.run(
            datasets,
            max_epochs=epochs,
            validator=AccuracyValidator(),
            validation_interval=epochs,
        )
        val_acc.append(acc)
        trainers.append(trainer)

    index = val_acc.index(max(val_acc))

    validation_set = SourceDataset(
        pml_cf.EmbeddingDataset(validation_feature, np.ones(len(validation_feature)))
    )
    trainer = trainers[index]
    dataloader = torch.utils.data.DataLoader(validation_set)
    domain_out = trainer.get_all_outputs(dataloader, "val")
    domain_out = domain_out["val"]["preds"]
    return (domain_out[:, :1] / domain_out[:, 1:]) * (float(N_s) / N_t)
