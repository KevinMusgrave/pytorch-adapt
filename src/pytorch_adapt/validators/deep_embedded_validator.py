import logging
import os
import shutil

import numpy as np
import torch
from pytorch_metric_learning.utils import common_functions as pml_cf
from sklearn.model_selection import train_test_split
from torchmetrics.functional import accuracy as tmf_accuracy

from ..adapters import Finetuner
from ..containers import Models, Optimizers
from ..datasets import DataloaderCreator, SourceDataset
from ..models import Discriminator
from ..utils import common_functions as c_f
from ..utils.savers import Saver
from .accuracy_validator import AccuracyValidator
from .base_validator import BaseValidator
from .score_history import ScoreHistory


class DeepEmbeddedValidator(BaseValidator):
    """
    Implementation of
    [Towards Accurate Model Selection in Deep Unsupervised Domain Adaptation](http://proceedings.mlr.press/v97/you19a.html)
    """

    def __init__(
        self,
        temp_folder,
        layer="features",
        num_workers=0,
        batch_size=32,
        error_fn=None,
        error_layer="logits",
        framework_cls=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.temp_folder = temp_folder
        self.layer = layer
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.error_fn = c_f.default(
            error_fn, torch.nn.CrossEntropyLoss(reduction="none")
        )
        self.error_layer = error_layer
        self.framework_cls = framework_cls
        if self.framework_cls is None:
            from ..frameworks.ignite import Ignite

            self.framework_cls = Ignite
        self.D_accuracy_val = None
        self.D_accuracy_test = None
        self.mean_error = None
        self._DEV_recordable = ["D_accuracy_val", "D_accuracy_test", "mean_error"]
        pml_cf.add_to_recordable_attributes(self, list_of_names=self._DEV_recordable)

    def compute_score(self, src_train, src_val, target_train):
        init_logging_level = c_f.LOGGER.level
        c_f.LOGGER.setLevel(logging.WARNING)
        weights, self.D_accuracy_val, self.D_accuracy_test = get_weights(
            src_train[self.layer],
            src_val[self.layer],
            target_train[self.layer],
            self.num_workers,
            self.batch_size,
            self.temp_folder,
            self.framework_cls,
        )
        error_per_sample = self.error_fn(src_val[self.error_layer], src_val["labels"])
        output = get_dev_risk(weights, error_per_sample[:, None])
        self.mean_error = torch.mean(error_per_sample).item()
        c_f.LOGGER.setLevel(init_logging_level)
        return -output

    def extra_repr(self):
        x = super().extra_repr()
        x += f"\n{c_f.extra_repr(self, self._DEV_recordable)}"
        return x


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
    eta = -cov / (var_w + 1e-6)
    return np.mean(weighted_error) + eta * np.mean(weight) - eta


def get_weights(
    source_feature,
    validation_feature,
    target_feature,
    num_workers,
    batch_size,
    temp_folder,
    framework_cls,
):
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
    val_acc, trainers, savers, folders = [], [], [], []
    epochs = 100
    patience = 2

    for i, decay in enumerate(decays):
        torch.cuda.empty_cache()
        curr_folder = os.path.join(temp_folder, f"DeepEmbeddedValidation{i}")
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
        validator = AccuracyValidator(
            torchmetric_kwargs={"average": "macro", "num_classes": 2}
        )
        validator = ScoreHistory(validator)
        saver = Saver(folder=curr_folder)
        trainer = framework_cls(
            trainer, validator=validator, saver=saver, with_pbars=False
        )
        datasets = {"train": train_set, "src_val": val_set}
        bs = int(np.min([len(train_set), len(val_set), batch_size]))

        acc, _ = trainer.run(
            datasets,
            dataloader_creator=DataloaderCreator(
                num_workers=num_workers, batch_size=bs
            ),
            max_epochs=epochs,
            validation_interval=1,
            patience=patience,
        )
        val_acc.append(acc)
        trainers.append(trainer)
        savers.append(saver)
        folders.append(curr_folder)

    torch.cuda.empty_cache()
    D_accuracy_val = max(val_acc)
    index = val_acc.index(D_accuracy_val)

    labels = torch.ones(len(validation_feature), dtype=int)
    validation_set = SourceDataset(pml_cf.EmbeddingDataset(validation_feature, labels))
    trainer, saver = trainers[index], savers[index]
    saver.load_adapter(trainer.adapter, "best")
    bs = min(len(validation_set), batch_size)
    dataloader = torch.utils.data.DataLoader(
        validation_set, num_workers=num_workers, batch_size=bs
    )
    domain_out = trainer.get_all_outputs(dataloader, "val")
    domain_out = domain_out["val"]["preds"]
    weights = (domain_out[:, :1] / domain_out[:, 1:]) * (float(N_s) / N_t)

    [shutil.rmtree(f) for f in folders]

    D_accuracy_test = tmf_accuracy(domain_out, labels.to(domain_out.device)).item()

    return weights, D_accuracy_val, D_accuracy_test
