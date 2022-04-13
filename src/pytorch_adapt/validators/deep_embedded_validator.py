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
from .accuracy_validator import AccuracyValidator
from .base_validator import BaseValidator
from .score_history import ScoreHistory


def dev_binary_fn(preds, labels):
    preds = torch.argmax(preds, dim=1)
    return (preds != labels).float()


def default_framework_fn(adapter, validator, folder):
    from ..frameworks.ignite import CheckpointFnCreator, Ignite

    validator = ScoreHistory(validator)
    checkpoint_fn = CheckpointFnCreator(dirname=folder)
    return Ignite(
        adapter, validator=validator, checkpoint_fn=checkpoint_fn, with_pbars=False
    )


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
        normalization=None,
        framework_fn=None,
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
        check_normalization(normalization)
        self.normalization = normalization
        self.framework_fn = c_f.default(framework_fn, default_framework_fn)
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
            self.framework_fn,
        )
        error_per_sample = self.error_fn(src_val[self.error_layer], src_val["labels"])
        output = get_dev_risk(weights, error_per_sample[:, None], self.normalization)
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


def check_normalization(normalization):
    if normalization not in [None, "max", "standardize"]:
        raise ValueError("normalization must be one of [None, 'max', 'standardize']")


def normalize_weights(weights, normalization):
    check_normalization(normalization)
    if normalization == "max":
        weights /= np.max(weights)  # normalize between 0 and 1
        weights -= np.mean(weights) - 1  # shift to have mean of 1
    elif normalization == "standardize":
        weights = (weights - np.mean(weights)) / np.std(weights)  # standardize
        weights += 1  # shift to have mean of 1
    return weights


def get_dev_risk(weight, error, normalization):
    weight = weight.cpu().numpy()
    error = error.cpu().numpy()

    N, d = weight.shape
    _N, _d = error.shape
    assert N == _N and d == _d, "dimension mismatch!"

    weight = normalize_weights(weight, normalization)
    weighted_error = weight * error
    cov = np.cov(np.concatenate((weighted_error, weight), axis=1), rowvar=False)[0][1]
    var_w = np.var(weight, ddof=1)
    eta = -cov / var_w
    return np.mean(weighted_error) + eta * np.mean(weight) - eta


def get_weights(
    source_feature,
    validation_feature,
    target_feature,
    num_workers,
    batch_size,
    temp_folder,
    framework_fn,
):
    device = source_feature.device
    source_feature = source_feature.cpu().numpy()
    validation_feature = validation_feature.cpu().numpy()
    target_feature = target_feature.cpu().numpy()

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
    val_acc, trainers, folders = [], [], []
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
        trainer = framework_fn(trainer, validator, curr_folder)
        datasets = {"train": train_set, "src_val": val_set}
        bs = int(np.min([len(train_set), len(val_set), batch_size]))

        acc, _ = trainer.run(
            datasets,
            dataloader_creator=DataloaderCreator(
                num_workers=num_workers, batch_size=bs
            ),
            max_epochs=epochs,
            val_interval=1,
            early_stopper_kwargs={"patience": patience},
        )
        val_acc.append(acc)
        trainers.append(trainer)
        folders.append(curr_folder)

    torch.cuda.empty_cache()
    D_accuracy_val = max(val_acc)
    index = val_acc.index(D_accuracy_val)

    labels = torch.ones(len(validation_feature), dtype=int)
    validation_set = SourceDataset(pml_cf.EmbeddingDataset(validation_feature, labels))
    trainer = trainers[index]
    trainer.checkpoint_fn.load_best_checkpoint(
        {"models": trainer.adapter.models},
    )
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
