import logging
import unittest

import torch
import torch.nn as nn
from pytorch_metric_learning.utils.common_functions import EmbeddingDataset

from pytorch_adapt.adapters import Finetuner
from pytorch_adapt.containers import Models
from pytorch_adapt.datasets import (
    CombinedSourceAndTargetDataset,
    DataloaderCreator,
    SourceDataset,
    TargetDataset,
)
from pytorch_adapt.frameworks.ignite import Ignite
from pytorch_adapt.frameworks.ignite.loggers import BasicLossLogger
from pytorch_adapt.utils import common_functions as c_f
from pytorch_adapt.validators import EntropyValidator, ScoreHistory


class ValHook:
    def __init__(self, unittester, val_hook_has_required_data):
        self.unittester = unittester
        self.num_calls = 0
        self.val_hook_has_required_data = val_hook_has_required_data
        if self.val_hook_has_required_data:
            self.required_data = ["src_train"]

    def __call__(self, epoch, **kwargs):
        if self.val_hook_has_required_data:
            self.unittester.assertTrue("src_train" in kwargs)
        self.num_calls += 1


def helper(
    with_validator=False,
    final_best_epoch=5,
    ignore_epoch=None,
    logger=None,
    val_hook=None,
):
    device = torch.device("cuda")
    datasets = {}
    for k in ["src_train", "target_train"]:
        datasets[k] = EmbeddingDataset(
            torch.randn(128, 32), torch.randint(0, 10, size=(128,))
        )
        if k.startswith("src"):
            datasets[k] = SourceDataset(datasets[k])
        else:
            datasets[k] = TargetDataset(datasets[k])

    datasets["train"] = CombinedSourceAndTargetDataset(
        datasets["src_train"], datasets["target_train"]
    )

    models = Models({"G": nn.Identity(), "C": nn.Linear(32, 16, device=device)})
    adapter = Finetuner(models)
    validator = None
    if with_validator:
        validator = EntropyValidator()
        validator = DumbScoreHistory(
            final_best_epoch=final_best_epoch,
            validator=validator,
            ignore_epoch=ignore_epoch,
        )
    adapter = Ignite(
        adapter,
        validator=validator,
        val_hooks=[val_hook],
        logger=logger,
        log_freq=1,
        with_pbars=False,
        device=device,
    )
    return adapter, datasets


class DumbScoreHistory(ScoreHistory):
    def __init__(self, final_best_epoch, **kwargs):
        super().__init__(**kwargs)
        self.final_best_epoch = final_best_epoch

    def append_to_history_and_normalize(self, score, epoch):
        score = epoch if epoch <= self.final_best_epoch else -1
        super().append_to_history_and_normalize(score, epoch)


class TestIgnite(unittest.TestCase):
    def test_datasets_dataloaders(self):
        adapter, datasets = helper()

        # passing in datasets
        dc = DataloaderCreator(num_workers=0)
        adapter.run(datasets=datasets, dataloader_creator=dc, epoch_length=10)

        # passing in dataloaders
        dataloaders = DataloaderCreator(num_workers=0)(**datasets)
        adapter.run(dataloaders=dataloaders, epoch_length=10)

    def test_early_stopping(self):
        logging.getLogger(c_f.LOGGER_NAME).setLevel(logging.CRITICAL)
        for final_best_epoch in [1, 4]:
            for validation_interval in [1, 2, 3]:
                for patience in [0, 1, 5, 9]:
                    for ignore_epoch in [None, 0]:
                        for check_initial_score in [False, True]:
                            adapter, datasets = helper(
                                with_validator=True,
                                final_best_epoch=final_best_epoch,
                                ignore_epoch=ignore_epoch,
                            )
                            dataloaders = DataloaderCreator(num_workers=0)(**datasets)
                            adapter.run(
                                dataloaders=dataloaders,
                                epoch_length=1,
                                patience=patience,
                                max_epochs=100,
                                validation_interval=validation_interval,
                                check_initial_score=check_initial_score,
                            )

                            num_best_check = final_best_epoch // validation_interval
                            if num_best_check == 0:
                                # with patience == 1
                                # the scores will be [-1, -1, -1]
                                correct_len = patience + 2
                            else:
                                correct_len = num_best_check + patience + 1
                            if check_initial_score and (
                                num_best_check > 0 or ignore_epoch is not None
                            ):
                                correct_len += 1
                            self.assertTrue(
                                len(adapter.validator.score_history) == correct_len
                            )

        logging.getLogger(c_f.LOGGER_NAME).setLevel(logging.INFO)

    def test_get_all_outputs(self):
        adapter, datasets = helper()

        # passing in dataloaders
        dataloaders = DataloaderCreator(num_workers=0)(**datasets)
        split = "target_train"
        data = adapter.get_all_outputs(dataloaders[split], split)
        self.assertTrue(data.keys() == {split})
        self.assertTrue(
            data[split].keys()
            == {"features", "logits", "preds", "domain", "sample_idx"}
        )

    def test_basic_loss_logger(self):
        logger = BasicLossLogger()
        adapter, datasets = helper(logger=logger)
        dataloaders = DataloaderCreator(num_workers=0)(**datasets)
        epoch_length = 10
        adapter.run(dataloaders=dataloaders, epoch_length=epoch_length)
        losses = logger.get_losses()
        self.assertTrue(len(losses) == 1)
        losses = losses["total_loss"]
        self.assertTrue(len(losses) == 2)
        for v in losses.values():
            self.assertTrue(len(v) == epoch_length)

        # should be cleared in previous get_losses call
        losses = logger.get_losses()
        self.assertTrue(len(losses) == 0)

    def test_validation_runner(self):
        max_epochs = 6
        for with_validator in [True, False]:
            for validation_interval in [1, 3]:
                for val_hook_has_required_data in [True, False]:
                    val_hook = ValHook(self, val_hook_has_required_data)
                    adapter, datasets = helper(with_validator=True, val_hook=val_hook)
                    dataloaders = DataloaderCreator(num_workers=0)(**datasets)
                    adapter.run(
                        dataloaders=dataloaders,
                        max_epochs=max_epochs,
                        validation_interval=validation_interval,
                    )
                    self.assertTrue(
                        val_hook.num_calls == max_epochs // validation_interval
                    )
