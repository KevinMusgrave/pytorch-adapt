import logging
import os
import shutil
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
from pytorch_adapt.frameworks.ignite import CheckpointFnCreator, Ignite
from pytorch_adapt.frameworks.ignite import utils as ignite_utils
from pytorch_adapt.frameworks.ignite.loggers import BasicLossLogger
from pytorch_adapt.utils import common_functions as c_f
from pytorch_adapt.validators import AccuracyValidator, EntropyValidator, ScoreHistory

from .. import TEST_DEVICE, TEST_FOLDER


class NanModel(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x) * torch.tensor(float("nan"), requires_grad=True)


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
    with_dumb_history=False,
    final_best_epoch=5,
    ignore_epoch=None,
    logger=None,
    val_hooks=None,
    with_checkpoint_fn=False,
    use_nan_model=False,
):
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

    C = nn.Linear(32, 16, device=TEST_DEVICE)
    if use_nan_model:
        C = NanModel(C)
    models = Models({"G": nn.Identity(), "C": C})
    adapter = Finetuner(models)
    validator = None
    if with_validator:
        validator = EntropyValidator()
        if with_dumb_history:
            validator = DumbScoreHistory(
                final_best_epoch=final_best_epoch,
                validator=validator,
                ignore_epoch=ignore_epoch,
            )
        else:
            validator = ScoreHistory(validator)

    checkpoint_fn = None
    if with_checkpoint_fn:
        checkpoint_fn = CheckpointFnCreator(
            dirname=TEST_FOLDER, n_saved=None, require_empty=False
        )

    adapter = Ignite(
        adapter,
        validator=validator,
        val_hooks=val_hooks,
        checkpoint_fn=checkpoint_fn,
        logger=logger,
        log_freq=1,
        with_pbars=False,
        device=TEST_DEVICE,
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
            for val_interval in [1, 2, 3]:
                for patience in [1, 5, 9]:
                    for ignore_epoch in [None, 0]:
                        for check_initial_score in [False, True]:
                            adapter, datasets = helper(
                                with_validator=True,
                                with_dumb_history=True,
                                final_best_epoch=final_best_epoch,
                                ignore_epoch=ignore_epoch,
                            )
                            dataloaders = DataloaderCreator(num_workers=0)(**datasets)
                            adapter.run(
                                dataloaders=dataloaders,
                                epoch_length=1,
                                early_stopper_kwargs={"patience": patience},
                                max_epochs=100,
                                val_interval=val_interval,
                                check_initial_score=check_initial_score,
                            )

                            num_best_check = final_best_epoch // val_interval
                            if num_best_check == 0:
                                # with patience == 1
                                # the scores will be [-1, -1]
                                correct_len = patience + 1
                            else:
                                correct_len = num_best_check + patience
                            if check_initial_score:
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
            for val_interval in [1, 3]:
                for val_hook_has_required_data in [True, False]:
                    val_hook = ValHook(self, val_hook_has_required_data)
                    adapter, datasets = helper(
                        with_validator=True, val_hooks=[val_hook]
                    )
                    dataloaders = DataloaderCreator(num_workers=0)(**datasets)
                    adapter.run(
                        dataloaders=dataloaders,
                        max_epochs=max_epochs,
                        val_interval=val_interval,
                    )
                    self.assertTrue(val_hook.num_calls == max_epochs // val_interval)

    def test_evaluate_best_model(self):
        adapter, datasets = helper(with_validator=True, with_checkpoint_fn=True)
        dc = DataloaderCreator(num_workers=0)
        adapter.run(
            datasets=datasets, dataloader_creator=dc, epoch_length=10, max_epochs=5
        )

        validator = EntropyValidator()
        best_score = adapter.evaluate_best_model(datasets, validator)
        self.assertTrue(best_score == adapter.validator.best_score)

        shutil.rmtree(TEST_FOLDER)

    def test_get_best_checkpoint(self):
        final_best_epoch = 3
        adapter, datasets = helper(
            with_validator=True,
            with_dumb_history=True,
            with_checkpoint_fn=True,
            final_best_epoch=final_best_epoch,
        )
        dc = DataloaderCreator(num_workers=0)
        adapter.run(
            datasets=datasets, dataloader_creator=dc, epoch_length=10, max_epochs=5
        )

        last_checkpoint = str(adapter.checkpoint_fn.get_best_checkpoint())
        self.assertTrue(
            last_checkpoint
            == os.path.join(TEST_FOLDER, f"checkpoint_{final_best_epoch}.pt")
        )

        shutil.rmtree(TEST_FOLDER)

    def test_load_best_checkpoint(self):
        adapter, datasets = helper(
            with_validator=True,
            with_checkpoint_fn=True,
        )
        dc = DataloaderCreator(num_workers=0)

        max_epochs = 10
        adapter.run(
            datasets=datasets,
            dataloader_creator=dc,
            epoch_length=10,
            max_epochs=max_epochs,
        )

        best_score = adapter.validator.best_score
        best_epoch = adapter.validator.best_epoch
        self.assertTrue(len(adapter.validator.score_history) == max_epochs)

        for checkpoint_fn in [
            adapter.checkpoint_fn,
            CheckpointFnCreator(dirname=TEST_FOLDER, require_empty=False),
        ]:
            checkpoint_fn.load_best_checkpoint({"validator": adapter.validator})
            self.assertTrue(best_score == adapter.validator.best_score)
            self.assertTrue(best_epoch == adapter.validator.best_epoch)
            self.assertTrue(len(adapter.validator.score_history) == best_epoch)

            fresh_adapter, _ = helper(
                with_checkpoint_fn=True,
            )
            for x in [adapter, fresh_adapter]:
                eval_best_score = x.evaluate_best_model(datasets, EntropyValidator())
                self.assertTrue(eval_best_score == best_score)

        shutil.rmtree(TEST_FOLDER)

    def test_is_done(self):
        for use_nan_model in [False, True]:
            adapter, datasets = helper(use_nan_model=use_nan_model)
            dc = DataloaderCreator(num_workers=0)
            max_epochs = 3
            adapter.run(
                datasets=datasets,
                dataloader_creator=dc,
                epoch_length=10,
                max_epochs=max_epochs,
            )

            is_done = ignite_utils.is_done(adapter.trainer, max_epochs)
            # if use_nan_model, then is_done should be false, and vice versa
            print(is_done, use_nan_model)
            self.assertTrue(is_done != use_nan_model)

    def test_val_hooks(self):
        val_hooks = [
            ScoreHistory(AccuracyValidator(key_map={"src_train": "src_val"})),
            EntropyValidator(),
        ]
        adapter, datasets = helper(val_hooks=val_hooks)
        dc = DataloaderCreator(num_workers=0)

        # Test that collected data gets filtered correctly
        # otherwise an exception will be raised
        # So we're just testing to make sure that exception
        # doesn't occur
        adapter.run(
            datasets=datasets,
            dataloader_creator=dc,
            epoch_length=10,
        )

        # test score history
        self.assertTrue(val_hooks[0].best_epoch == 1)
