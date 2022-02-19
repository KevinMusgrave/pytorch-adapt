import os
import shutil
import unittest

import numpy as np

from pytorch_adapt.containers.base_container import containers_are_equal
from pytorch_adapt.datasets import DataloaderCreator
from pytorch_adapt.frameworks.ignite import IgniteValHookWrapper, savers
from pytorch_adapt.utils import exceptions
from pytorch_adapt.validators import (
    AccuracyValidator,
    MultipleValidators,
    ScoreHistories,
    ScoreHistory,
)

from .. import TEST_FOLDER
from .get_dann import get_dann


def get_val_hook():
    validator = ScoreHistories(
        MultipleValidators(
            [
                AccuracyValidator(key_map={"src_train": "src_val"}),
                AccuracyValidator(),
            ],
        )
    )
    return IgniteValHookWrapper(validator)


def get_validator():
    return ScoreHistories(
        MultipleValidators(
            [
                AccuracyValidator(),
                AccuracyValidator(),
            ],
        )
    )


class TestSaveAndLoad(unittest.TestCase):
    def test_save_and_load(self):
        max_epochs = 3
        checkpoint_fn = savers.CheckpointFnCreator(dirname=TEST_FOLDER, n_saved=None)

        val_hook1 = get_val_hook()
        validator1 = get_validator()

        dann1, datasets = get_dann(
            validator=validator1, val_hooks=[val_hook1], checkpoint_fn=checkpoint_fn
        )
        dataloader_creator = DataloaderCreator(num_workers=2)

        dann1.run(
            datasets=datasets,
            dataloader_creator=dataloader_creator,
            epoch_length=2,
            max_epochs=max_epochs,
        )

        for load_partial in [True, False]:
            val_hook2 = get_val_hook()
            validator2 = get_validator()
            dann2, _ = get_dann(validator=validator2, val_hooks=[val_hook2])

            self.assert_not_equal(
                dann1, validator1, val_hook1, dann2, validator2, val_hook2
            )
            if load_partial:
                objs = [
                    {"engine": dann2.trainer, "adapter": dann2.adapter},
                    {"validator": dann2.validator, "val_hook0": val_hook2},
                ]
            else:
                objs = [
                    {
                        "engine": dann2.trainer,
                        "adapter": dann2.adapter,
                        "validator": dann2.validator,
                        "val_hook0": val_hook2,
                    }
                ]

            for to_load in objs:
                checkpoint_fn.load_objects(to_load, global_step=3)

            self.assert_equal(
                dann1, validator1, val_hook1, dann2, validator2, val_hook2
            )

        checkpoint_fn = savers.CheckpointFnCreator(
            dirname=TEST_FOLDER, n_saved=None, require_empty=False
        )
        val_hook3 = get_val_hook()
        validator3 = get_validator()
        dann3, _ = get_dann(
            validator=validator3, val_hooks=[val_hook3], checkpoint_fn=checkpoint_fn
        )
        self.assert_not_equal(
            dann1, validator1, val_hook1, dann3, validator3, val_hook3
        )
        # this should load and then not run
        # because it has already run for max_epochs
        dann3.run(
            datasets=datasets,
            epoch_length=2,
            max_epochs=max_epochs,
            resume=dann1.checkpoint_fn.last_checkpoint,
        )
        self.assert_equal(dann1, validator1, val_hook1, dann3, validator3, val_hook3)

        # Truncate validator.epochs, then save
        # When resuming, epochs won't match trainer epoch, and will raise exception
        validator3.epochs = validator3.epochs[:1]
        validator3.score_history = validator3.score_history[:1]
        checkpoint_fn(dann3.adapter, dann3.validator, dann3.val_hooks)(dann3.trainer)
        with self.assertRaises(exceptions.ResumeCheckError):
            dann3.run(
                datasets=datasets,
                epoch_length=2,
                max_epochs=max_epochs,
                resume=dann1.checkpoint_fn.last_checkpoint,
            )

        validator4 = ScoreHistory(AccuracyValidator())

        validator5 = ScoreHistories(
            MultipleValidators(
                [
                    AccuracyValidator(),
                    AccuracyValidator(),
                    AccuracyValidator(),
                ],
            )
        )

        # should run without problems
        checkpoint_fn.load_objects({"validator": validator3}, global_step=3)

        with self.assertRaises(KeyError):
            checkpoint_fn.load_objects({"validator": validator4}, global_step=3)

        with self.assertRaises(KeyError):
            checkpoint_fn.load_objects({"validator": validator5}, global_step=3)
        shutil.rmtree(TEST_FOLDER)

    def assert_not_equal(
        self, dann1, validator1, val_hook1, dann2, validator2, val_hook2
    ):
        # check ignite engine state
        self.assertTrue(dann1.trainer.state_dict() != dann2.trainer.state_dict())

        # check adapter.containers
        self.assertTrue(
            not containers_are_equal(dann1.adapter.containers, dann2.adapter.containers)
        )

        # check the attributes as well
        for k in ["models"]:
            c1 = getattr(dann1.adapter, k)
            c2 = getattr(dann2.adapter, k)
            self.assertTrue(not containers_are_equal(c1, c2))

        for attrname in ["best_epoch", "best_score", "latest_score"]:
            self.assertTrue(getattr(val_hook2.validator, attrname) is None)
            self.assertTrue(getattr(validator2, attrname) is None)

    def assert_equal(self, dann1, validator1, val_hook1, dann2, validator2, val_hook2):
        self.assertTrue(dann1.trainer.state_dict() == dann2.trainer.state_dict())

        # check adapter.containers
        self.assertTrue(
            containers_are_equal(dann1.adapter.containers, dann2.adapter.containers)
        )

        # check the attributes as well
        for k in [
            "models",
            "optimizers",
            "lr_schedulers",
            "misc",
        ]:
            c1 = getattr(dann1.adapter, k)
            c2 = getattr(dann2.adapter, k)
            self.assertTrue(containers_are_equal(c1, c2))

        self.assert_validator_equal(validator1, validator2)
        self.assert_validator_equal(val_hook1.validator, val_hook2.validator)
        for k in val_hook1.validator.histories.keys():
            self.assert_validator_equal(
                val_hook1.validator.histories[k], val_hook2.validator.histories[k]
            )

    def assert_validator_equal(self, v1, v2):
        for attrname in ["best_epoch", "best_score", "latest_score"]:
            self.assertTrue(getattr(v1, attrname) == getattr(v2, attrname))
        for attrname in ["score_history", "epochs"]:
            self.assertTrue(
                np.array_equal(
                    getattr(v1, attrname),
                    getattr(v2, attrname),
                )
            )
