import shutil
import unittest

import numpy as np

from pytorch_adapt.containers.base_container import containers_are_equal
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


def get_val_hook(saver):
    validator = ScoreHistories(
        MultipleValidators(
            [
                AccuracyValidator(key_map={"src_train": "src_val"}),
                AccuracyValidator(),
            ],
        )
    )
    return IgniteValHookWrapper(validator, saver=saver.validator_saver)


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
        saver = savers.Saver(folder=TEST_FOLDER)

        val_hook1 = get_val_hook(saver)
        validator1 = get_validator()

        dann1, datasets = get_dann(
            validator=validator1, val_hook=val_hook1, saver=saver
        )
        dann1.run(
            datasets=datasets,
            epoch_length=2,
            max_epochs=max_epochs,
        )

        for load_all_at_once in [True, False]:
            val_hook2 = get_val_hook(saver)
            validator2 = get_validator()
            dann2, _ = get_dann()

            self.assert_not_equal(
                dann1, validator1, val_hook1, dann2, validator2, val_hook2
            )

            saver.validator_saver.load(val_hook2.validator, "val_hook")
            if load_all_at_once:
                saver.load_all(dann2.adapter, validator2, dann2)
            else:
                saver.load_ignite(dann2.trainer)
                saver.load_adapter(dann2.adapter, max_epochs)
                saver.load_validator(validator2)

            self.assert_equal(
                dann1, validator1, val_hook1, dann2, validator2, val_hook2
            )

        saver = savers.Saver(folder=TEST_FOLDER)
        val_hook3 = get_val_hook(saver)
        validator3 = get_validator()
        dann3, _ = get_dann(validator=validator3, val_hook=val_hook3, saver=saver)
        self.assert_not_equal(
            dann1, validator1, val_hook1, dann3, validator3, val_hook3
        )
        # this should load and then not run
        # because it has already run for max_epochs
        saver.validator_saver.load(val_hook3.validator, "val_hook")
        dann3.run(
            datasets=datasets,
            epoch_length=2,
            max_epochs=max_epochs,
            resume="latest",
        )
        self.assert_equal(dann1, validator1, val_hook1, dann3, validator3, val_hook3)

        validator3.epochs = validator3.epochs[:1]
        validator3.score_history = validator3.score_history[:1]
        saver.save_validator(validator3)
        with self.assertRaises(exceptions.ResumeCheckError):
            dann3.run(
                datasets=datasets,
                epoch_length=2,
                max_epochs=max_epochs,
                resume="latest",
            )

        validator3 = ScoreHistories(
            MultipleValidators(
                [
                    AccuracyValidator(),
                    AccuracyValidator(),
                    AccuracyValidator(),
                ],
            )
        )

        validator4 = ScoreHistory(AccuracyValidator())

        self.assertRaises(FileNotFoundError, lambda: saver.load_validator(validator3))
        self.assertRaises(FileNotFoundError, lambda: saver.load_validator(validator4))

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

        for attrname in ["best_epoch", "best_score", "latest_score"]:
            self.assertTrue(
                getattr(val_hook1.validator, attrname)
                == getattr(val_hook2.validator, attrname)
            )
            self.assertTrue(
                getattr(validator1, attrname) == getattr(validator2, attrname)
            )

        for attrname in ["score_history", "epochs"]:
            self.assertTrue(
                np.array_equal(
                    getattr(val_hook1.validator, attrname),
                    getattr(val_hook2.validator, attrname),
                )
            )
            self.assertTrue(
                np.array_equal(
                    getattr(validator1, attrname), getattr(validator2, attrname)
                )
            )
