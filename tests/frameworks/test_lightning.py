import shutil
import unittest

import pytorch_lightning as pl

from pytorch_adapt.adapters import DANN
from pytorch_adapt.datasets import DataloaderCreator
from pytorch_adapt.frameworks.lightning import Lightning
from pytorch_adapt.validators import IMValidator

from ..adapters.utils import get_datasets, get_gcd


class TestLightning(unittest.TestCase):
    def test_lightning(self):
        datasets = get_datasets()
        models = get_gcd()
        adapter = DANN(models)
        validator = IMValidator()
        datasets = {
            k: v
            for k, v in datasets.items()
            if k in ["train"] + validator.required_data
        }
        adapter = Lightning(adapter, validator=validator)
        trainer = pl.Trainer(gpus=1, max_epochs=2, log_every_n_steps=1)

        dataloaders = DataloaderCreator(num_workers=2)(**datasets)
        trainer.fit(adapter, dataloaders["train"], dataloaders["target_train"])

        # shutil.rmtree("lightning_logs")
