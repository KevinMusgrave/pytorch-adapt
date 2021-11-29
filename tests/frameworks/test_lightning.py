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
        adapter = Lightning(adapter, validator=validator)
        trainer = pl.Trainer(gpus=1, max_epochs=1, max_steps=1)

        dataloaders = DataloaderCreator(num_workers=2)(**datasets)
        trainer.fit(adapter, dataloaders["train"])

        shutil.rmtree("lightning_logs")
