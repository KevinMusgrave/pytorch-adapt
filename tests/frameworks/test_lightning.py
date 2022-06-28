import shutil
import unittest

import pytorch_lightning as pl
import torch

from pytorch_adapt.adapters import DANN
from pytorch_adapt.datasets import DataloaderCreator
from pytorch_adapt.frameworks.lightning import Lightning
from pytorch_adapt.frameworks.utils import filter_datasets
from pytorch_adapt.validators import IMValidator

from .. import TEST_DEVICE
from ..adapters.utils import get_datasets, get_gcd


class TestLightning(unittest.TestCase):
    def test_lightning(self):
        datasets = get_datasets()
        models = get_gcd()
        adapter = DANN(models)
        validator = IMValidator()
        adapter = Lightning(adapter, validator=validator)

        accelerator = "cpu" if TEST_DEVICE == torch.device("cpu") else "gpu"
        trainer = pl.Trainer(
            devices=1, accelerator=accelerator, max_epochs=2, log_every_n_steps=1
        )
        datasets = filter_datasets(datasets, validator)
        dataloaders = DataloaderCreator(num_workers=2)(**datasets)
        trainer.fit(adapter, dataloaders["train"], dataloaders["target_train"])

        shutil.rmtree("lightning_logs")
