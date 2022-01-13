import unittest

from pytorch_adapt.adapters import DANN
from pytorch_adapt.datasets import DataloaderCreator
from pytorch_adapt.frameworks.ignite import Ignite

from ..adapters.utils import get_datasets, get_gcd


class TestIgnite(unittest.TestCase):
    def test_ignite(self):
        datasets = get_datasets()
        models = get_gcd()
        adapter = DANN(models)
        adapter = Ignite(adapter)

        # passing in datasets
        dc = DataloaderCreator(num_workers=2)
        adapter.run(datasets=datasets, dataloader_creator=dc, epoch_length=10)

        # passing in dataloaders
        dataloaders = DataloaderCreator(num_workers=2)(**datasets)
        adapter.run(dataloaders=dataloaders, epoch_length=10)
