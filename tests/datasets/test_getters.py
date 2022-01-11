import shutil
import unittest

from torchvision.datasets import MNIST

from pytorch_adapt.datasets import (
    MNISTM,
    Office31,
    OfficeHome,
    get_mnist_mnistm,
    get_office31,
    get_officehome,
)

from .. import DATASET_FOLDER, RUN_DATASET_TESTS
from .utils import skip_reason


class TestGetters(unittest.TestCase):
    def helper(self, datasets, src_class, target_class, sizes):
        for k in ["src_train", "src_val"]:
            self.assertTrue(isinstance(datasets[k].dataset.datasets[0], src_class))
            self.assertTrue(len(datasets[k]) == sizes[k])
        for k in ["target_train", "target_val"]:
            self.assertTrue(isinstance(datasets[k].dataset.datasets[0], target_class))
            self.assertTrue(len(datasets[k]) == sizes[k])

    @unittest.skipIf(not RUN_DATASET_TESTS, skip_reason)
    def test_get_mnist_mnistm(self):
        datasets = get_mnist_mnistm(
            ["mnist"], ["mnistm"], folder=DATASET_FOLDER, download=True
        )
        self.helper(
            datasets,
            MNIST,
            MNISTM,
            {
                "src_train": 60000,
                "src_val": 10000,
                "target_train": 59001,
                "target_val": 9001,
            },
        )
        shutil.rmtree(DATASET_FOLDER)

    @unittest.skipIf(not RUN_DATASET_TESTS, skip_reason)
    def test_officehome(self):
        datasets = get_officehome(
            ["art"], ["product"], folder=DATASET_FOLDER, download=True
        )
        self.helper(
            datasets,
            OfficeHome,
            OfficeHome,
            {
                "src_train": 1941,
                "src_val": 486,
                "target_train": 3551,
                "target_val": 888,
            },
        )
        shutil.rmtree(DATASET_FOLDER)

    @unittest.skipIf(not RUN_DATASET_TESTS, skip_reason)
    def test_office31(self):
        datasets = get_office31(
            ["amazon"], ["webcam"], folder=DATASET_FOLDER, download=True
        )
        self.helper(
            datasets,
            Office31,
            Office31,
            {
                "src_train": 2253,
                "src_val": 564,
                "target_train": 636,
                "target_val": 159,
            },
        )

        datasets = get_office31(
            ["amazon", "dslr"], ["webcam"], folder=DATASET_FOLDER, download=True
        )
        self.helper(
            datasets,
            Office31,
            Office31,
            {
                "src_train": 2253 + 398,
                "src_val": 564 + 100,
                "target_train": 636,
                "target_val": 159,
            },
        )

        shutil.rmtree(DATASET_FOLDER)
