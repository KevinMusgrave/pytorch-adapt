import shutil
import unittest

from torchvision.datasets import MNIST

from pytorch_adapt.datasets import MNISTM, OfficeHome, get_mnist_mnistm, get_officehome

from .. import DATASET_FOLDER


class TestGetters(unittest.TestCase):
    def helper(self, datasets, src_class, target_class, sizes):
        for k in ["src_train", "src_val"]:
            self.assertTrue(isinstance(datasets[k].dataset.datasets[0], src_class))
            self.assertTrue(len(datasets[k]) == sizes[k])
        for k in ["target_train", "target_val"]:
            self.assertTrue(isinstance(datasets[k].dataset.datasets[0], target_class))
            self.assertTrue(len(datasets[k]) == sizes[k])

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
