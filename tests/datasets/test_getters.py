import shutil
import unittest

from torchvision.datasets import MNIST

from pytorch_adapt.datasets import MNISTM, get_mnist_mnistm

from .. import DATASET_FOLDER


class TestGetters(unittest.TestCase):
    def test_get_mnist_mnistm(self):
        datasets = get_mnist_mnistm(
            ["mnist"], ["mnistm"], folder=DATASET_FOLDER, download=True
        )
        for k in ["src_train", "src_val"]:
            self.assertTrue(isinstance(datasets[k].dataset.datasets[0], MNIST))
            self.assertTrue(len(datasets[k]) == (60000 if k == "src_train" else 10000))
        for k in ["target_train", "target_val"]:
            self.assertTrue(isinstance(datasets[k].dataset.datasets[0], MNISTM))
            self.assertTrue(
                len(datasets[k]) == (59001 if k == "target_train" else 9001)
            )
        shutil.rmtree(DATASET_FOLDER)
