import unittest

from torchvision.datasets import MNIST

from pytorch_adapt.datasets import (
    MNISTM,
    Office31,
    OfficeHome,
    SourceDataset,
    TargetDataset,
    get_mnist_mnistm,
    get_office31,
    get_officehome,
)

from .. import DATASET_FOLDER, RUN_DATASET_TESTS
from .utils import skip_reason


class TestGetters(unittest.TestCase):
    def helper(
        self, datasets, src_class, target_class, sizes, target_with_labels=False
    ):
        for k in ["src_train", "src_val"]:
            self.assertTrue(isinstance(datasets[k].dataset.datasets[0], src_class))
            self.assertTrue(isinstance(datasets[k], SourceDataset))
            self.assertTrue(len(datasets[k]) == sizes[k])

        target_splits = ["target_train", "target_val"]
        if target_with_labels:
            target_splits += ["target_train_with_labels", "target_val_with_labels"]
        target_sizes = {k for k in sizes.keys() if k.startswith("target")}

        self.assertTrue(set(target_splits) == target_sizes)
        for k in target_splits:
            self.assertTrue(isinstance(datasets[k].dataset.datasets[0], target_class))
            self.assertTrue(isinstance(datasets[k], TargetDataset))
            self.assertTrue(len(datasets[k]) == sizes[k])
            self.assertTrue(datasets[k].supervised == k.endswith("with_labels"))

    @unittest.skipIf(not RUN_DATASET_TESTS, skip_reason)
    def test_empty_array(self):
        datasets = get_mnist_mnistm(["mnist"], [], folder=DATASET_FOLDER, download=True)
        self.assertTrue(datasets.keys() == {"src_train", "src_val", "train"})
        self.assertTrue(len(datasets["train"]) == 60000)
        self.assertTrue(isinstance(datasets["train"], SourceDataset))

        datasets = get_mnist_mnistm(
            [], ["mnistm"], folder=DATASET_FOLDER, download=True
        )
        self.assertTrue(datasets.keys() == {"target_train", "target_val", "train"})
        self.assertTrue(len(datasets["train"]) == 59001)
        self.assertTrue(isinstance(datasets["train"], TargetDataset))

    @unittest.skipIf(not RUN_DATASET_TESTS, skip_reason)
    def test_get_mnist_mnistm(self):
        for target_with_labels in [False, True]:
            datasets = get_mnist_mnistm(
                ["mnist"],
                ["mnistm"],
                folder=DATASET_FOLDER,
                download=True,
                return_target_with_labels=target_with_labels,
            )
            len_dict = {
                "src_train": 60000,
                "src_val": 10000,
                "target_train": 59001,
                "target_val": 9001,
            }
            if target_with_labels:
                len_dict["target_train_with_labels"] = len_dict["target_train"]
                len_dict["target_val_with_labels"] = len_dict["target_val"]

            self.helper(datasets, MNIST, MNISTM, len_dict, target_with_labels)

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
