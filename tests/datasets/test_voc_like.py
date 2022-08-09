import unittest

from pytorch_adapt.datasets import Clipart1kMultiLabel, VOCMultiLabel

from .. import DATASET_FOLDER, RUN_DATASET_TESTS
from .utils import loop_through_dataset, simple_transform, skip_reason


class TestVOCLike(unittest.TestCase):
    @unittest.skipIf(not RUN_DATASET_TESTS, skip_reason)
    def test_voc_multilabel(self):
        transform = simple_transform()
        dataset = VOCMultiLabel(root=DATASET_FOLDER, transform=transform, download=True)
        loop_through_dataset(dataset)

    @unittest.skipIf(not RUN_DATASET_TESTS, skip_reason)
    def test_clipart1k_multilabel(self):
        transform = simple_transform()
        dataset = Clipart1kMultiLabel(
            root=DATASET_FOLDER, transform=transform, download=True
        )
        loop_through_dataset(dataset)
