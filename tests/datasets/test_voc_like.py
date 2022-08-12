import unittest

from pytorch_adapt.datasets import Clipart1kMultiLabel, VOCMultiLabel

from .. import DATASET_FOLDER, RUN_DATASET_TESTS
from .utils import loop_through_dataset, simple_detection_transform, skip_reason


class TestVOCLike(unittest.TestCase):
    @unittest.skipIf(not RUN_DATASET_TESTS, skip_reason)
    def test_voc_multilabel(self):
        transforms = simple_detection_transform()
        for image_set in ["train", "val"]:
            dataset = VOCMultiLabel(
                root=DATASET_FOLDER,
                image_set=image_set,
                transforms=transforms,
                download=True,
            )
            self.assertTrue({"train": 5717, "val": 5823}[image_set])
            loop_through_dataset(dataset)

    @unittest.skipIf(not RUN_DATASET_TESTS, skip_reason)
    def test_clipart1k_multilabel(self):
        transforms = simple_detection_transform()
        for image_set in ["train", "test"]:
            dataset = Clipart1kMultiLabel(
                root=DATASET_FOLDER,
                image_set=image_set,
                transforms=transforms,
                download=True,
            )
            loop_through_dataset(dataset)
