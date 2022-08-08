import unittest

from pytorch_adapt.datasets import VOCMultiLabel

from .. import DATASET_FOLDER, RUN_DATASET_TESTS
from .utils import skip_reason


class TestVOCMultiLabel(unittest.TestCase):
    @unittest.skipIf(not RUN_DATASET_TESTS, skip_reason)
    def test_voc_multilabel(self):
        dataset = VOCMultiLabel(root=DATASET_FOLDER, download=True)
