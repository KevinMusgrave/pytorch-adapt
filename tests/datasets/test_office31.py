import unittest

from pytorch_adapt.datasets import Office31, Office31Full

from .. import DATASET_FOLDER, RUN_DATASET_TESTS
from .utils import check_train_test_matches_full, simple_transform, skip_reason


class TestOffice31(unittest.TestCase):
    @unittest.skipIf(not RUN_DATASET_TESTS, skip_reason)
    def test_office31(self):
        check_train_test_matches_full(
            self,
            31,
            ["amazon", "dslr", "webcam"],
            Office31Full,
            Office31,
            DATASET_FOLDER,
        )

    @unittest.skipIf(not RUN_DATASET_TESTS, skip_reason)
    def test_office31_full(self):
        all_d = [
            Office31Full(DATASET_FOLDER, d, simple_transform())
            for d in ["amazon", "dslr", "webcam"]
        ]
        all_d = [len(d) for d in all_d]
        self.assertTrue(sum(all_d) == 4110)
