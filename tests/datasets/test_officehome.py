import unittest

from pytorch_adapt.datasets import OfficeHome, OfficeHomeFull

from .. import DATASET_FOLDER, RUN_DATASET_TESTS
from .utils import check_train_test_matches_full, simple_transform, skip_reason


class TestOfficeHome(unittest.TestCase):
    @unittest.skipIf(not RUN_DATASET_TESTS, skip_reason)
    def test_officehome(self):
        check_train_test_matches_full(
            self,
            65,
            ["art", "clipart", "product", "real"],
            OfficeHomeFull,
            OfficeHome,
            DATASET_FOLDER,
        )

    @unittest.skipIf(not RUN_DATASET_TESTS, skip_reason)
    def test_officehome_full(self):
        all_d = [
            OfficeHomeFull(DATASET_FOLDER, d, simple_transform())
            for d in ["art", "clipart", "product", "real"]
        ]
        all_d = [len(d) for d in all_d]
        self.assertTrue(sum(all_d) == 15588)
