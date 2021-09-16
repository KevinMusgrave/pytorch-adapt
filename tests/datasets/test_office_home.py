import unittest

from pytorch_adapt.datasets import OfficeHome, OfficeHomeFull

from .. import DATASET_FOLDER, RUN_DATASET_TESTS
from .utils import check_train_test_matches_full, simple_transform, skip_reason


class TestOfficeHome(unittest.TestCase):
    @unittest.skipIf(not RUN_DATASET_TESTS, skip_reason)
    def test_office_home(self):
        check_train_test_matches_full(
            self,
            65,
            ["Art", "Clipart", "Product", "Real"],
            OfficeHomeFull,
            OfficeHome,
            DATASET_FOLDER,
        )

    @unittest.skipIf(not RUN_DATASET_TESTS, skip_reason)
    def test_office_home_full(self):
        all_d = [
            OfficeHomeFull(DATASET_FOLDER, d, simple_transform())
            for d in ["Art", "Clipart", "Product", "Real"]
        ]
        all_d = [len(d) for d in all_d]
        self.assertTrue(sum(all_d) == 15588)
