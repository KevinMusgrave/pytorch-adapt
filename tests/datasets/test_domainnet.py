import unittest

from torchvision import transforms as torch_transforms

from pytorch_adapt.datasets import DomainNet, DomainNet126, DomainNet126Full

from .. import DATASET_FOLDER, RUN_DATASET_TESTS
from .utils import (
    check_full,
    check_train_test_disjoint,
    check_train_test_matches_full,
    loop_through_dataset,
    skip_reason,
)


class TestDomainNet(unittest.TestCase):
    @unittest.skipIf(not RUN_DATASET_TESTS, skip_reason)
    def test_domainnet(self):
        transform = torch_transforms.Compose(
            [
                torch_transforms.Resize((2, 2)),
                torch_transforms.ToTensor(),
            ]
        )
        for domain, length, num_classes in [
            ("clipart", 48129, 345),
            ("infograph", 51605, 345),
            ("painting", 72266, 344),
            ("quickdraw", 172500, 345),
            ("real", 172947, 345),
            ("sketch", 69128, 345),
        ]:
            train = DomainNet(DATASET_FOLDER, domain, True, transform)
            test = DomainNet(DATASET_FOLDER, domain, False, transform)
            dataset = check_train_test_disjoint(
                self, num_classes, train, test, DATASET_FOLDER
            )
            self.assertTrue(len(dataset) == length)
            loop_through_dataset(dataset)

    @unittest.skipIf(not RUN_DATASET_TESTS, skip_reason)
    def test_domainnet126(self):
        check_train_test_matches_full(
            self,
            126,
            ["clipart", "painting", "real", "sketch"],
            DomainNet126Full,
            DomainNet126,
            DATASET_FOLDER,
        )

    @unittest.skipIf(not RUN_DATASET_TESTS, skip_reason)
    def test_domainnet126_full(self):
        check_full(
            self,
            126,
            ["clipart", "painting", "real", "sketch"],
            DomainNet126Full,
            DATASET_FOLDER,
        )
