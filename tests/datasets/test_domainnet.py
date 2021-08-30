import os
import unittest
from collections import defaultdict

import torch
import tqdm
from torchvision import transforms as torch_transforms

from pytorch_adapt.datasets import DomainNet, DomainNet126, DomainNet126Full
from pytorch_adapt.utils.constants import IMAGENET_MEAN, IMAGENET_STD

from .. import DATASET_FOLDER, RUN_DATASET_TESTS
from .utils import skip_reason


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
            for d in [train, test]:
                self.assertTrue(len(set(d.img_paths)) == len(d.img_paths))
                self.assertTrue(len(set(d.labels)) == num_classes)
                self.assert_classnames_match_labels(d)

            self.assertTrue(set(train.img_paths).isdisjoint(set(test.img_paths)))
            dataset = torch.utils.data.ConcatDataset([train, test])
            self.assertTrue(len(dataset) == length)

            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=128, num_workers=4
            )
            for _ in tqdm.tqdm(dataloader):
                pass

    @unittest.skipIf(not RUN_DATASET_TESTS, skip_reason)
    def test_domainnet126(self):
        transform = torch_transforms.Compose(
            [
                torch_transforms.Resize((2, 2)),
                torch_transforms.ToTensor(),
            ]
        )

        num_classes = 126
        for domain in ["clipart", "painting", "real", "sketch"]:
            full = DomainNet126Full(DATASET_FOLDER, domain, transform)
            train = DomainNet126(DATASET_FOLDER, domain, True, transform)
            test = DomainNet126(DATASET_FOLDER, domain, False, transform)
            for d in [train, test]:
                self.assertTrue(len(set(d.img_paths)) == len(d.img_paths))
                self.assertTrue(len(set(d.labels)) == num_classes)
                self.assert_classnames_match_labels(d)

            self.assertTrue(set(train.img_paths).isdisjoint(set(test.img_paths)))
            dataset = torch.utils.data.ConcatDataset([train, test])
            self.assertTrue(len(dataset) == len(full))

            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=128, num_workers=4
            )
            for _ in tqdm.tqdm(dataloader):
                pass

    @unittest.skipIf(not RUN_DATASET_TESTS, skip_reason)
    def test_domainnet126_full(self):
        transform = torch_transforms.Compose(
            [
                torch_transforms.Resize((2, 2)),
                torch_transforms.ToTensor(),
            ]
        )
        num_classes = 126
        for domain in ["clipart", "painting", "real", "sketch"]:
            d = DomainNet126Full(DATASET_FOLDER, domain, transform)
            self.assertTrue(len(set(d.img_paths)) == len(d.img_paths))
            self.assertTrue(len(set(d.labels)) == num_classes)
            self.assert_classnames_match_labels(d)

            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=128, num_workers=4
            )
            for _ in tqdm.tqdm(dataloader):
                pass

    def assert_classnames_match_labels(self, d):
        classes = [x.replace(DATASET_FOLDER, "").split(os.sep)[3] for x in d.img_paths]
        classes_to_labels = defaultdict(list)
        for i, classname in enumerate(classes):
            classes_to_labels[classname].append(d.labels[i])
        self.assertTrue(all(len(set(x)) == 1 for x in classes_to_labels.values()))
