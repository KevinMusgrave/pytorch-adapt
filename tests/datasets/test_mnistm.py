import unittest

import torch
import tqdm
from torchvision import transforms as torch_transforms

from pytorch_adapt.datasets import MNISTM
from pytorch_adapt.utils.constants import IMAGENET_MEAN, IMAGENET_STD

from .. import DATASET_FOLDER, RUN_DATASET_TESTS
from .utils import skip_reason


class TestMNISTM(unittest.TestCase):
    @unittest.skipIf(not RUN_DATASET_TESTS, skip_reason)
    def test_mnistm(self):
        transform = torch_transforms.Compose(
            [
                torch_transforms.ToTensor(),
                torch_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        for train, length in [(True, 59001), (False, 9001)]:
            dataset = MNISTM(DATASET_FOLDER, train, transform)
            self.assertTrue(len(dataset) == length)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=128, num_workers=4
            )
            for _ in tqdm.tqdm(dataloader):
                pass

        train = MNISTM(DATASET_FOLDER, True, transform)
        test = MNISTM(DATASET_FOLDER, False, transform)
        self.assertTrue(set(train.img_paths).isdisjoint(set(test.img_paths)))
