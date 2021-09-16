import os

import torch
from torchvision import datasets as torch_datasets

from .base_dataset import BaseDataset
from .utils import check_img_paths, check_length


class OfficeHomeFull(BaseDataset):
    def __init__(self, root: str, domain: str, transform):
        super().__init__(domain=domain)
        self.transform = transform
        self.dataset = torch_datasets.ImageFolder(
            os.path.join(root, "officehome", domain),
            transform=self.transform,
        )
        check_length(
            self, {"art": 2427, "clipart": 4365, "product": 4439, "real": 4357}[domain]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class OfficeHome(BaseDataset):
    def __init__(self, root: str, domain: str, train: bool, transform, **kwargs):
        super().__init__(domain=domain, **kwargs)
        if not isinstance(train, bool):
            raise TypeError("train should be True or False")
        name = "train" if train else "test"
        labels_file = os.path.join(root, "officehome", f"{domain}_{name}.txt")
        img_dir = os.path.join(root, "officehome")

        with open(labels_file) as f:
            content = [line.rstrip().split(" ") for line in f]
        self.img_paths = [os.path.join(img_dir, x[0]) for x in content]
        check_img_paths(img_dir, self.img_paths, domain)
        check_length(
            self,
            {
                "art": {"train": 1941, "test": 486}[name],
                "clipart": {"train": 3492, "test": 873}[name],
                "product": {"train": 3551, "test": 888}[name],
                "real": {"train": 3485, "test": 872}[name],
            }[domain],
        )
        self.labels = [int(x[1]) for x in content]
        self.transform = transform
