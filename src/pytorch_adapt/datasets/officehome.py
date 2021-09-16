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
            os.path.join(root, "OfficeHomeDataset_10072016", domain),
            transform=self.transform,
        )
        check_length(
            self, {"Art": 2427, "Clipart": 4365, "Product": 4439, "Real": 4357}[domain]
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
        labels_file = os.path.join(
            root, "OfficeHomeDataset_10072016", f"{domain}_{name}.txt"
        )
        img_dir = os.path.join(root, "OfficeHomeDataset_10072016")

        with open(labels_file) as f:
            content = [line.rstrip().split(" ") for line in f]
        self.img_paths = [os.path.join(img_dir, x[0]) for x in content]
        check_img_paths(img_dir, self.img_paths, domain)
        check_length(
            self,
            {
                "Art": {"train": 1941, "test": 486}[name],
                "Clipart": {"train": 3492, "test": 873}[name],
                "Product": {"train": 3551, "test": 888}[name],
                "Real": {"train": 3485, "test": 872}[name],
            }[domain],
        )
        self.labels = [int(x[1]) for x in content]
        self.transform = transform
